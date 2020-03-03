#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import rospy

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# Laser scan message:
# http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html
from sensor_msgs.msg import LaserScan
# For groundtruth information.
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion

X = 0
Y = 1
YAW = 2

def angle_between(v1, v2):
  v1_u = v1 / np.linalg.norm(v1)
  v2_u = v2 / np.linalg.norm(v2)
  return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rotate(v, a):
  rotation = np.array([[np.cos(a), -np.sin(a)],
                      [np.sin(a), np.cos(a)]])
  return np.dot(rotation, v)


class SimpleLaser(object):
    def __init__(self, name=""):
        rospy.Subscriber('/'+name+'/scan', LaserScan, self.callback)
        self._angles = [0., np.pi / 4., -np.pi / 4., np.pi / 2., -np.pi / 2.]
        self._width = np.pi / 180. * 10.  # 10 degrees cone of view.
        self._measurements = [float('inf')] * len(self._angles)
        self._indices = None

    def callback(self, msg):
        # Helper for angles.
        def _within(x, a, b):
            pi2 = np.pi * 2.
            x %= pi2
            a %= pi2
            b %= pi2
            if a < b:
                return a <= x and x <= b
            return a <= x or x <= b

        # Compute indices the first time.
        if self._indices is None:
            self._indices = [[] for _ in range(len(self._angles))]
            for i, d in enumerate(msg.ranges):
                angle = msg.angle_min + i * msg.angle_increment
                for j, center_angle in enumerate(self._angles):
                    if _within(angle, center_angle - self._width / 2., center_angle + self._width / 2.):
                        self._indices[j].append(i)

        ranges = np.array(msg.ranges)
        for i, idx in enumerate(self._indices):
            # We do not take the minimum range of the cone but the 10-th percentile for robustness.
            self._measurements[i] = np.percentile(ranges[idx], 10)

    @property
    def ready(self):
        return not np.isnan(self._measurements[0])

    @property
    def measurements(self):
        return self._measurements


class GroundtruthPose(object):
    def __init__(self, name='turtlebot3_burger'):
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)
        self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        self._name = name

    def callback(self, msg):
        idx = [i for i, n in enumerate(msg.name) if n == self._name]
        if not idx:
            raise ValueError(
                'Specified name "{}" does not exist.'.format(self._name))
        idx = idx[0]
        self._pose[0] = msg.pose[idx].position.x
        self._pose[1] = msg.pose[idx].position.y
        _, _, yaw = euler_from_quaternion([
            msg.pose[idx].orientation.x,
            msg.pose[idx].orientation.y,
            msg.pose[idx].orientation.z,
            msg.pose[idx].orientation.w])
             
        self._pose[2] = yaw

    @property
    def ready(self):
        return not np.isnan(self._pose[0])

    @property
    def pose(self):
        return self._pose


class Robot(object):
    def __init__(self, name):
        self.groundtruth = GroundtruthPose(name)
        self.pose_history = []
        self.publisher = rospy.Publisher('/' + name + '/cmd_vel', Twist, queue_size=5)
        self.laser = SimpleLaser(name=name)
        self.name = name
        self.vel_msg = Twist()
        with open('/tmp/gazebo_exercise_'+self.name+'.txt', 'w'):
            pass


    def braitenberg(self, front, front_left, front_right, left, right):
        u = 1.
        w = 0.
        sensors = np.array([1./left, 1./front_left, 1./front, 1./front_right, 1./right])  
        vl_weights = np.array([.5, 1.1, -1.5, -.9, -.5])
        vr_weights = np.array([-.5, -.9, 2.5, 1.1, .5])
        vr = np.dot(vr_weights, sensors)
        vl = np.dot(vl_weights, sensors)
        u = 1 - ((vr + vl) / 2)
        w = (vr - vl)

        self.vel_msg.linear.x = u / 2.
        self.vel_msg.angular.z = w / 6.

    def write_pose(self):        
        with open('/tmp/gazebo_exercise_'+self.name+'.txt', 'a') as fp:
            fp.write('\n'.join(','.join(str(v) for v in p)
                                for p in self.pose_history) + '\n')
            self.pose_history = []


class Leader(Robot):
    def __init__(self, name):
        super(Leader, self).__init__(name)
    
    def update_velocities(self, rate_limiter):
        if not self.laser.ready or not self.groundtruth.ready:
            rate_limiter.sleep()
            return
        self.braitenberg(*self.laser.measurements)
        self.publisher.publish(self.vel_msg)
        self.pose_history.append(self.groundtruth.pose)
        if not len(self.pose_history) % 10:
            self.write_pose()
        return self.vel_msg

    
class Follower(Robot):
    def __init__(self, name, desired_distance, desired_angle):
        super(Follower, self).__init__(name)
        self.dist_d = desired_distance
        self.angle_d = desired_angle
        self.L = self.dist_d * np.cos(self.angle_d)
        self.L_prime = self.dist_d * np.sin(self.angle_d)
        self.k1 = .1
        self.k2 = .1


    def update_velocities(self, rate_limiter, leader_vel, relative_yaw, distance_to, angle_to):
        if not self.laser.ready or not self.groundtruth.ready:
            rate_limiter.sleep()
            return
        self.follow(leader_vel, relative_yaw, distance_to, angle_to)
        self.publisher.publish(self.vel_msg)
        self.pose_history.append(self.groundtruth.pose)
        if not len(self.pose_history) % 10:
            self.write_pose()

    def follow(self, leader_vel, relative_yaw, distance_to, angle_to):
        if self.name == 't1':
            print("Relative YAW: " + str(relative_yaw))
            print("Distance to: " + str(distance_to))
            print("Angle to: " + str(angle_to))
        if leader_vel is not None:
            v_l = leader_vel.linear.x
            omega_l = leader_vel.angular.z

            Phi_bar = relative_yaw

            rho = distance_to
            phi = angle_to

            v_F1 = v_l * np.cos(Phi_bar)
            v_F2 = self.k1 * (rho*np.cos(phi) + self.L_prime * np.cos(Phi_bar - (np.pi / 2.)) - self.L )
            v_F3 = self.dist_d * omega_l * np.sin(self.angle_d + Phi_bar)
            v_F4 = self.L_prime * omega_l * np.sin(Phi_bar - (np.pi / 2.))

            w_F1 = v_l * np.sin(Phi_bar)
            w_F2 = self.k2 * (rho * np.sin(phi) + self.L_prime * np.sin(Phi_bar - (np.pi / 2.)))
            w_F3 = self.L_prime * omega_l * np.cos(Phi_bar - (np.pi / 2.))
            
            # velocity
            v = v_F1 + v_F2 + v_F3 - v_F4
            w = (w_F1 - w_F2 + w_F3) * 1./self.L

            print(v, w)

            self.vel_msg.linear.x = v
            self.vel_msg.angular.z = w


    def linearised_feedback(self, velocity):
        pose = self.groundtruth.pose

        u = velocity[X] * np.cos(pose[YAW]) + velocity[Y] * np.sin(pose[YAW])
        w = (-velocity[X] * np.sin(pose[YAW]) + velocity[Y] * np.cos(pose[YAW])) / self.epsilon

        self.vel_msg.linear.x = u * 1.2
        self.vel_msg.angular.z = w


def run(args):
    rospy.init_node('obstacle_avoidance')
    # avoidance_method = globals()[args.mode]
    
    # Update control every 100 ms.
    rate_limiter = rospy.Rate(100)
    # Leader robot
    l = Leader("t0")
    # Follower robot 1
    f1 = Follower("t1", .2, np.pi/4.)
    # Follower robot 2
    f2 = Follower("t2", .2, -np.pi/4.)


    while not rospy.is_shutdown():
        # Make sure all measurements are ready.
        leader_vel = l.update_velocities(rate_limiter)
        print(leader_vel)
        f1_relative_yaw = l.groundtruth.pose[YAW] - f1.groundtruth.pose[YAW]
        f1_dist = np.linalg.norm(l.groundtruth.pose[:-1] - f1.groundtruth.pose[:-1])
        f1_angle = angle_between(l.groundtruth.pose[:-1], f1.groundtruth.pose[:-1])

        f2_relative_yaw = l.groundtruth.pose[YAW] - f2.groundtruth.pose[YAW]
        f2_dist = np.linalg.norm(l.groundtruth.pose[:-1] - f2.groundtruth.pose[:-1])
        f2_angle = angle_between(l.groundtruth.pose[:-1], f2.groundtruth.pose[:-1])

        f1.update_velocities(rate_limiter, leader_vel, f1_relative_yaw, f1_dist, f1_angle)
        f2.update_velocities(rate_limiter, leader_vel, f2_relative_yaw, f2_dist, f2_angle)

        rate_limiter.sleep()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Runs obstacle avoidance')
    # parser.add_argument('--mode', action='store', default='braitenberg',
    #                     help='Method.', choices=['braitenberg', 'rule_based'])
    # args, unknown = parser.parse_known_args()
    args = None
    try:
        run(args)
    except rospy.ROSInterruptException:
        pass
