#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import rospy
import rrt
import sys
import time

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# Occupancy grid.
from nav_msgs.msg import OccupancyGrid
# Position.
from tf import TransformListener
from geometry_msgs.msg import Point
# Path.
from nav_msgs.msg import Path
# Laser scan message:
# http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html
from sensor_msgs.msg import PointCloud2, LaserScan
# For groundtruth information.
from gazebo_msgs.msg import ModelStates
# For visualising the centroid.
from visualization_msgs.msg import Marker
# For pose information.
from tf.transformations import euler_from_quaternion
# For clustering the points
from sklearn.cluster import DBSCAN
# For colouring clusters
from std_msgs.msg import ColorRGBA
# For getting distances between centroids
from sklearn.metrics.pairwise import euclidean_distances

ROBOT_RADIUS = 0.105 / 2.

CLUSTER_COLOURS = [
    ColorRGBA(140.0, 224.0, 255.0, 1.0),
    ColorRGBA(12.0, 227.0, 208.0, 1.0),
    ColorRGBA(252.0, 255.0, 89.0, 1.0),
    ColorRGBA(127.0, 31.0, 127.0, 1.0),
    ColorRGBA(232.0, 163.0, 74.0, 1.0)]

SPEED = 0.2

X = 0
Y = 1
YAW = 2

def cap(v, max_speed):
  n = np.linalg.norm(v)
  if n > max_speed:
    return v / n * max_speed
  return v


def rotate(v, a):
  rotation = np.array([[np.cos(a), -np.sin(a)],
                      [np.sin(a), np.cos(a)]])
  return np.dot(rotation, v)


class SLAM(object):
  def __init__(self, name):
    rospy.Subscriber(name + '/map', OccupancyGrid, self.callback)
    self.name = name
    self._tf = TransformListener()
    self._occupancy_grid = None
    self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    
  def callback(self, msg):
    values = np.array(msg.data, dtype=np.int8).reshape((msg.info.width, msg.info.height))
    processed = np.empty_like(values)
    processed[:] = rrt.FREE
    processed[values < 0] = rrt.UNKNOWN
    processed[values > 50] = rrt.OCCUPIED
    processed = processed.T
    origin = [msg.info.origin.position.x, msg.info.origin.position.y, 0.]
    resolution = msg.info.resolution
    self._occupancy_grid = rrt.OccupancyGrid(processed, origin, resolution)

  def update(self):
    # Get pose w.r.t. map.
    a = 'occupancy_grid'
    b = self.name + '/base_link'                                        
    try:
        t = rospy.Time(0)
        self._tf.waitForTransform('/' + a, '/' + b, t, rospy.Duration(4.0))
        position, orientation = self._tf.lookupTransform('/' + a, '/' + b, t)
        self._pose[X] = position[X]
        self._pose[Y] = position[Y]
        _, _, self._pose[YAW] = euler_from_quaternion(orientation)
    except Exception as e:
        print(e)
    pass

  @property
  def ready(self):
    return self._occupancy_grid is not None and not np.isnan(self._pose[0])

  @property
  def pose(self):
    return self._pose

  @property
  def occupancy_grid(self):
    return self._occupancy_grid


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
        self.name = name
        self.vel_msg = Twist()
        self.slam = SLAM(name)
        with open('/tmp/gazebo_exercise_' + name + '.txt', 'w'):
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

        self.vel_msg.linear.x = u / 10.
        self.vel_msg.angular.z = w / 15.

    def write_pose(self):        
        with open('/tmp/gazebo_exercise_' + self.name + '.txt', 'a') as fp:
            fp.write('\n'.join(','.join(str(v) for v in p)
                                for p in self.pose_history) + '\n')
            self.pose_history = []


class Laser(object):
    def __init__(self, name, min_angle, max_angle, max_distance):
        rospy.Subscriber('/' + name + '/scan', LaserScan, self.callback)
        # For publishing the clusters
        self._clusters_publishers = [
            rospy.Publisher('/' + name + '/cluster0', Marker, queue_size=5),
            rospy.Publisher('/' + name + '/cluster1', Marker, queue_size=5),
            rospy.Publisher('/' + name + '/cluster2', Marker, queue_size=5),
            rospy.Publisher('/' + name + '/cluster3', Marker, queue_size=5),
            rospy.Publisher('/' + name + '/cluster4', Marker, queue_size=5)]
        # For publishing the centroid
        self._centroid_publisher = rospy.Publisher('/' + name + '/centroid', Marker, queue_size=1)

        # Name of the robot to which the laser belongs
        self._name = name

        # Take measurements between min_angle and max_angle at distances less than max_distance
        self._min_angle = min_angle
        self._max_angle = max_angle
        self._max_distance = max_distance

        # Distance measurements and angles at which they were measured
        # i.e. (measurements[0], angles[0]), (measurements[1], angles[1]), ...
        self._angles = []
        self._measurements = []

        # PointCloud of the laser measurements between min_angle and max_angle
        # point_cloud[i] is a point [x, y, z]
        self._point_cloud = []

        # Clusters of points generated by the laser measurements
        self._clusters = []

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

        # At each callback, take all distances measured between _min_angle and
        # _max_angle that are less than _max_distance
        self._measurements = []
        self._angles = []
        for i, d in enumerate(msg.ranges):
            # Angle at which the distance d was measured
            angle = msg.angle_min + i * msg.angle_increment
            if not np.isnan(d) and not np.isinf(d) and _within(angle, self._min_angle, self._max_angle) and d < self._max_distance:
                self._angles.append(angle)
                self._measurements.append(d)
        
        # Generate PointCloud of the laser measurements between _min_angle and _max_angle
        # that are less than _max_distance. Points are relative to the robot (base_link)
        points = []
        for i, d in enumerate(self._measurements):
            point = [
                d * np.cos(self._angles[i] % (2 * np.pi)),
                d * np.sin(self._angles[i] % (2 * np.pi)),
                0.]
            if (not point == [0., 0., 0.]):
                points.append(point)
        self._point_cloud = np.array(points, dtype=np.float32)
        point_cloud = self._point_cloud

        self._clusters = []
        if len(point_cloud) != 0:
            # DBSCAN clustering
            model = DBSCAN(eps=0.2, min_samples=2)
            labels = model.fit_predict(point_cloud)
            clusters = []
            for i in range(0, np.max(labels) + 1):
                clusters.append(point_cloud[np.nonzero(labels == i)])
            self._clusters = np.array(clusters)

    def set_angle(self, angle):
        self._min_angle = angle - (np.pi / 6.)
        self._max_angle = angle + (np.pi / 6.)

    @property
    def ready(self):
        return len(self._measurements) == 0 or not np.isnan(self._measurements[0])

    @property
    def measurements(self):
        return self._measurements
    
    @property
    def angles(self):
        return self._angles

    @property
    def point_cloud(self):
        return self._point_cloud

    @property
    def get_clusters(self):
        return self._clusters
    
    @property
    # Returns the centroids of all clusters relative to the robot (base link)
    def get_centroids(self):
        clusters = self._clusters

        # Publish clusters using markers
        for i in range(0, len(clusters)):
            cluster = clusters[i]
            pc = Marker()
            pc.header.frame_id = '/' + self._name + '/base_link'
            pc.type = pc.POINTS
            pc.action = pc.ADD
            pc.pose.orientation.w = 1
            pc.points = []
            for point in cluster:
                pc.points.append(Point(point[0], point[1], point[2]))
            t = rospy.Duration()
            pc.lifetime = t
            pc.scale.x = 0.05
            pc.scale.y = 0.05
            pc.color = CLUSTER_COLOURS[i]
            self._clusters_publishers[i].publish(pc)
        
        if len(clusters) == 0:
            # No clusters, so return no centroid
            return np.array([], dtype=np.float32)
        
        # Get centroids of all clusters
        centroids = np.array([np.mean(cluster, axis=0)[:-1] for cluster in clusters], dtype=np.float32)

        print("Centroids for " + self._name + " in get_centroids:")
        print(centroids)

        return centroids


class Leader(Robot):
    def __init__(self, name):
        super(Leader, self).__init__(name)
        self.laser = LeaderLaser(name=name)
        # For feedback linearization
        self.epsilon = 0.2
    
    def update_velocities(self, rate_limiter):
        if not self.laser.ready or not self.slam.ready:
            return
        
        # Get point to follow relative to the robot (base_link)
        point_to_follow = self.laser.point_to_follow

        if np.linalg.norm(point_to_follow) < 0.5:
            # We have reached the desired position, so stop
            self.vel_msg.linear.x = 0.
            self.vel_msg.angular.z = 0.
        else:
            # Go towards the desired position using feedback linearisation
            self.linearised_feedback(point_to_follow)
        
        self.publisher.publish(self.vel_msg)
        self.pose_history.append(self.slam.pose)
        if not len(self.pose_history) % 10:
            self.write_pose()
    
    def linearised_feedback(self, velocity):
        velocity = cap(velocity, SPEED)

        # Feedback linearisation with relative positions
        u = velocity[X]
        w = velocity[Y] / self.epsilon

        self.vel_msg.linear.x =  u
        self.vel_msg.angular.z = w


class LeaderLaser(Laser):
    def __init__(self, name):
        super(LeaderLaser, self).__init__(name=name, min_angle=-np.pi/6., max_angle=np.pi/6., max_distance=2.)

    @property
    def point_to_follow(self):
        # Get centroids of all clusters
        centroids = self.get_centroids

        # Point to follow
        follow = np.array([0., 0.], dtype=np.float32)

        if len(centroids) == 0:
            # No cluster, so stop
            follow = [0., 0.]
        elif len(centroids) == 1:
            # Only one cluster, so follow that centroid
            follow = centroids[0]
        else:
            # At least 2 clusters, so get the 2 closest ones
            distance_matrix = euclidean_distances(centroids)
            # Non-zero values of the distance matrix
            non_zero_distance_matrix = distance_matrix[distance_matrix > 0.]
            # Get minimum non-zero value in the matrix
            min_distance = np.min(non_zero_distance_matrix)
            # Get position in matrix of the minimun non-zero distance
            pos = np.argwhere(distance_matrix == min_distance)[0]
            # Get the 2 centroids the distance corresponds to
            c1, c2 = centroids[pos[0]], centroids[pos[1]]
            # Follow the midpoint
            follow = (c1 + c2) / 2.
        
        # Publish the point to follow using a marker
        marker = Marker()
        marker.header.frame_id = "/t0/base_link"
        marker.type = marker.POINTS
        marker.action = marker.ADD
        marker.pose.orientation.w = 1
        p = Point(follow[0], follow[1], 0.)
        marker.points = [p]
        t = rospy.Duration()
        marker.lifetime = t
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        self._centroid_publisher.publish(marker)

        return np.array(follow, dtype=np.float32)

class Follower(Robot):
    def __init__(self, name, leader, leader_relative_position, desired_range_and_bearing):
        super(Follower, self).__init__(name)
        
        self.laser = FollowerLaser(name=name, min_angle=-np.pi/4., max_angle=np.pi/4., max_distance=1.5, leader=leader, leader_relative_position=leader_relative_position, desired_range_and_bearing=desired_range_and_bearing)

        # The leader this robot needs to follow
        self.leader = leader

    def update_velocities(self, rate_limiter):
        if not self.laser.ready or not self.slam.ready:
            rate_limiter.sleep()
            return
        
        controls = self.laser.get_controls
        self.vel_msg.linear.x = controls[0][0]
        self.vel_msg.angular.z = controls[1][0]

        self.publisher.publish(self.vel_msg)
        self.pose_history.append(self.slam.pose)
        if not len(self.pose_history) % 10:
            self.write_pose()
        
    def linearised_feedback(self, velocity):
        print("Moving a tiny bit...")
        velocity = cap(velocity, SPEED)

        # Feedback linearisation with relative positions
        u = velocity[X]
        w = velocity[Y] / self.epsilon

        self.vel_msg.linear.x = u * 1.3
        self.vel_msg.angular.z = w

    def obstacle(self, front, front_left, front_right, left, right):
        # return min([front, front_left, front_right]) < 0.5
        return False


class FollowerLaser(Laser):
    def __init__(self, name, min_angle, max_angle, max_distance, leader, leader_relative_position, desired_range_and_bearing):
        super(FollowerLaser, self).__init__(name=name, min_angle=min_angle, max_angle=max_angle, max_distance=max_distance)

        # Publisher for the leader's centroid as sensed by this robot's laser
        self._centroid_publisher = rospy.Publisher('/' + name + '/centroid', Marker, queue_size=1)

        # Publisher for the last leader movement
        self._leader_movement_publisher = rospy.Publisher('/' + name + '/movement', Marker, queue_size=1)

        # The leader this robot needs to follow
        self.leader = leader

        # Last sensed leader position relative to this follower robot
        # Start with predefined position
        self._last_leader_position = leader_relative_position

        # Constant to use in range and bearing method
        self._k = np.array([[.6],[.5]], dtype=np.float32)

        # Desired range and bearing to be achieved
        self._z_ij_d = desired_range_and_bearing

    @property
    # Returns linear and angular velocity to use next
    def get_controls(self):
        def get_closest_centroid_to_robot(centroids):
            distances_to_centroids = np.sum(centroids ** 2, axis=1)
            return centroids[np.argmin(distances_to_centroids)]
    
        # Get centroids of all clusters
        centroids = self.get_centroids
        
        if len(centroids) == 0:
            # Make leader centroid just yourself, so that you stop moving
            leader_centroid = np.array([0., 0.], dtype=np.float32)
        elif len(centroids) == 1:
            # Only one cluster, so get that centroid
            leader_centroid = centroids[0]
        else:
            leader_centroid = get_closest_centroid_to_robot(centroids)

        # Publish leader centroid
        marker = Marker()
        marker.header.frame_id = '/' + self._name + "/base_link"
        marker.type = marker.POINTS
        marker.action = marker.ADD
        marker.pose.orientation.w = 1
        p = Point(leader_centroid[X], leader_centroid[Y], 0.)
        marker.points = [p]
        t = rospy.Duration()
        marker.lifetime = t
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        if self._name == "t1":
            marker.color.b = 1.0
        if self._name == "t2":
            marker.color.g = 1.0
        self._centroid_publisher.publish(marker)

        if np.linalg.norm(leader_centroid) == 0.:
            # Stop the robot (send 0-controls)
            return np.array([[0.],[0.]], dtype=np.float32)
        
        movement_of_leader = leader_centroid - self._last_leader_position

        # Publish last movement of leader
        marker = Marker()
        marker.header.frame_id = '/' + self._name + "/base_link"
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.pose.orientation.w = 1
        p1 = Point(self._last_leader_position[X], self._last_leader_position[Y], 0.)
        p2 = Point(leader_centroid[X], leader_centroid[Y], 0.)
        marker.points = [p1, p2]
        t = rospy.Duration()
        marker.lifetime = t
        marker.scale.x = 0.02
        marker.scale.y = 0.04
        marker.scale.z = 0.02
        marker.color.a = 1.0
        if self._name == "t1":
            marker.color.a = 1.0
        if self._name == "t2":
            marker.color.g = 1.0
        self._leader_movement_publisher.publish(marker)

        # Relative angle of leader w.r.t. the follower
        beta_ij = np.arctan2(movement_of_leader[X], movement_of_leader[Y])
        # Distance from robot to leader
        l_ij = np.linalg.norm(leader_centroid)
        # Orientation of leader centroid relative to the robot
        alpha = np.arctan2(leader_centroid[Y], leader_centroid[X])
        
        psi_ij = np.pi + alpha - beta_ij
        gamma_ij = np.pi + alpha

        # Define matrices G and F
        G = np.array([[np.cos(gamma_ij), ROBOT_RADIUS * np.sin(gamma_ij)],[-np.sin(gamma_ij) / l_ij, ROBOT_RADIUS * np.cos(gamma_ij) / ROBOT_RADIUS]], dtype=np.float32)
        F = np.array([[-np.cos(psi_ij), 0.],[np.sin(psi_ij)/ l_ij, -1.]], dtype=np.float32)

        # Current range and bearing
        z_ij = np.array([[l_ij],[psi_ij]], dtype=np.float32)

        # Last registered controls of leader
        u_i = np.array([[self.leader.vel_msg.linear.x],[self.leader.vel_msg.angular.z]], dtype=np.float32)
        # Desired controls (linear and angluar velocities)
        u_j = np.linalg.inv(G) * (self._k * (self._z_ij_d - z_ij) - F * u_i)

        # Update last sensed position of the leader
        self._last_leader_position = leader_centroid

        return u_j


def run(args):
    rospy.init_node('main')
    
    # Control update rate = 100Hz.
    rate_limiter = rospy.Rate(1000)

    # Leader robot
    l = Leader("t0")
    # Follower robot 2 -- leader starts at [.6, -.3] relative to it
    f2 = Follower("t2", l, np.array([.6, -.3], dtype=np.float32), np.array([[0.4],[3.*np.pi/4.]], dtype=np.float32))
    # Follower robot 1 -- follower robot 2 starts at [.6, -.3.] relative to it
    f1 = Follower("t1", f2, np.array([6., -.3], dtype=np.float32), np.array([[0.4],[3.*np.pi/4.]], dtype=np.float32))

    while not rospy.is_shutdown():
        l.slam.update()
        f1.slam.update()
        f2.slam.update()
        # Make sure all measurements are ready.
        l.update_velocities(rate_limiter)
        f1.update_velocities(rate_limiter)
        f2.update_velocities(rate_limiter)
        rate_limiter.sleep()


if __name__ == '__main__':
    args = None
    try:
        run(args)
    except rospy.ROSInterruptException:
        pass
