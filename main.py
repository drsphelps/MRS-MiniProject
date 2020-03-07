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
    b = 'base_link'                                        
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
        self.groundtruth = GroundtruthPose()
        self.pose_history = []
        self.publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
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


class Leader(Robot):
    def __init__(self, name):
        super(Leader, self).__init__(name)
        self.laser = LeaderLaser(name=name)
        # For feedback linearization
        self.epsilon = 0.1
    
    def update_velocities(self, rate_limiter):
        if not self.laser.ready or not self.slam.ready:
            return
        
        # Get centroid position relative to the robot (base_link)
        centroid = self.laser.centroid

        if np.linalg.norm(centroid) < 0.4:
            # We have reached the desired position, so stop
            self.vel_msg.linear.x = 0.
            self.vel_msg.angular.z = 0.
        else:
            self.linearised_feedback(centroid)
        
        self.publisher.publish(self.vel_msg)
        self.pose_history.append(self.slam.pose)
        if not len(self.pose_history) % 10:
            self.write_pose()
    
    def linearised_feedback(self, velocity):
        velocity = cap(velocity, SPEED)

        # Feedback linearisation with relative positions
        u = velocity[X]
        w = velocity[Y] / self.epsilon

        self.vel_msg.linear.x = u * 1.4
        self.vel_msg.angular.z = w
    

class LeaderLaser(object):
    def __init__(self, name):
        rospy.Subscriber('/scan', LaserScan, self.callback)
        # For publishing the clusters
        self._clusters_publishers = [
            rospy.Publisher('/cluster0', Marker, queue_size=5),
            rospy.Publisher('/cluster1', Marker, queue_size=5),
            rospy.Publisher('/cluster2', Marker, queue_size=5),
            rospy.Publisher('/cluster3', Marker, queue_size=5),
            rospy.Publisher('/cluster4', Marker, queue_size=5)]
        # For publishing the centroid
        self._centroid_publisher = rospy.Publisher('/centroid', Marker, queue_size=1)

        # Take measurements between -pi/6 and pi/6
        self._min_angle = -np.pi / 4.
        self._max_angle = np.pi / 4.

        # Distance measurements and angles at which they were measured
        # i.e. (measurements[0], angles[0]), (measurements[1], angles[1]), ...
        self._angles = []
        self._measurements = []

        # PointCloud of the laser measurements between _min_angle and _max_angle
        # point_cloud[i] is a point [x, y, z]
        self._point_cloud = []

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

        # At each callback, take all distances measured between _min_angle and _max_angle that are less than 1 meters
        self._measurements = []
        self._angles = []
        for i, d in enumerate(msg.ranges):
            # Angle at which the distance d was measured
            angle = msg.angle_min + i * msg.angle_increment
            if not np.isnan(d) and not np.isinf(d) and _within(angle, self._min_angle, self._max_angle) and d < 1.5:
                self._angles.append(angle)
                self._measurements.append(d)
        
        # Generate PointCloud of the laser measurements between _min_angle and _max_angle that are less than 3 meters
        # Points are relative to the robot (base_link)
        points = []
        for i, d in enumerate(self._measurements):
            point = [
                d * np.cos(self._angles[i] % (2 * np.pi)),
                d * np.sin(self._angles[i] % (2 * np.pi)),
                0.]
            if (not point == [0., 0., 0.]):
                points.append(point)
        self._point_cloud = np.array(points, dtype=np.float32)

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
    # Centroid is relative to the robot (base_link)
    def centroid(self):
        point_cloud = self._point_cloud
        if len(point_cloud) == 0:
            return np.array([0., 0.], dtype=np.float32)

        # DBSCAN clustering
        model = DBSCAN(eps=0.1)
        labels = model.fit_predict(point_cloud)
        clusters = []
        for i in range(0, np.max(labels) + 1):
            clusters.append(point_cloud[np.nonzero(labels == i)])
        clusters = np.array(clusters)

        # Publish clusters using markers
        for i in range(0, len(clusters)):
            cluster = clusters[i]
            pc = Marker()
            pc.header.frame_id = "/base_link"
            pc.type = pc.POINTS
            pc.action = pc.ADD
            pc.pose.orientation.w = 1
            for point in cluster:
                pc.points.append(Point(point[0], point[1], point[2]))
            t = rospy.Duration()
            pc.lifetime = t
            pc.scale.x = 0.05
            pc.scale.y = 0.05
            pc.color = CLUSTER_COLOURS[i]
            self._clusters_publishers[i].publish(pc)
        
        # Get centroids of all clusters
        centroids = np.array([np.mean(cluster, axis=0)[:-1] for cluster in clusters])

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
    def __init__(self, name, rel_pos):
        super(Follower, self).__init__(name)
        self.laser = FollowerLaser(name)
        self.relative_position = rel_pos
        self.epsilon = 0.2
        self.path = []

    def update_velocities(self, rate_limiter, leader_pose):
        if not self.laser.ready or not self.groundtruth.ready:
            rate_limiter.sleep()
            return
        if not self.obstacle(*self.laser.measurements):
            self.follow(leader_pose)
        else:
            self.braitenberg(*self.laser.measurements)
        self.publisher.publish(self.vel_msg)
        self.pose_history.append(self.slam.pose)
        if not len(self.pose_history) % 10:
            self.write_pose()

    def follow(self, leader_pose):
        desired_position = leader_pose[:-1] + rotate(self.relative_position[:-1], leader_pose[YAW])
        tow = self.slam.pose[:-1] + rotate(np.array([0.1,0.]), self.slam.pose[YAW])
        vector_to_travel = desired_position - tow
        if np.linalg.norm(vector_to_travel) < 0.2:
            self.vel_msg.linear.x = 0
            self.vel_msg.angular.z = 0
        else:
            self.linearised_feedback(vector_to_travel)

    def linearised_feedback(self, velocity):
        velocity = cap(velocity, SPEED)
        pose = self.slam.pose

        u = velocity[X] * np.cos(pose[YAW]) + velocity[Y] * np.sin(pose[YAW])
        w = (-velocity[X] * np.sin(pose[YAW]) + velocity[Y] * np.cos(pose[YAW])) / self.epsilon

        self.vel_msg.linear.x = u * 1.2
        self.vel_msg.angular.z = w

    def obstacle(self, front, front_left, front_right, left, right):
        # return min([front, front_left, front_right]) < 0.5
        return False


class FollowerLaser(object):
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


def run(args):
    rospy.init_node('main')
    
    # Update control every 100 ms.
    rate_limiter = rospy.Rate(1000)

    l = Leader("")

    # # Leader robot
    # l = Leader("t0")
    # # Follower robot 1
    # f1 = Follower("t1", np.array([-.2, .2, 0.]))
    # # Follower robot 2
    # f2 = Follower("t2", np.array([-.2, -.2, 0.]))


    while not rospy.is_shutdown():
        l.slam.update()
        # f1.slam.update()
        # f2.slam.update()
        # Make sure all measurements are ready.
        l.update_velocities(rate_limiter)
        # f1.update_velocities(rate_limiter, l.slam.pose)
        # f2.update_velocities(rate_limiter, l.slam.pose)
        rate_limiter.sleep()


if __name__ == '__main__':
    args = None
    try:
        run(args)
    except rospy.ROSInterruptException:
        pass