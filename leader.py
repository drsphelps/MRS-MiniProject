#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import rospy

from robot import Robot

from geometry_msgs.msg import Point, Vector3
# For getting distances between centroids
from sklearn.metrics.pairwise import euclidean_distances
# For colouring points
from std_msgs.msg import ColorRGBA

SPEED = 0.2

X = 0
Y = 1
YAW = 2


def cap(v, max_speed):
  n = np.linalg.norm(v)
  if n > max_speed:
    return v / n * max_speed
  return v


class Leader(Robot):
    def __init__(self, name, rate_limiter):
        super(Leader, self).__init__(
            name=name,
            rate_limiter=rate_limiter,
            laser_min_angle=-np.pi/6.,
            laser_max_angle=np.pi/6.,
            laser_max_distance=2.)
        
        # For feedback linearization
        self._epsilon = 0.2
    
    def update_velocities(self, rate_limiter):
        if not self.laser.ready or not self.slam.ready:
            return
        
        # Get point to follow relative to the robot (base_link)
        point_to_follow = self.point_to_follow

        if np.linalg.norm(point_to_follow) < 0.5:
            # We have reached the desired position, so stop
            self.vel_msg.linear.x = 0.
            self.vel_msg.angular.z = 0.
        else:
            # Go towards the desired position using feedback linearisation
            self.linearised_feedback(point_to_follow)
        
        self.publisher.publish(self.vel_msg)
        self.pose_history.append(self.slam.pose)
        self.write_pose
    
    def linearised_feedback(self, velocity):
        velocity = cap(velocity, SPEED)

        # Feedback linearisation with relative positions
        u = velocity[X]
        w = velocity[Y] / self._epsilon

        self.vel_msg.linear.x =  u
        self.vel_msg.angular.z = w

    @property
    def point_to_follow(self):
        # Get centroids of all clusters
        centroids = self.laser.get_centroids

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
        
        # Publish the point to follow
        self.publish_points(
            points=[follow],
            publisher=self.laser.centroid_publisher,
            color=ColorRGBA(1., 0., 0., 1.),
            scale=Vector3(.1, .1, .1))

        return np.array(follow, dtype=np.float32)