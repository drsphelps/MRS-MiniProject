#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import rospy

from robot import Robot

# For visualising the leader's centroid.
from geometry_msgs.msg import Vector3
# For colouring points
from std_msgs.msg import ColorRGBA
# For estimating the leader pose
from scipy.spatial import distance

ROBOT_RADIUS = 0.105 / 2.

X = 0
Y = 1
YAW = 2


class Follower(Robot):
    def __init__(self, name, rate_limiter, leader, desired_range_and_bearing, desired_leader_relative_position):
        super(Follower, self).__init__(
            name=name,
            rate_limiter=rate_limiter,
            laser_min_angle=-np.pi/3.,
            laser_max_angle=np.pi/3.,
            laser_max_distance=1.2)

        # The leader this robot needs to follow
        self.leader = leader

        # Last known position of the leader relative to the follower robot
        # Start with no known position
        self._last_leader_position = None

        # Constant to use in range and bearing method
        self._k = np.array([[.5],[.3]], dtype=np.float32)

        # Desired range and bearing to be achieved
        self._z_ij_d = desired_range_and_bearing

        # Desired position of the leader w.r.t. the follower robot
        self._desired_leader_relative_position = desired_leader_relative_position

        # Offset between the follower's SLAM position and the leader's SLAM position
        self._slam_offset = None

    def update_velocities(self, rate_limiter):
        if not self.laser.ready or not self.slam.ready:
            rate_limiter.sleep()
            return
        
        controls = self.get_controls
        self.vel_msg.linear.x = controls[0][0]
        self.vel_msg.angular.z = controls[1][0]

        self.publisher.publish(self.vel_msg)
        self.pose_history.append(self.slam.pose)
        self.write_pose
    

    @property
    # Returns an estimate of where the leader is relative to the follower robot
    # using the laser measurements and clustering
    def get_leader_position(self):
        # Get centroids of all clusters
        centroids = self.laser.get_centroids

        if len(centroids) == 0:
            # Make leader centroid just yourself, so that you stop moving
            leader_centroid = np.array([0., 0.], dtype=np.float32)
        elif len(centroids) == 1:
            # Only one cluster, so get that centroid
            leader_centroid = centroids[0]
        elif self._last_leader_position is None:
            # There isn't any last known leader position, so get the centroid closest
            # to the desired leader position relative to the robot
            leader_centroid = centroids[distance.cdist([self._desired_leader_relative_position], centroids).argmin()]
        else:
            # There is a last known leader position, so get the centroid closest to it
            leader_centroid = centroids[distance.cdist([self._last_leader_position], centroids).argmin()]
        
        # Make this the last known leader position
        if np.any(leader_centroid):
            self._last_leader_position = leader_centroid
        
        # Publish leader centroid
        if self.name == "t1":
            # Make centroid blue
            color = ColorRGBA(0., 0., 1., 1.)
        else:
            # Make centroid green
            color = ColorRGBA(0., 1., 0., 1.)
        self.publish_points(
            points=[leader_centroid],
            publisher=self.laser.centroid_publisher,
            color=color,
            scale=Vector3(.1, .1, .1))

        return leader_centroid

    @property
    # Returns the pose of the leader relative to the follower robot
    def get_leader_pose(self):
        pose = np.array([0., 0., 0.], dtype=np.float32)
        
        last_leader_position = self._last_leader_position
        current_leader_position = self.get_leader_position
        pose[X], pose[Y] = current_leader_position[X], current_leader_position[Y]

        if self._slam_offset is not None:
            pose[YAW] = self.leader.slam.pose[YAW] - self.slam.pose[YAW] + self._slam_offset
        else:
            # Set a starting offset and assume 0 relative orientation of leader w.r.t. to the follower
            self._slam_offset = np.abs(self.leader.slam.pose[YAW] - self.slam.pose[YAW])
            pose[YAW] = 0.
        
        # Replace Nan to 0. and infinity with large finite numbers
        pose = np.nan_to_num(pose)

        return pose

    @property
    # Returns linear and angular velocity to use next
    def get_controls(self):
        leader_pose = self.get_leader_pose

        if np.linalg.norm(leader_pose[:-1]) == 0.:
            # Stop the robot (send 0-controls)
            return np.array([[0.],[0.]], dtype=np.float32)
        
        # Distance from robot to leader
        l_ij = np.linalg.norm(leader_pose[:-1])
        # Relative YAW between leader and follower
        beta_ij = leader_pose[YAW]
        # Orientation of leader centroid relative to the robot
        alpha = np.arctan2(leader_pose[Y], leader_pose[X])
        
        psi_ij = np.pi + alpha - beta_ij
        gamma_ij = np.pi + alpha

        # Define matrices G and F
        G = np.array([[np.cos(gamma_ij), ROBOT_RADIUS * np.sin(gamma_ij)],
                      [-np.sin(gamma_ij) / l_ij, ROBOT_RADIUS * np.cos(gamma_ij) / l_ij]], dtype=np.float32)
        F = np.array([[-np.cos(psi_ij), 0.],
                      [np.sin(psi_ij)/ l_ij, -1.]], dtype=np.float32)

        # Current range and bearing
        z_ij = np.array([[l_ij],[psi_ij]], dtype=np.float32)

        # Last registered controls of leader
        u_i = np.array([[self.leader.vel_msg.linear.x],[self.leader.vel_msg.angular.z]], dtype=np.float32)
        # Desired controls (linear and angluar velocities)
        u_j = np.linalg.inv(G) * (self._k * (self._z_ij_d - z_ij) - F * u_i) * 1.2

        return u_j