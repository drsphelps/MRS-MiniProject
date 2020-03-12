#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import rospy

from slam import SLAM
from laser import Laser
from ground_truth import GroundtruthPose

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist, Point
# Laser scan message:
# http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html
from sensor_msgs.msg import LaserScan
# For visualising the centroid.
from visualization_msgs.msg import Marker

X = 0
Y = 1
YAW = 2


class Robot(object):
    def __init__(self, name, rate_limiter, map_frame="occupancy_grid", laser_min_angle=-np.pi/6., laser_max_angle=np.pi/6., laser_max_distance=2.):
        self.groundtruth = GroundtruthPose(name)
        self.pose_history = []
        self.publisher = rospy.Publisher('/' + name + '/cmd_vel', Twist, queue_size=5)
        self.name = name
        self.vel_msg = Twist()
        self.slam = SLAM(name=name, map_frame=map_frame)
        self.rate_limiter = rate_limiter
        self.laser = Laser(name=name, min_angle=laser_min_angle, max_angle=laser_max_angle, max_distance=laser_max_distance)
        with open('/tmp/gazebo_exercise_' + name + '.txt', 'w'):
            pass
    
    @property
    def write_pose(self):
        if len(self.pose_history) % 10:        
            with open('/tmp/gazebo_exercise_' + self.name + '.txt', 'a') as fp:
                fp.write('\n'.join(','.join(str(v) for v in p)
                                    for p in self.pose_history) + '\n')
                self.pose_history = []
    
    def publish_points(self, points, publisher, color, scale):
        if points is None or len(points) == 0:
            return
        marker = Marker()
        marker.header.frame_id = "/" + self.name + "/base_link"
        marker.type = marker.POINTS
        marker.action = marker.ADD
        marker.pose.orientation.w = 1

        ps = []
        for p in points:
            f = Point()
            f.x = p[X]
            f.y = p[Y]
            ps.append(f)
        marker.points = ps

        t = rospy.Duration()
        marker.lifetime = t
        marker.scale = scale
        marker.color = color
        publisher.publish(marker)

        self.rate_limiter.sleep()