#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import rospy

from leader import Leader
from follower import Follower

X = 0
Y = 1
YAW = 2


def run(args):
    rospy.init_node('main')
    
    # Control update rate = 100Hz.
    rate_limiter = rospy.Rate(100)

    # Leader robot
    l = Leader(
        name="t0",
        rate_limiter=rate_limiter)
    
    # Follower robot 1
    f1 = Follower(
        name="t1",
        rate_limiter=rate_limiter,
        leader=l,
        desired_leader_relative_position=np.array([.6, -.35], dtype=np.float32),
        desired_range_and_bearing=np.array([[0.4],[np.pi]], dtype=np.float32))
    
     # Follower robot 2
    f2 = Follower(
        name="t2",
        rate_limiter=rate_limiter,
        leader=f1,
        desired_leader_relative_position=np.array([.6, -.35], dtype=np.float32),
        desired_range_and_bearing=np.array([[0.4],[np.pi]], dtype=np.float32))

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
