import scipy.signal
import numpy as np
import rospy

from tf import TransformListener
from nav_msgs.msg import OccupancyGrid
from tf.transformations import euler_from_quaternion
from occupancy_grid import OccupancyGrid
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg

X = 0
Y = 1
YAW = 2

# Constants for occupancy grid.
FREE = 0
UNKNOWN = 1
OCCUPIED = 2


class SLAM(object):
  def __init__(self, name, map_frame="occupancy_grid"):
    rospy.Subscriber('/' + name + '/map', OccupancyGridMsg, self.callback)
    self.name = name
    self._tf = TransformListener()
    self._occupancy_grid = None
    self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    self.map_frame = map_frame
    
  def callback(self, msg):
    values = np.array(msg.data, dtype=np.int8).reshape((msg.info.width, msg.info.height))
    processed = np.empty_like(values)
    processed[:] = FREE
    processed[values < 0] = UNKNOWN
    processed[values > 50] = OCCUPIED
    processed = processed.T
    origin = [msg.info.origin.position.x, msg.info.origin.position.y, 0.]
    resolution = msg.info.resolution
    self._occupancy_grid = OccupancyGrid(processed, origin, resolution)

  def update(self):
    # Get pose w.r.t. map.
    a = self.map_frame
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