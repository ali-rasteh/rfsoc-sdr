#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def quat_to_yaw(x, y, z, w) -> float:

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(a: float) -> float:

    a = math.fmod(a + math.pi, 2.0 * math.pi)
    if a <= 0:
        a += 2.0 * math.pi
    return a - math.pi


class MotionSimple(Node):


    def __init__(self,
                 max_linear: float = 0.2,
                 max_angular: float = 0.5,
                 cmd_topic: str = '/cmd_vel_unstamped',
                 odom_topic: str = '/odom',
                 rate: float = 10.0):
        super().__init__('motion_simple')

        self.max_linear = max_linear
        self.max_angular = max_angular


        self._cmd = Twist()


        self._pub = self.create_publisher(Twist, cmd_topic, 10)


        self._timer_period = 1.0 / rate
        self._timer = self.create_timer(self._timer_period, self._on_timer)


        self._last_odom = None
        self._odom_sub = self.create_subscription(
            Odometry, odom_topic, self._odom_callback, 10
        )

        self.get_logger().info(
            f'MotionSimple started, cmd={cmd_topic}, odom={odom_topic}, '
            f'rate={rate}Hz, vmax={max_linear}, wmax={max_angular}'
        )


    def _odom_callback(self, msg: Odometry):
        self._last_odom = msg

    def has_odom(self) -> bool:
        return self._last_odom is not None

    def get_xy_yaw(self):

        if self._last_odom is None:
            return None

        p = self._last_odom.pose.pose.position
        q = self._last_odom.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        return p.x, p.y, yaw


    def _on_timer(self):

        self._pub.publish(self._cmd)

    def set_cmd(self, linear_x: float, angular_z: float):
        v = max(min(linear_x, self.max_linear), -self.max_linear)
        w = max(min(angular_z, self.max_angular), -self.max_angular)

        self._cmd.linear.x = v
        self._cmd.linear.y = 0.0
        self._cmd.linear.z = 0.0
        self._cmd.angular.x = 0.0
        self._cmd.angular.y = 0.0
        self._cmd.angular.z = w

    def stop(self):
        self._cmd = Twist()
        self._pub.publish(self._cmd)
        self.get_logger().info('Stop command sent')
