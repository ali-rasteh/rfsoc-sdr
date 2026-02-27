#!/usr/bin/env python3
"""
map_motion_api.py

A clean, importable API for TurtleBot motion using TF (map->base) + cmd_vel.

Public API:
  - read_pos() -> (x, y, yaw) in map frame
  - move(yaw, distance) : rotate to absolute yaw in map, then forward distance (m)
  - compute_yaw_distance_to_target: calculate the yaw and distance moving to the target from current

Requires:
  - motion_simple.py providing MotionSimple and normalize_angle
"""

import math
import time
from typing import Optional, Tuple

import rclpy
from rclpy.executors import SingleThreadedExecutor

from tf2_ros import Buffer, TransformListener, TransformException

from motion_simple import MotionSimple, normalize_angle


Pose = Tuple[float, float, float]


def _quat_to_yaw(qx, qy, qz, qw) -> float:
    # yaw from quaternion
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _slew(prev: float, target: float, rate_limit: float, dt: float) -> float:
    """limit change rate: |x_dot| <= rate_limit"""
    if dt <= 0.0:
        return target
    lo = prev - rate_limit * dt
    hi = prev + rate_limit * dt
    return _clamp(target, lo, hi)


class MapMotionAPI:
    """
    Simple map-frame motion controller.

    Typical usage:
        api = MapMotionAPI()
        print(api.read_pos())
        api.move(yaw=0.0, distance=0.5)
        api.shutdown()
    """

    def __init__(
        self,
        cmd_topic: str = "/cmd_vel_unstamped",
        odom_topic: str = "/odom",
        rate: float = 10.0,
        max_linear: float = 0.10,
        max_angular: float = 0.25,
        target_frame: str = "map",
        source_frame: str = "base_link",
        tf_timeout: float = 20.0,
        lin_accel_limit: float = 0.2,  # m/s^2 (your requirement)
        ang_accel_limit: float = 0.8,  # rad/s^2 (reasonable default)
    ):
        rclpy.init()

        self.motion = MotionSimple(
            max_linear=max_linear,
            max_angular=max_angular,
            cmd_topic=cmd_topic,
            odom_topic=odom_topic,
            rate=rate,
        )

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.motion)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.motion)

        self.target_frame = target_frame
        self.source_frame = source_frame

        self.lin_accel_limit = float(lin_accel_limit)
        self.ang_accel_limit = float(ang_accel_limit)

        # internal state for accel limiting
        self._v_prev = 0.0
        self._w_prev = 0.0
        self._t_prev = time.time()

        # ensure TF is ready
        if not self._wait_for_tf(timeout=tf_timeout):
            self.shutdown()
            raise RuntimeError(f"TF {target_frame}->{source_frame} not available (timeout {tf_timeout}s)")

    # -----------------------------
    # Lifecycle
    # -----------------------------
    def shutdown(self):
        """Stop robot and shutdown ROS cleanly."""
        try:
            self.stop()
        except Exception:
            pass

        try:
            self.executor.shutdown()
        except Exception:
            pass

        try:
            self.motion.destroy_node()
        except Exception:
            pass

        try:
            rclpy.shutdown()
        except Exception:
            pass

    # -----------------------------
    # TF + pose
    # -----------------------------
    def _wait_for_tf(self, timeout: float) -> bool:
        t0 = time.time()
        self.motion.get_logger().info(f"Waiting for TF {self.target_frame} -> {self.source_frame} ...")
        while time.time() - t0 < timeout:
            self.executor.spin_once(timeout_sec=0.1)
            if self._lookup_tf() is not None:
                self.motion.get_logger().info("TF available.")
                return True
        self.motion.get_logger().error("TF timeout.")
        return False

    def _lookup_tf(self) -> Optional[Pose]:
        """Return (x,y,yaw) from TF, or None if not available."""
        try:
            trans = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time(),  # latest
            )
        except TransformException:
            return None

        t = trans.transform.translation
        q = trans.transform.rotation
        yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)
        return (t.x, t.y, yaw)

    def read_pos(self) -> Optional[Pose]:
        """Public API: get current pose in map frame (x,y,yaw)."""
        self.executor.spin_once(timeout_sec=0.05)
        return self._lookup_tf()

    # -----------------------------
    # Low-level command helpers
    # -----------------------------
    def _update_dt(self) -> float:
        now = time.time()
        dt = now - self._t_prev
        self._t_prev = now
        return max(dt, 1e-3)

    def _set_cmd_smooth(self, v_des: float, w_des: float):
        """
        Apply accel limits then send cmd.
        """
        # clamp desired to max
        v_des = _clamp(v_des, -self.motion.max_linear, self.motion.max_linear)
        w_des = _clamp(w_des, -self.motion.max_angular, self.motion.max_angular)

        dt = self._update_dt()
        v_cmd = _slew(self._v_prev, v_des, self.lin_accel_limit, dt)
        w_cmd = _slew(self._w_prev, w_des, self.ang_accel_limit, dt)

        self._v_prev, self._w_prev = v_cmd, w_cmd
        self.motion.set_cmd(v_cmd, w_cmd)

    def stop(self):
        """Stop immediately and reset internal accel state."""
        self.motion.stop()
        self._v_prev = 0.0
        self._w_prev = 0.0
        self._t_prev = time.time()

    # -----------------------------
    # High-level motion primitives
    # -----------------------------
    def _rotate_to_yaw_abs(
        self,
        target_yaw: float,
        kp: float = 2.0,
        max_w: float = 0.25,
        min_w: float = 0.05,
        yaw_tol_deg: float = 1.0,
        timeout: float = 20.0,
    ) -> bool:
        target_yaw = normalize_angle(target_yaw)
        max_w = min(max_w, self.motion.max_angular)
        yaw_tol = math.radians(yaw_tol_deg)

        pose0 = self._lookup_tf()
        if pose0 is None:
            self.motion.get_logger().error("Cannot read TF pose for rotation.")
            return False

        t0 = time.time()
        while rclpy.ok():
            self.executor.spin_once(timeout_sec=0.01)
            pose = self._lookup_tf()
            if pose is None:
                continue

            _, _, yaw = pose
            err = normalize_angle(target_yaw - yaw)

            if abs(err) < yaw_tol:
                self.stop()
                return True

            w_des = kp * err
            w_des = _clamp(w_des, -max_w, max_w)

            # avoid stalling near zero
            if abs(w_des) < min_w:
                w_des = math.copysign(min_w, w_des if w_des != 0 else err)

            self._set_cmd_smooth(0.0, w_des)

            if time.time() - t0 > timeout:
                self.motion.get_logger().warn("Rotation timeout.")
                self.stop()
                return False

        self.stop()
        return False

    def _forward_distance(
        self,
        distance: float,
        kv: float = 0.2,
        max_v: float = 0.10,
        min_v: float = 0.03,
        stop_tol_m: float = 0.02,
        kw_yaw: float = 2.0,
        kw_cross: float = 0.5,
        timeout: float = 60.0,
    ) -> bool:
        """
        Move forward by 'distance' meters (positive forward, negative backward),
        using map-frame TF feedback. Uses yaw/cross-track correction to keep straight
        along the initial heading at start of this primitive.
        """
        pose0 = self._lookup_tf()
        if pose0 is None:
            self.motion.get_logger().error("Cannot read TF pose for forward motion.")
            return False

        x0, y0, yaw0 = pose0
        sign = 1.0 if distance >= 0 else -1.0
        dist_target = abs(distance)

        max_v = min(max_v, self.motion.max_linear)

        t0 = time.time()
        while rclpy.ok():
            self.executor.spin_once(timeout_sec=0.01)
            pose = self._lookup_tf()
            if pose is None:
                continue

            x, y, yaw = pose
            dx = x - x0
            dy = y - y0

            # projection to initial heading yaw0
            forward = dx * math.cos(yaw0) + dy * math.sin(yaw0)
            cross = -dx * math.sin(yaw0) + dy * math.cos(yaw0)

            remaining = dist_target - abs(forward)

            if remaining <= stop_tol_m:
                self.stop()
                return True

            # desired linear speed (P on remaining)
            v_des = kv * remaining
            v_des = _clamp(v_des, min_v, max_v)
            v_des *= sign

            # heading + cross-track correction
            yaw_err = normalize_angle(yaw0 - yaw)
            w_des = kw_yaw * yaw_err + kw_cross * cross
            w_des = _clamp(w_des, -self.motion.max_angular, self.motion.max_angular)

            self._set_cmd_smooth(v_des, w_des)

            if time.time() - t0 > timeout:
                self.motion.get_logger().warn("Forward motion timeout.")
                self.stop()
                return False

        self.stop()
        return False

    # -----------------------------
    # Public API
    # -----------------------------
    def move(self, yaw: float, distance: float) -> bool:
        """
        Public API:
          1) rotate to absolute yaw in map frame
          2) then move forward 'distance' meters
        """
        pose = self.read_pos()
        if pose is None:
            self.motion.get_logger().error("Cannot read current pose in move().")
            return False

        _, _, yaw_now = pose
        yaw = normalize_angle(yaw)
        yaw_err = normalize_angle(yaw - yaw_now)

        # rotate if needed
        if abs(yaw_err) > math.radians(1.0):
            ok = self._rotate_to_yaw_abs(target_yaw=yaw)
            if not ok:
                return False

        # small settle
        time.sleep(2.0)

        # forward if needed
        if abs(distance) > 1e-4:
            return self._forward_distance(distance=distance)

        return True
    
    def compute_yaw_distance_to_target(
        self,
        current_xy: Tuple[float, float],
        target_xy: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Given current (x,y) and target (x,y) in map frame,
        return (yaw_abs, distance) where:
          - yaw_abs: absolute yaw in map frame to face target
          - distance: forward distance to reach target (>=0)
        """
        x, y = float(current_xy[0]), float(current_xy[1])
        tx, ty = float(target_xy[0]), float(target_xy[1])

        dx = tx - x
        dy = ty - y

        yaw_abs = math.atan2(dy, dx)              # [-pi, pi]
        yaw_abs = normalize_angle(yaw_abs)

        distance = math.hypot(dx, dy)             # >= 0
        return yaw_abs, distance


# -----------------------------
# Example usage (optional)
# -----------------------------
def main():
    api = MapMotionAPI(
        cmd_topic="/cmd_vel_unstamped",
        odom_topic="/odom",
        rate=10.0,
        max_linear=0.10,
        max_angular=0.25,
        source_frame="base_link",      # change to "base_footprint" if your TF uses that
        lin_accel_limit=0.2,
        ang_accel_limit=0.8,
    )

    try:
        print("pose:", api.read_pos())
        api.move(yaw=0.0, distance=0.3)
        print("pose:", api.read_pos())
    finally:
        api.shutdown()


if __name__ == "__main__":
    main()
