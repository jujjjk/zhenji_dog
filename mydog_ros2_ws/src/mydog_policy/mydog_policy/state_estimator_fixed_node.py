#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""State-estimator wrapper with no-stance decay and snapshot metadata."""

from __future__ import annotations

import time

import numpy as np
import rclpy

from .state_estimator_node import MydogStateEstimatorNode


class MydogStateEstimatorFixedNode(MydogStateEstimatorNode):
    """Harden the existing leg-odometry estimator for policy deployment."""

    def __init__(self):
        super().__init__()
        if not self.has_parameter("no_stance_velocity_decay"):
            self.declare_parameter("no_stance_velocity_decay", 0.85)
        self.no_stance_velocity_decay = float(
            np.clip(
                float(self.get_parameter("no_stance_velocity_decay").value),
                0.0,
                1.0,
            )
        )
        self.get_logger().warn(
            "[STATE_FIXED] no-stance velocity decay enabled: "
            f"factor={self.no_stance_velocity_decay:.3f}; publishing motor snapshot metadata"
        )

    def update(self):
        try:
            motor = self.motor.get_latest()
            if not motor.valid:
                self.get_logger().warn("Motor state invalid. Skip estimator frame.")
                return

            max_age = float(np.max(motor.age_ms))
            if max_age > self.max_motor_age_ms:
                self.get_logger().warn(
                    f"Motor feedback too old: max_age={max_age:.1f} ms "
                    f"> {self.max_motor_age_ms:.1f} ms. Skip estimator frame."
                )
                return

            if self.require_online and not np.all(motor.online):
                self.get_logger().warn("Some motors offline. Skip estimator frame.")
                return

            imu = self.imu.get_latest()
            if not imu.valid:
                self.get_logger().warn("IMU invalid. Skip estimator frame.")
                return

            q_abs_policy, dq_policy = self.mapper.real_to_policy_abs_q_dq(
                motor.q_real,
                motor.dq_real,
            )

            result = self.estimator.estimate(
                q_abs_policy=q_abs_policy,
                dq_policy=dq_policy,
                omega_body=imu.gyro_rad_s,
            )

            # Original behavior holds the last velocity forever when no stance
            # candidate exists.  Exponential decay prevents a stale "ghost"
            # velocity from being fed back to the policy.
            if result.confidence <= 0.0:
                self.estimator.filtered *= self.no_stance_velocity_decay
                self.estimator.last_raw = self.estimator.filtered.copy()
                self.estimator.filtered[2] = 0.0
                self.estimator.last_raw[2] = 0.0
                result.base_lin_vel = self.estimator.filtered.astype(np.float32).copy()
                result.raw_base_lin_vel = self.estimator.last_raw.astype(np.float32).copy()

            self.publish_array(self.pub_base_lin_vel, result.base_lin_vel)

            seq = np.asarray(motor.snapshot_seq, dtype=np.int64).reshape(12)
            tick = np.asarray(motor.board_tick_ms, dtype=np.int64).reshape(12)
            seq_a = int(seq[0]) & 0xFFFF
            seq_b = int(seq[6]) & 0xFFFF
            tick_a = int(tick[0])
            tick_b = int(tick[6])

            # First ten values preserve the legacy message contract.  The extra
            # seven values are ignored by old consumers and used by the fixed
            # parity node to detect estimator/motor frame skew.
            state = np.concatenate(
                [
                    result.base_lin_vel,
                    imu.gyro_rad_s.astype(np.float32),
                    imu.projected_gravity.astype(np.float32),
                    np.array([np.deg2rad(imu.rpy_deg[2])], dtype=np.float32),
                    np.array(
                        [
                            result.confidence,
                            float(motor.cache_age_ms),
                            float(seq_a),
                            float(seq_b),
                            float(tick_a),
                            float(tick_b),
                            max_age,
                        ],
                        dtype=np.float32,
                    ),
                ]
            )
            self.publish_array(self.pub_state, state)

            debug = np.concatenate(
                [
                    result.raw_base_lin_vel,
                    result.stance_mask.astype(np.float32),
                    result.foot_height.astype(np.float32),
                    result.foot_speed.astype(np.float32),
                    np.array(
                        [
                            result.confidence,
                            max_age,
                            float(seq_a),
                            float(seq_b),
                        ],
                        dtype=np.float32,
                    ),
                ]
            )
            self.publish_array(self.pub_debug, debug)

            now = time.monotonic()
            if now - self._last_info_log_time >= 1.0:
                self._last_info_log_time = now
                self.get_logger().info(
                    "base_lin_vel="
                    f"{np.array2string(result.base_lin_vel, precision=3)} "
                    f"raw={np.array2string(result.raw_base_lin_vel, precision=3)} "
                    f"stance={result.stance_mask.astype(int).tolist()} "
                    f"height={np.array2string(result.foot_height, precision=3)} "
                    f"conf={result.confidence:.2f} "
                    f"seqA/B={seq_a}/{seq_b} max_age={max_age:.1f}ms"
                )

        except Exception as exc:
            self.get_logger().error(f"fixed state estimator error: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = MydogStateEstimatorFixedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
