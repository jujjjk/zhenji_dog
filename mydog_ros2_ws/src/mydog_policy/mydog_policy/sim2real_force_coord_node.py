#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dedicated transactional sim-to-real node for fanfan_force_coord_5280.onnx."""

from __future__ import annotations

import numpy as np
import rclpy

from .force_coord_contract import TORQUE_LIMITS_POLICY, validate_metadata
from .sim2real_parity_fixed_node import MydogPolicyParityFixedNode


class MydogForceCoord5280Node(MydogPolicyParityFixedNode):
    """Run model 5280 while enforcing its complete exported control contract."""

    def __init__(self):
        super().__init__()

        if self.deployment_config is None:
            raise RuntimeError(
                "force_coord deployment requires fanfan_deployment_config metadata"
            )
        validate_metadata(self.deployment_config)

        model_limits_policy = np.asarray(
            self.deployment_config["control"]["torque_limits"],
            dtype=np.float32,
        ).reshape(12)
        expected_policy = np.asarray(TORQUE_LIMITS_POLICY, dtype=np.float32)
        if not np.allclose(model_limits_policy, expected_policy, atol=1.0e-6):
            raise RuntimeError(
                "force_coord model torque profile mismatch: "
                f"expected={expected_policy.tolist()} "
                f"got={model_limits_policy.tolist()}"
            )

        mapper = self.obs_builder.mapper
        expected_real = mapper.policy_values_to_real_order(
            expected_policy
        ).astype(np.float32)
        if not np.allclose(
            self.contract_torque_limits_real,
            expected_real,
            atol=1.0e-6,
        ):
            raise RuntimeError(
                "policy-to-real torque-limit mapping mismatch: "
                f"expected_real={expected_real.tolist()} "
                f"active_contract={self.contract_torque_limits_real.tolist()}"
            )

        active_real = self._active_torque_limits_real()
        if np.any(active_real > expected_real + 1.0e-6):
            raise RuntimeError("active deployment limit exceeds ONNX model limit")

        global_budget = float(self.compute_torque_safety_budget_nm())
        self.get_logger().warn(
            "[FORCE_COORD_5280] exact metadata validated; "
            f"model_limits_policy={model_limits_policy.tolist()} "
            f"model_limits_real={expected_real.tolist()} "
            f"global_safety_cap={global_budget:.2f}Nm "
            f"active_limits_real={active_real.tolist()}"
        )
        if global_budget < 13.0 - 1.0e-6:
            self.get_logger().warn(
                "[FORCE_COORD_5280] global safety cap is below 13Nm. "
                "This is appropriate for staged testing, but calf authority is "
                "intentionally below the trained 10/10/13Nm envelope."
            )
        else:
            if not np.allclose(active_real, expected_real, atol=1.0e-6):
                raise RuntimeError(
                    "13Nm global cap should resolve to exact 10/10/13Nm "
                    "model limits after semantic mapping"
                )
            self.get_logger().warn(
                "[FORCE_COORD_5280] full trained torque profile enabled: "
                "hip=10Nm, thigh=10Nm, calf=13Nm."
            )


def main(args=None):
    rclpy.init(args=args)
    node = MydogForceCoord5280Node()
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
