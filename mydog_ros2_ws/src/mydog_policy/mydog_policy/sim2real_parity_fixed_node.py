#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hardened sim-to-real parity node for the Fanfan 5100 ONNX policy.

This node subclasses the repository's existing ``MydogPolicyParityNode`` and
fixes deployment-only state/phase consistency problems without changing the
exported training contract:

* target-only mode is no longer blocked by startup stand;
* policy phase and previous_action advance only after a successful send;
* failed sends roll back the stateful policy-action filter;
* stale estimator/motor snapshots zero only the estimated linear velocity;
* the 50 Hz CSV records raw/filtered actions, gait offsets, send outcome and
  server timing for every policy cycle.
"""

from __future__ import annotations

import csv
import os
import queue
import threading
import time
from typing import Any

import numpy as np
import rclpy

from .sim2real_parity_node import MydogPolicyParityNode


class MydogPolicyParityFixedNode(MydogPolicyParityNode):
    """Transactional parity deployment node.

    A policy cycle is committed only when the motor command is accepted.  On a
    failed send the filter state, previous_action and gait phase remain at the
    last successfully applied cycle.
    """

    ESTIMATOR_META_MIN_SIZE = 17

    def setup_debug_csv(self):
        """Install a one-row-per-cycle parity diagnostic schema.

        This override is intentionally safe when called from the parent
        constructor, before this subclass's ``__init__`` body runs.
        """
        path = str(getattr(self, "debug_csv_path", "")).strip()
        if not path:
            return

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self._debug_csv_file = open(path, "w", newline="")
        self._debug_csv_writer = csv.writer(self._debug_csv_file)

        policy_names = list(self.obs_builder.mapper.policy_joint_names)
        real_names = list(self.obs_builder.mapper.real_joint_names)
        headers = [
            "timestamp",
            "cycle_sequence",
            "mode",
            "cmd_vx",
            "cmd_vy",
            "cmd_yaw",
            "base_phase",
            "effective_phase",
            "gait_gate",
            "policy_loop_dt_ms",
            "policy_loop_hz",
            "loop_overrun_ms",
            "state_age_ms",
            "motor_state_poll_dt_ms",
            "onnx_infer_dt_ms",
            "estimator_sync_ok",
            "estimator_sync_reason",
            "estimator_seq_a",
            "estimator_seq_b",
            "motor_seq_a",
            "motor_seq_b",
            "send_attempted",
            "send_ok",
            "send_error",
            "send_start_time",
            "send_end_time",
            "command_sequence",
            "http_total_ms",
            "server_prepare_ms",
            "server_spi_send_ms",
            "server_total_ms",
            "num_spi_frames",
            "batch_active",
            "projected_gravity_x",
            "projected_gravity_y",
            "projected_gravity_z",
            "tilt_rad",
            "rear_torque_boost_active",
            "rear_torque_boost_elapsed_s",
            "rear_torque_limit_nm",
        ]
        for name in policy_names:
            safe = str(name).replace("-", "_").replace(" ", "_")
            headers.extend(
                [
                    f"{safe}_action_raw",
                    f"{safe}_action_filtered",
                    f"{safe}_gait_offset",
                ]
            )
        for name in real_names:
            safe = str(name).replace("-", "_").replace(" ", "_")
            headers.extend(
                [
                    f"{safe}_q_target_raw",
                    f"{safe}_q_target_safe",
                    f"{safe}_q_current",
                    f"{safe}_dq_current",
                    f"{safe}_raw_pd_torque",
                    f"{safe}_limited_torque",
                    f"{safe}_command_torque_ff",
                    f"{safe}_feedback_torque",
                    f"{safe}_torque_limited_flag",
                    f"{safe}_motor_online",
                    f"{safe}_motor_fault",
                    f"{safe}_motor_age_ms",
                ]
            )
        self._debug_csv_writer.writerow(headers)
        self._debug_csv_file.flush()
        self.get_logger().warn(f"[PARITY_FIXED][CSV] writing 50Hz cycle log to {path}")

        if self.debug_csv_async:
            self._debug_csv_stop.clear()
            self._debug_csv_thread = threading.Thread(
                target=self._debug_csv_worker,
                name="policy-debug-csv",
                daemon=True,
            )
            self._debug_csv_thread.start()
            self.get_logger().info(
                f"[PARITY_FIXED][CSV] async writer queue={self.debug_csv_queue_size}"
            )

    def __init__(self):
        super().__init__()

        if not self.has_parameter("max_estimator_snapshot_lag"):
            self.declare_parameter("max_estimator_snapshot_lag", 2)
        if not self.has_parameter("zero_lin_vel_on_estimator_mismatch"):
            self.declare_parameter("zero_lin_vel_on_estimator_mismatch", True)
        if not self.has_parameter("send_failure_warn_count"):
            self.declare_parameter("send_failure_warn_count", 3)

        self.max_estimator_snapshot_lag = max(
            0, int(self.get_parameter("max_estimator_snapshot_lag").value)
        )
        self.zero_lin_vel_on_estimator_mismatch = bool(
            self.get_parameter("zero_lin_vel_on_estimator_mismatch").value
        )
        self.send_failure_warn_count = max(
            1, int(self.get_parameter("send_failure_warn_count").value)
        )

        self._estimator_meta: dict[str, Any] = {}
        self._obs_motor_meta: dict[str, Any] = {}
        self._consecutive_send_failures = 0
        self._last_send_meta = self._empty_send_meta()
        self._last_estimator_sync_warn = 0.0
        self._stale_command_hold_active = False
        self._tilt_protection_active = False

        if not self.has_parameter("enable_tilt_protection"):
            self.declare_parameter("enable_tilt_protection", False)
        if not self.has_parameter("enable_command_timeout_stand_hold"):
            self.declare_parameter("enable_command_timeout_stand_hold", False)
        if not self.has_parameter("max_tilt_rad"):
            self.declare_parameter("max_tilt_rad", 0.75)
        if not self.has_parameter("enable_rear_torque_boost"):
            self.declare_parameter("enable_rear_torque_boost", False)
        if not self.has_parameter("rear_torque_boost_nm"):
            self.declare_parameter("rear_torque_boost_nm", 17.0)
        if not self.has_parameter("rear_torque_boost_duration_sec"):
            self.declare_parameter("rear_torque_boost_duration_sec", 2.5)
        if not self.has_parameter("rear_torque_boost_tilt_threshold_rad"):
            self.declare_parameter("rear_torque_boost_tilt_threshold_rad", 0.10)
        if not self.has_parameter("rear_torque_boost_q_error_rad"):
            self.declare_parameter("rear_torque_boost_q_error_rad", 0.12)
        if not self.has_parameter("rear_torque_boost_overload_margin_nm"):
            self.declare_parameter("rear_torque_boost_overload_margin_nm", 1.0)
        self.enable_tilt_protection = bool(
            self.get_parameter("enable_tilt_protection").value
        )
        self.enable_command_timeout_stand_hold = bool(
            self.get_parameter("enable_command_timeout_stand_hold").value
        )
        self.max_tilt_rad = float(self.get_parameter("max_tilt_rad").value)
        if (
            not np.isfinite(self.max_tilt_rad)
            or self.max_tilt_rad <= 0.0
            or self.max_tilt_rad >= 1.57
        ):
            raise RuntimeError("max_tilt_rad must be finite and in (0, pi/2)")

        self.enable_rear_torque_boost = bool(
            self.get_parameter("enable_rear_torque_boost").value
        )
        self.rear_torque_boost_nm = float(
            self.get_parameter("rear_torque_boost_nm").value
        )
        self.rear_torque_boost_duration_sec = float(
            self.get_parameter("rear_torque_boost_duration_sec").value
        )
        self.rear_torque_boost_tilt_threshold_rad = float(
            self.get_parameter("rear_torque_boost_tilt_threshold_rad").value
        )
        self.rear_torque_boost_q_error_rad = float(
            self.get_parameter("rear_torque_boost_q_error_rad").value
        )
        self.rear_torque_boost_overload_margin_nm = float(
            self.get_parameter("rear_torque_boost_overload_margin_nm").value
        )
        if (
            not np.isfinite(self.rear_torque_boost_nm)
            or self.rear_torque_boost_nm <= 0.0
            or self.rear_torque_boost_nm > self.motion_torque_ff_limit_nm
        ):
            raise RuntimeError(
                "rear_torque_boost_nm must be in (0, motion_torque_ff_limit_nm]"
            )
        if (
            not np.isfinite(self.rear_torque_boost_duration_sec)
            or self.rear_torque_boost_duration_sec <= 0.0
        ):
            raise RuntimeError("rear_torque_boost_duration_sec must be positive")
        if (
            not np.isfinite(self.rear_torque_boost_tilt_threshold_rad)
            or self.rear_torque_boost_tilt_threshold_rad <= 0.0
            or self.rear_torque_boost_tilt_threshold_rad >= self.max_tilt_rad
        ):
            raise RuntimeError(
                "rear_torque_boost_tilt_threshold_rad must be in (0, max_tilt_rad)"
            )
        if (
            not np.isfinite(self.rear_torque_boost_q_error_rad)
            or self.rear_torque_boost_q_error_rad <= 0.0
            or not np.isfinite(self.rear_torque_boost_overload_margin_nm)
            or self.rear_torque_boost_overload_margin_nm < 0.0
        ):
            raise RuntimeError("rear torque boost trigger thresholds are invalid")

        self._rear_torque_boost_until = 0.0
        self._rear_torque_boost_used = False
        self._rear_torque_boost_reason = ""
        self._rear_torque_boost_started_at = 0.0

        self._install_state_capture_hooks()

        # Parent construction may have selected waiting_feedback because the
        # launch defaults startup_stand_first=true.  In target-only diagnostics
        # no command can be sent, so waiting would otherwise time out forever.
        if not self.enable_send and self._startup_stand_state != "complete":
            self.startup_stand_first = False
            self._startup_stand_state = "complete"
            self._startup_stand_start_time = None
            self._startup_stand_ready_since = None
            self.get_logger().warn(
                "[PARITY_FIXED] enable_send=false: bypass startup stand so ONNX "
                "target preview can run. Motors remain untouched."
            )

        self.get_logger().warn(
            "[PARITY_FIXED] transactional phase/action commit enabled; failed or "
            "skipped sends freeze filter state, previous_action and gait phase."
        )
        if self.enable_send:
            self.get_logger().warn(
                "[PARITY_FIXED][HARDWARE] motor_torque_limit_nm is a software PD "
                "budget. Verify the motor-driver current limit separately before "
                "ground testing; this node does not guess an ampere value."
            )
            self.get_logger().warn(
                "[PARITY_FIXED][SIM2REAL] model command/action/PD path is kept "
                "unchanged; extra timeout stand hold is "
                f"{self.enable_command_timeout_stand_hold}."
            )
            if self.enable_rear_torque_boost:
                self.get_logger().warn(
                    "[PARITY_FIXED][TORQUE_BOOST] rear legs may use "
                    f"{self.rear_torque_boost_nm:.1f} Nm for "
                    f"{self.rear_torque_boost_duration_sec:.2f} s after a "
                    "rear overload/tilt trigger; normal limits remain 10/10/13 Nm."
                )

    def _reset_rear_torque_boost_window(self) -> None:
        self._rear_torque_boost_until = 0.0
        self._rear_torque_boost_used = False
        self._rear_torque_boost_reason = ""
        self._rear_torque_boost_started_at = 0.0

    def _rear_torque_boost_active(self) -> bool:
        return bool(
            self.enable_rear_torque_boost
            and time.monotonic() < self._rear_torque_boost_until
        )

    def _rear_torque_boost_elapsed(self) -> float:
        if not self._rear_torque_boost_active():
            return 0.0
        return max(0.0, time.monotonic() - self._rear_torque_boost_started_at)

    def _active_torque_limits_real(self) -> np.ndarray:
        limits = super()._active_torque_limits_real()
        if self._rear_torque_boost_active():
            # Real motor order is FR, FL, RL, RR. The short boost applies only
            # to the six rear joints; front joints stay at the model limits.
            limits[6:12] = np.float32(self.rear_torque_boost_nm)
        return limits.astype(np.float32)

    def _maybe_activate_rear_torque_boost(
        self,
        target_real_raw: np.ndarray,
        current_q: np.ndarray,
        current_dq: np.ndarray,
        tilt_rad: float,
    ) -> None:
        if not self.enable_rear_torque_boost:
            return
        if self._rear_torque_boost_active() or self._rear_torque_boost_used:
            return
        if self.zero_cmd_stand_active():
            return

        target_real_raw = np.asarray(target_real_raw, dtype=np.float32).reshape(12)
        current_q = np.asarray(current_q, dtype=np.float32).reshape(12)
        current_dq = np.asarray(current_dq, dtype=np.float32).reshape(12)
        rear = np.arange(6, 12, dtype=np.int64)
        rear_calf = np.asarray([8, 11], dtype=np.int64)
        nominal_limits = np.asarray(
            self.contract_torque_limits_real,
            dtype=np.float32,
        ).reshape(12)
        tau_raw = (
            self.send_kp_real * (target_real_raw - current_q)
            - self.send_kd_real * current_dq
            + float(self.send_torque)
        )
        rear_overload = float(
            np.max(np.abs(tau_raw[rear]) - nominal_limits[rear])
        )
        rear_q_error = float(
            np.max(np.abs(target_real_raw[rear_calf] - current_q[rear_calf]))
        )
        trigger = []
        if rear_overload >= self.rear_torque_boost_overload_margin_nm:
            trigger.append(f"rear_tau_over={rear_overload:.2f}Nm")
        if rear_q_error >= self.rear_torque_boost_q_error_rad:
            trigger.append(f"rear_calf_qerr={rear_q_error:.3f}rad")
        if np.isfinite(tilt_rad) and tilt_rad >= self.rear_torque_boost_tilt_threshold_rad:
            trigger.append(f"tilt={tilt_rad:.3f}rad")
        if not trigger:
            return

        now = time.monotonic()
        self._rear_torque_boost_started_at = now
        self._rear_torque_boost_until = now + self.rear_torque_boost_duration_sec
        self._rear_torque_boost_used = True
        self._rear_torque_boost_reason = ",".join(trigger)
        self.get_logger().warn(
            "[PARITY_FIXED][TORQUE_BOOST] activated | "
            f"limit={self.rear_torque_boost_nm:.1f}Nm "
            f"duration={self.rear_torque_boost_duration_sec:.2f}s "
            f"reason={self._rear_torque_boost_reason}"
        )

    @staticmethod
    def _tilt_from_observation(obs: np.ndarray) -> float:
        """Return body tilt from projected gravity, or +inf if invalid."""
        obs_flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs_flat.size < 9:
            return float("inf")
        gravity = obs_flat[6:9]
        norm = float(np.linalg.norm(gravity))
        if norm < 0.5 or not np.all(np.isfinite(gravity)):
            return float("inf")
        upright_cos = float(np.clip(-gravity[2] / norm, -1.0, 1.0))
        return float(np.arccos(upright_cos))

    def _write_protected_cycle(
        self,
        obs: np.ndarray,
        info: dict,
        max_age: float,
        target_real: np.ndarray,
        target_real_raw: np.ndarray,
        torque_limit_info: dict,
        mode: str,
        reason: str,
    ) -> bool:
        """Send and record one non-policy safety target."""
        current_q = np.asarray(info["q_real"], dtype=np.float32).reshape(12)
        current_dq = np.asarray(info["dq_real"], dtype=np.float32).reshape(12)
        zeros = np.zeros(12, dtype=np.float32)
        send_ok = True
        send_meta = self._empty_send_meta()
        if self.enable_send:
            send_ok = self.send_motion_batch(
                target_real,
                info,
                motor_vel_cmd=zeros,
                motor_torque_ff=torque_limit_info.get(
                    "torque_ff_cmd",
                    np.full(
                        12,
                        float(self.send_torque),
                        dtype=np.float32,
                    ),
                ),
            )
            send_meta = dict(self._last_send_meta)
        else:
            send_meta.update(
                {
                    "send_attempted": False,
                    "send_ok": True,
                    "send_error": "preview_no_send",
                }
            )

        phase = float(getattr(self.obs_builder, "last_gait_phase", 0.0))
        self.publish_array(self.pub_target, target_real)
        self.maybe_write_policy_csv(
            obs=obs,
            action_raw=zeros,
            action_policy_obs=zeros,
            action=zeros,
            q_des=target_real,
            current_q=current_q,
            current_dq=current_dq,
            error=np.asarray(target_real, dtype=np.float32) - current_q,
            max_age=max_age,
            measured_torque=info.get("torque_real"),
            motor_temp=info.get("temp_real"),
            motor_online=info.get("online"),
            motor_error_code=info.get("error_code"),
            motor_age_ms=info.get("age_ms"),
            q_raw_des=target_real_raw,
            q_smooth_des=target_real,
            motor_vel_cmd=zeros,
            smoothing_info=self._disabled_pre_limit_info(
                target_real_raw,
                current_q,
                self.compute_torque_safety_budget_nm(),
            ),
            torque_limit_info=torque_limit_info,
            cpg_action_info={
                "action_mode": mode,
                "command_gate_scale": 0.0,
                "zero_cmd_stand": True,
                "frequency": 0.0,
                "phase": phase,
                "base_phase": float(getattr(self, "_contract_phase", 0.0)),
                "phase_lead_sec": float(getattr(self, "gait_phase_lead_sec", 0.0)),
                "gait_gate": 0.0,
                "model_gait_period_s": float(
                    getattr(self, "model_gait_phase_period", 0.0)
                ),
                "gait_period_scale": float(
                    getattr(self, "contract_gait_period_scale", 1.0)
                ),
                "active_gait_period_s": float(getattr(self, "gait_phase_period", 0.0)),
                "gait_offset_policy": zeros,
                "protection_reason": reason,
                "send_meta": send_meta,
            },
            mode=mode,
        )
        return bool(send_ok)

    def _handle_stale_command(self, obs: np.ndarray, info: dict, max_age: float) -> None:
        """Keep the robot supported after command publisher loss."""
        if not self._stale_command_hold_active:
            self._stale_command_hold_active = True
            self._tilt_protection_active = False
            self._reset_contract_state(reset_phase=False)
            self.get_logger().error(
                "[SAFE][CMD_TIMEOUT] /cmd_vel is stale; switching to continuous "
                "default-stand hold. Publish a fresh non-zero command to resume."
            )

        current_q = np.asarray(info["q_real"], dtype=np.float32).reshape(12)
        current_dq = np.asarray(info["dq_real"], dtype=np.float32).reshape(12)
        target_raw = self.default_stand_target_real_order()
        target, limit_info = self._apply_pd_equivalent_limit(
            target_raw,
            current_q,
            current_dq,
        )
        limit_info = dict(limit_info)
        limit_info["protection_mode"] = "stale_command_pd_equivalent"
        self._write_protected_cycle(
            obs,
            info,
            max_age,
            target,
            target_raw,
            limit_info,
            "stale_cmd_stand",
            "cmd_vel_missing_or_stale",
        )

    def _handle_tilt_protection(self, obs: np.ndarray, info: dict, max_age: float) -> None:
        """Stop walking targets and damp the measured pose after excessive tilt."""
        if not self._tilt_protection_active:
            self._tilt_protection_active = True
            self._reset_contract_state(reset_phase=False)
            self.get_logger().error(
                "[SAFE][TILT] excessive or invalid projected gravity; "
                "holding the measured pose and freezing policy state."
            )

        current_q = np.asarray(info["q_real"], dtype=np.float32).reshape(12)
        current_dq = np.asarray(info["dq_real"], dtype=np.float32).reshape(12)
        target_raw = current_q.copy()
        target, limit_info = self._apply_pd_equivalent_limit(
            target_raw,
            current_q,
            current_dq,
        )
        limit_info = dict(limit_info)
        limit_info["protection_mode"] = "tilt_hold_current_pose"
        self._write_protected_cycle(
            obs,
            info,
            max_age,
            target,
            target_raw,
            limit_info,
            "tilt_hold",
            "tilt_or_invalid_gravity",
        )

    @staticmethod
    def _empty_send_meta() -> dict[str, Any]:
        return {
            "send_attempted": False,
            "send_ok": False,
            "send_error": "",
            "http_total_ms": 0.0,
            "server_prepare_ms": 0.0,
            "server_spi_send_ms": 0.0,
            "server_total_ms": 0.0,
            "num_spi_frames": 0,
            "batch_active": False,
        }

    def _install_state_capture_hooks(self) -> None:
        """Capture estimator and motor snapshot metadata without replacing ObsBuilder."""
        original_set_state_estimator = self.obs_builder.set_state_estimator

        def set_state_estimator_with_meta(data):
            original_set_state_estimator(data)
            arr = np.asarray(data, dtype=np.float32).reshape(-1)
            if arr.size >= self.ESTIMATOR_META_MIN_SIZE:
                self._estimator_meta = {
                    "confidence": float(arr[10]),
                    "cache_age_ms": float(arr[11]),
                    "seq_a": int(round(float(arr[12]))) & 0xFFFF,
                    "seq_b": int(round(float(arr[13]))) & 0xFFFF,
                    "tick_a_ms": int(round(float(arr[14]))),
                    "tick_b_ms": int(round(float(arr[15]))),
                    "max_motor_age_ms": float(arr[16]),
                    "received_monotonic": time.monotonic(),
                }
            else:
                self._estimator_meta = {
                    "received_monotonic": time.monotonic(),
                    "legacy": True,
                }

        self.obs_builder.set_state_estimator = set_state_estimator_with_meta

        original_get_latest = self.obs_builder.motor.get_latest

        def get_latest_with_meta():
            snap = original_get_latest()
            seq = np.asarray(snap.snapshot_seq, dtype=np.int64).reshape(12)
            tick = np.asarray(snap.board_tick_ms, dtype=np.int64).reshape(12)
            self._obs_motor_meta = {
                "seq_a": int(seq[0]) & 0xFFFF,
                "seq_b": int(seq[6]) & 0xFFFF,
                "tick_a_ms": int(tick[0]),
                "tick_b_ms": int(tick[6]),
                "cache_age_ms": float(snap.cache_age_ms),
            }
            return snap

        self.obs_builder.motor.get_latest = get_latest_with_meta

    @staticmethod
    def _seq_distance(a: int, b: int) -> int:
        forward = (int(a) - int(b)) & 0xFFFF
        backward = (int(b) - int(a)) & 0xFFFF
        return min(forward, backward)

    def _guard_estimator_sync(self, obs: np.ndarray, info: dict):
        """Zero only estimated linear velocity when estimator/motor frames diverge."""
        status = {
            "ok": True,
            "reason": "ok",
            "estimator_seq_a": -1,
            "estimator_seq_b": -1,
            "motor_seq_a": int(self._obs_motor_meta.get("seq_a", -1)),
            "motor_seq_b": int(self._obs_motor_meta.get("seq_b", -1)),
        }
        meta = self._estimator_meta
        if not meta or meta.get("legacy"):
            status["ok"] = False
            status["reason"] = "legacy_or_missing_estimator_metadata"
            if self.zero_lin_vel_on_estimator_mismatch:
                obs = np.asarray(obs, dtype=np.float32).copy()
                obs[0:3] = 0.0
                info = dict(info)
                info["base_lin_vel"] = np.zeros(3, dtype=np.float32)
            return obs, info, status

        est_a = int(meta.get("seq_a", -1))
        est_b = int(meta.get("seq_b", -1))
        mot_a = int(self._obs_motor_meta.get("seq_a", -1))
        mot_b = int(self._obs_motor_meta.get("seq_b", -1))
        status.update(
            {
                "estimator_seq_a": est_a,
                "estimator_seq_b": est_b,
                "motor_seq_a": mot_a,
                "motor_seq_b": mot_b,
            }
        )
        if min(est_a, est_b, mot_a, mot_b) < 0:
            status["reason"] = "incomplete_snapshot_metadata"
            return obs, info, status

        lag_a = self._seq_distance(est_a, mot_a)
        lag_b = self._seq_distance(est_b, mot_b)
        if max(lag_a, lag_b) <= self.max_estimator_snapshot_lag:
            return obs, info, status

        status["ok"] = False
        status["reason"] = f"snapshot_lag_a={lag_a},lag_b={lag_b}"
        if self.zero_lin_vel_on_estimator_mismatch:
            obs = np.asarray(obs, dtype=np.float32).copy()
            obs[0:3] = 0.0
            info = dict(info)
            info["base_lin_vel"] = np.zeros(3, dtype=np.float32)

        now = time.monotonic()
        if now - self._last_estimator_sync_warn >= 1.0:
            self._last_estimator_sync_warn = now
            self.get_logger().warn(
                "[PARITY_FIXED] estimator/motor snapshot mismatch: "
                f"{status['reason']}; obs base_lin_vel zeroed="
                f"{self.zero_lin_vel_on_estimator_mismatch}"
            )
        return obs, info, status

    def _capture_filter_state(self):
        return (
            self.contract_action_filter.action.copy(),
            self.contract_action_filter.action_velocity.copy(),
        )

    def _restore_filter_state(self, state) -> None:
        if state is None:
            return
        action, velocity = state
        self.contract_action_filter.action[:] = action
        self.contract_action_filter.action_velocity[:] = velocity

    def send_motion_batch(
        self,
        target_real: np.ndarray,
        info: dict,
        motor_vel_cmd=None,
        motor_torque_ff=None,
    ):
        """Send one policy command and retain complete HTTP/server diagnostics."""
        target_real = np.asarray(target_real, dtype=np.float32).reshape(12)
        q_real = np.asarray(info["q_real"], dtype=np.float32).reshape(12)
        if motor_vel_cmd is None:
            motor_vel_cmd = np.full(12, float(self.send_speed), dtype=np.float32)
        else:
            motor_vel_cmd = np.asarray(motor_vel_cmd, dtype=np.float32).reshape(12)
        if motor_torque_ff is None:
            motor_torque_ff = np.full(
                12,
                float(self.send_torque),
                dtype=np.float32,
            )
        else:
            motor_torque_ff = np.asarray(
                motor_torque_ff,
                dtype=np.float32,
            ).reshape(12)

        meta = self._empty_send_meta()
        meta["send_attempted"] = True
        self._last_send_meta = meta

        if not np.all(np.isfinite(target_real)):
            meta["send_error"] = "target_has_nan_or_inf"
            self.get_logger().warn("[PARITY_FIXED][SEND] target has NaN/Inf; skipped")
            return False
        if not np.all(np.isfinite(motor_vel_cmd)):
            meta["send_error"] = "velocity_target_has_nan_or_inf"
            self.get_logger().warn("[PARITY_FIXED][SEND] velocity has NaN/Inf; skipped")
            return False
        if not np.all(np.isfinite(motor_torque_ff)):
            meta["send_error"] = "torque_feedforward_has_nan_or_inf"
            self.get_logger().warn(
                "[PARITY_FIXED][SEND] torque feed-forward has NaN/Inf; skipped"
            )
            return False

        delta = target_real - q_real
        max_delta = float(np.max(np.abs(delta)))
        if max_delta > self.max_target_delta:
            meta["send_error"] = (
                f"target_jump_{max_delta:.4f}_gt_{self.max_target_delta:.4f}"
            )
            self.get_logger().warn(
                f"[PARITY_FIXED][SEND] target jump {max_delta:.3f} rad > "
                f"{self.max_target_delta:.3f} rad; skipped"
            )
            return False

        items = []
        for i, mid in enumerate(self.obs_builder.mapper.get_real_motor_ids()):
            items.append(
                {
                    "motor_id": int(mid),
                    "position": float(target_real[i]),
                    "speed": float(motor_vel_cmd[i]),
                    "torque": float(motor_torque_ff[i]),
                    "kp": float(self.send_kp_real[i]),
                    "kd": float(self.send_kd_real[i]),
                }
            )
        payload = {
            "items": items,
            "enable_first": bool(self.send_enable_first),
            "stop_first": bool(self.send_stop_first),
        }

        url = f"{self.motor_base_url}/api/rs04/motion_batch_fast"
        send_start_wall = time.time()
        send_start_perf = time.perf_counter()
        self._last_send_start_time = f"{send_start_wall:.6f}"
        self._last_command_sequence += 1
        try:
            response = self.http_session.post(
                url,
                json=payload,
                timeout=self.http_timeout,
            )
            http_total_ms = (time.perf_counter() - send_start_perf) * 1000.0
            self._last_send_end_time = f"{time.time():.6f}"
            meta["http_total_ms"] = http_total_ms

            try:
                body = response.json()
            except Exception:
                body = {}

            if response.status_code != 200 or not bool(body.get("ok", False)):
                meta["send_error"] = (
                    f"http_{response.status_code}:"
                    f"{body.get('detail', body.get('error', response.text[:160]))}"
                )
                self.get_logger().warn(
                    f"[PARITY_FIXED][SEND] failed: {meta['send_error']}"
                )
                return False

            meta.update(
                {
                    "send_ok": True,
                    "server_prepare_ms": float(body.get("prepare_ms", 0.0) or 0.0),
                    "server_spi_send_ms": float(body.get("spi_send_ms", 0.0) or 0.0),
                    "server_total_ms": float(body.get("total_ms", 0.0) or 0.0),
                    "num_spi_frames": int(body.get("num_spi_frames", 0) or 0),
                    "batch_active": bool(body.get("batch_active", False)),
                }
            )

            send_time_perf = time.perf_counter()
            if self._last_motor_send_time_perf is not None:
                send_dt = send_time_perf - self._last_motor_send_time_perf
                self._last_motor_send_dt_ms = send_dt * 1000.0
                self._last_motor_send_hz = 1.0 / send_dt if send_dt > 0.0 else 0.0
            self._last_motor_send_time_perf = send_time_perf
            self._motor_send_count += 1
            self._motor_sends_since_last_csv += 1

            now = time.monotonic()
            if now - self._last_send_ok_log_time >= 1.0:
                self._last_send_ok_log_time = now
                self.get_logger().info(
                    "[PARITY_FIXED][SEND] ok | "
                    f"http={http_total_ms:.2f}ms "
                    f"spi={meta['server_spi_send_ms']:.2f}ms "
                    f"frames={meta['num_spi_frames']} "
                    f"batch={meta['batch_active']} max_delta={max_delta:.3f}"
                )
            return True
        except Exception as exc:
            self._last_send_end_time = f"{time.time():.6f}"
            meta["http_total_ms"] = (time.perf_counter() - send_start_perf) * 1000.0
            meta["send_error"] = f"request_exception:{exc}"
            self.get_logger().warn(f"[PARITY_FIXED][SEND] {meta['send_error']}")
            return False
        finally:
            self._last_send_meta = dict(meta)

    def _write_policy_csv_sync(
        self,
        obs: np.ndarray,
        action_raw: np.ndarray,
        action_policy_obs: np.ndarray,
        action: np.ndarray,
        q_des: np.ndarray,
        current_q: np.ndarray,
        error: np.ndarray,
        max_age: float,
        current_dq: np.ndarray = None,
        measured_torque: np.ndarray = None,
        motor_temp: np.ndarray = None,
        motor_online: np.ndarray = None,
        motor_error_code: np.ndarray = None,
        motor_age_ms: np.ndarray = None,
        q_raw_des: np.ndarray = None,
        q_smooth_des: np.ndarray = None,
        motor_vel_cmd: np.ndarray = None,
        smoothing_info: dict = None,
        torque_limit_info: dict = None,
        rear_bias_info: dict = None,
        cpg_action_info: dict = None,
        mode: str = "policy",
        joint_probe_delta_rad: float = 0.0,
        joint_probe_name: str = "",
        record_time: float = None,
        timing_info: dict = None,
    ):
        if self._debug_csv_writer is None:
            return

        now = time.time() if record_time is None else float(record_time)
        action_raw = np.asarray(action_raw, dtype=np.float32).reshape(12)
        action_filtered = np.asarray(action_policy_obs, dtype=np.float32).reshape(12)
        q_des = np.asarray(q_des, dtype=np.float32).reshape(12)
        q_raw_des = q_des if q_raw_des is None else np.asarray(q_raw_des, dtype=np.float32).reshape(12)
        current_q = np.asarray(current_q, dtype=np.float32).reshape(12)
        current_dq = np.zeros(12, dtype=np.float32) if current_dq is None else np.asarray(current_dq, dtype=np.float32).reshape(12)
        measured_torque = np.zeros(12, dtype=np.float32) if measured_torque is None else np.asarray(measured_torque, dtype=np.float32).reshape(12)
        motor_online = np.zeros(12, dtype=bool) if motor_online is None else np.asarray(motor_online, dtype=bool).reshape(12)
        motor_error_code = np.zeros(12, dtype=np.int32) if motor_error_code is None else np.asarray(motor_error_code, dtype=np.int32).reshape(12)
        motor_age_ms = np.full(12, float(max_age), dtype=np.float32) if motor_age_ms is None else np.asarray(motor_age_ms, dtype=np.float32).reshape(12)
        torque_limit_info = {} if torque_limit_info is None else dict(torque_limit_info)
        cpg = {} if cpg_action_info is None else dict(cpg_action_info)
        timing = self.make_debug_timing_info() if timing_info is None else dict(timing_info)

        gait_offset = np.asarray(
            cpg.get("gait_offset_policy", np.zeros(12, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(12)
        limited_torque = np.asarray(
            torque_limit_info.get("tau_safe_signed", np.zeros(12, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(12)
        command_torque_ff = np.asarray(
            torque_limit_info.get(
                "torque_ff_cmd",
                np.full(12, float(self.send_torque), dtype=np.float32),
            ),
            dtype=np.float32,
        ).reshape(12)
        limited_mask = np.asarray(
            torque_limit_info.get("limited_mask", np.zeros(12, dtype=bool)),
            dtype=bool,
        ).reshape(12)
        raw_pd_torque = self.send_kp_real * (q_raw_des - current_q) - self.send_kd_real * current_dq

        sync = dict(cpg.get("estimator_sync", {}))
        send = dict(cpg.get("send_meta", {}))
        gravity = np.zeros(3, dtype=np.float32)
        obs_flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs_flat.size >= 9:
            gravity[:] = obs_flat[6:9]
        row = [
            f"{now:.6f}",
            self._csv_cycle_sequence,
            mode,
            f"{float(self.cmd[0]):.6f}",
            f"{float(self.cmd[1]):.6f}",
            f"{float(self.cmd[2]):.6f}",
            f"{float(cpg.get('base_phase', getattr(self, '_contract_phase', 0.0))):.6f}",
            f"{float(cpg.get('phase', getattr(self.obs_builder, 'last_gait_phase', 0.0))):.6f}",
            f"{float(cpg.get('gait_gate', 0.0)):.6f}",
            f"{float(timing.get('policy_loop_dt_ms', 0.0)):.3f}",
            f"{float(timing.get('policy_loop_hz', 0.0)):.3f}",
            f"{float(timing.get('loop_overrun_ms', 0.0)):.3f}",
            f"{float(timing.get('state_age_ms', max_age)):.3f}",
            f"{float(timing.get('motor_state_poll_dt_ms', 0.0)):.3f}",
            f"{float(timing.get('onnx_infer_dt_ms', 0.0)):.3f}",
            int(bool(sync.get("ok", True))),
            str(sync.get("reason", "")),
            int(sync.get("estimator_seq_a", -1)),
            int(sync.get("estimator_seq_b", -1)),
            int(sync.get("motor_seq_a", -1)),
            int(sync.get("motor_seq_b", -1)),
            int(bool(send.get("send_attempted", False))),
            int(bool(send.get("send_ok", False))),
            str(send.get("send_error", "")),
            self._last_send_start_time,
            self._last_send_end_time,
            self._last_command_sequence,
            f"{float(send.get('http_total_ms', 0.0)):.3f}",
            f"{float(send.get('server_prepare_ms', 0.0)):.3f}",
            f"{float(send.get('server_spi_send_ms', 0.0)):.3f}",
            f"{float(send.get('server_total_ms', 0.0)):.3f}",
            int(send.get("num_spi_frames", 0)),
            int(bool(send.get("batch_active", False))),
            f"{float(gravity[0]):.6f}",
            f"{float(gravity[1]):.6f}",
            f"{float(gravity[2]):.6f}",
            f"{self._tilt_from_observation(obs_flat):.6f}",
            int(bool(torque_limit_info.get("rear_torque_boost_active", False))),
            f"{float(torque_limit_info.get('rear_torque_boost_elapsed_s', 0.0)):.6f}",
            f"{float(torque_limit_info.get('rear_torque_limit_nm', 0.0)):.6f}",
        ]
        for i in range(12):
            row.extend(
                [
                    f"{float(action_raw[i]):.6f}",
                    f"{float(action_filtered[i]):.6f}",
                    f"{float(gait_offset[i]):.6f}",
                ]
            )
        for i in range(12):
            row.extend(
                [
                    f"{float(q_raw_des[i]):.6f}",
                    f"{float(q_des[i]):.6f}",
                    f"{float(current_q[i]):.6f}",
                    f"{float(current_dq[i]):.6f}",
                    f"{float(raw_pd_torque[i]):.6f}",
                    f"{float(limited_torque[i]):.6f}",
                    f"{float(command_torque_ff[i]):.6f}",
                    f"{float(measured_torque[i]):.6f}",
                    int(limited_mask[i]),
                    int(motor_online[i]),
                    int(motor_error_code[i] != 0),
                    f"{float(motor_age_ms[i]):.3f}",
                ]
            )
        self._debug_csv_writer.writerow(row)
        self._debug_csv_rows_since_flush += 1
        if self._debug_csv_rows_since_flush >= self.debug_csv_flush_every_n:
            self._debug_csv_file.flush()
            self._debug_csv_rows_since_flush = 0

    def control_loop(self):
        self._record_control_loop_rate()
        self._last_onnx_infer_dt_ms = 0.0
        filter_state = None
        previous_last_action = None
        try:
            measured_dt = self.get_control_dt()
            self.update_smoothed_command(measured_dt)
            obs, info = self.obs_builder.build_obs()
            obs, info, estimator_sync = self._guard_estimator_sync(obs, info)

            max_age = float(np.max(info["age_ms"]))
            self._last_state_age_ms = max_age
            self._last_motor_state_poll_dt_ms = float(
                info.get("motor_state_poll_dt_ms", 0.0)
            )

            if max_age > self.max_motor_age_ms:
                stale = np.where(np.asarray(info["age_ms"]) > self.max_motor_age_ms)[0]
                if self.recheck_stale_motor_once:
                    obs, info = self.recheck_stale_motor_feedback(obs, info, stale)
                    max_age = float(np.max(info["age_ms"]))
                if max_age > self.max_motor_age_ms:
                    self.get_logger().warn(
                        f"[PARITY_FIXED][SAFE] motor feedback too old: {max_age:.1f} ms; "
                        "cycle not committed"
                    )
                    return

            if self.require_online and not np.all(info["online"]):
                self.get_logger().warn(
                    "[PARITY_FIXED][SAFE] some motors offline; cycle not committed"
                )
                return
            if self._startup_stand_state != "complete":
                self.handle_startup_stand(obs, info, measured_dt, max_age)
                return

            # Keep the policy running on the current command by default, just
            # as the simulation does between command segments. An explicit
            # timeout stand-hold remains available as a real-machine option.
            if not self.command_is_fresh():
                if self.enable_command_timeout_stand_hold:
                    self._handle_stale_command(obs, info, max_age)
                    return
                # Simulation keeps its current command until the next command
                # segment. Preserve that behavior when the real publisher has
                # a gap; the fast command publisher sends an explicit zero at
                # the end of its sequence.
                self._stale_command_hold_active = False

            self._stale_command_hold_active = False
            tilt_rad = self._tilt_from_observation(obs)
            if self.enable_tilt_protection and tilt_rad > self.max_tilt_rad:
                self._handle_tilt_protection(obs, info, max_age)
                return
            if self._tilt_protection_active:
                self._tilt_protection_active = False
                self._reset_contract_state(reset_phase=True)
                self.get_logger().warn(
                    "[SAFE][TILT] projected gravity recovered; resuming only "
                    "after a fresh command state reset."
                )

            if self.stand_only:
                self.handle_stand_only(obs, info, max_age)
                return
            if self.joint_probe_enable:
                self.handle_joint_probe(obs, info, max_age)
                return
            if not self.policy_enable or self.policy is None:
                self.handle_stand_only(obs, info, max_age, mode="policy_disabled")
                return

            obs = self.ensure_policy_obs_dim(obs)
            zero_cmd_stand = self.zero_cmd_stand_active()
            self._last_zero_cmd_stand_active = zero_cmd_stand
            self._last_action_cmd_gate_scale = 1.0
            nominal_dt = 1.0 / max(float(self.policy_hz), 1.0e-6)

            action_raw = np.zeros(12, dtype=np.float32)
            filtered_action = np.zeros(12, dtype=np.float32)
            if zero_cmd_stand:
                self._reset_contract_state(reset_phase=False)
                target_policy_abs = self.obs_builder.mapper.default_joint_angle.copy()
                self._last_gait_offset_policy = np.zeros(12, dtype=np.float32)
                self._contract_gait_gate = 0.0
                cpg_action_info = {
                    "action_mode": "zero_cmd_stand",
                    "command_gate_scale": 0.0,
                    "zero_cmd_stand": True,
                    "frequency": 0.0,
                    "phase": float(self.obs_builder.last_gait_phase),
                    "base_phase": float(self._contract_phase),
                    "phase_lead_sec": float(self.gait_phase_lead_sec),
                    "gait_gate": 0.0,
                    "model_gait_period_s": self.model_gait_phase_period,
                    "gait_period_scale": self.contract_gait_period_scale,
                    "active_gait_period_s": self.gait_phase_period,
                    "protection_mode": "pd_equivalent",
                }
            else:
                filter_state = self._capture_filter_state()
                previous_last_action = self.obs_builder.last_action.copy()
                infer_t0 = time.perf_counter()
                action_raw = self.policy.infer(obs)
                self._last_onnx_infer_dt_ms = (
                    time.perf_counter() - infer_t0
                ) * 1000.0
                filtered_action = self.contract_action_filter.step(
                    action_raw,
                    dt=nominal_dt,
                )
                target_policy_abs, cpg_action_info = self.action_to_policy_target_abs_by_mode(
                    filtered_action,
                    dt=nominal_dt,
                )
                cpg_action_info["command_gate_scale"] = 1.0
                cpg_action_info["zero_cmd_stand"] = False
                cpg_action_info["phase"] = float(self.obs_builder.last_gait_phase)
                cpg_action_info["base_phase"] = float(self._contract_phase)
                cpg_action_info["phase_lead_sec"] = float(self.gait_phase_lead_sec)
                cpg_action_info["frequency"] = 1.0 / self.gait_phase_period
                cpg_action_info["gait_gate"] = float(self._contract_gait_gate)
                cpg_action_info["model_gait_period_s"] = self.model_gait_phase_period
                cpg_action_info["gait_period_scale"] = self.contract_gait_period_scale
                cpg_action_info["active_gait_period_s"] = self.gait_phase_period
                cpg_action_info["protection_mode"] = "pd_equivalent"

            target_real_raw = self.obs_builder.mapper.policy_target_to_real_target(
                target_policy_abs,
                clamp=True,
            )
            current_q = np.asarray(info["q_real"], dtype=np.float32).reshape(12)
            current_dq = np.asarray(info["dq_real"], dtype=np.float32).reshape(12)
            torque_budget_nm = self.compute_torque_safety_budget_nm()
            pre_limit_info = self._disabled_pre_limit_info(
                target_real_raw,
                current_q,
                torque_budget_nm,
            )
            motor_vel_cmd = np.zeros(12, dtype=np.float32)

            if zero_cmd_stand:
                self._reset_rear_torque_boost_window()
            else:
                self._maybe_activate_rear_torque_boost(
                    target_real_raw,
                    current_q,
                    current_dq,
                    tilt_rad,
                )

            if self.enable_send:
                target_real, torque_limit_info = self._apply_pd_equivalent_limit(
                    target_real_raw,
                    current_q,
                    current_dq,
                )
            else:
                active_limits = self._active_torque_limits_real()
                target_real, torque_limit_info = self.contract_torque_limiter.limit_with_torque_feedforward(
                    q_raw=target_real_raw,
                    q_current=current_q,
                    dq_current=current_dq,
                    kp=self.send_kp_real,
                    kd=self.send_kd_real,
                    torque_limits=active_limits,
                    qd_target=np.zeros(12, dtype=np.float32),
                    torque_ff=np.zeros(12, dtype=np.float32),
                    torque_ff_limit=self.motion_torque_ff_limit_nm,
                )
                torque_limit_info["joint_limit_adjusted_mask"] = np.zeros(
                    12, dtype=bool
                )
                torque_limit_info["rear_torque_boost_active"] = bool(
                    self._rear_torque_boost_active()
                )
                torque_limit_info["rear_torque_boost_elapsed_s"] = float(
                    self._rear_torque_boost_elapsed()
                )
                torque_limit_info["rear_torque_limit_nm"] = float(
                    np.max(active_limits[6:12])
                )
                torque_limit_info["protection_mode"] = "pd_equivalent_preview_no_send"

            error = target_real - current_q
            self.publish_array(self.pub_obs, obs)
            self.publish_array(self.pub_action_raw, action_raw)
            self.publish_array(self.pub_action, filtered_action)
            self.publish_array(self.pub_target, target_real)

            send_meta = self._empty_send_meta()
            send_ok = True
            if self.enable_send:
                send_ok = self.send_motion_batch(
                    target_real,
                    info,
                    motor_vel_cmd=motor_vel_cmd,
                    motor_torque_ff=torque_limit_info.get(
                        "torque_ff_cmd",
                        np.full(
                            12,
                            float(self.send_torque),
                            dtype=np.float32,
                        ),
                    ),
                )
                send_meta = dict(self._last_send_meta)
            else:
                # A no-send preview is a valid simulated policy step.
                send_meta.update(
                    {
                        "send_attempted": False,
                        "send_ok": True,
                        "send_error": "preview_no_send",
                    }
                )

            cpg_action_info = dict(cpg_action_info)
            cpg_action_info["gait_offset_policy"] = np.asarray(
                self._last_gait_offset_policy, dtype=np.float32
            ).copy()
            cpg_action_info["estimator_sync"] = dict(estimator_sync)
            cpg_action_info["send_meta"] = dict(send_meta)

            self.maybe_write_policy_csv(
                obs=obs,
                action_raw=action_raw,
                action_policy_obs=filtered_action,
                action=filtered_action,
                q_des=target_real,
                current_q=current_q,
                current_dq=current_dq,
                error=error,
                max_age=max_age,
                measured_torque=info.get("torque_real"),
                motor_temp=info.get("temp_real"),
                motor_online=info.get("online"),
                motor_error_code=info.get("error_code"),
                motor_age_ms=info.get("age_ms"),
                q_raw_des=target_real_raw,
                q_smooth_des=target_real_raw,
                motor_vel_cmd=motor_vel_cmd,
                smoothing_info=pre_limit_info,
                torque_limit_info=torque_limit_info,
                cpg_action_info=cpg_action_info,
                mode="pure_rl_pd_equivalent_fixed",
            )

            if zero_cmd_stand:
                self.obs_builder.set_last_action(np.zeros(12, dtype=np.float32))
                self._walking_active_prev = False
                self._consecutive_send_failures = 0
                return

            if send_ok:
                self.obs_builder.set_last_action(filtered_action)
                self._walking_active_prev = True
                self._advance_contract_phase()
                self._consecutive_send_failures = 0
            else:
                self._restore_filter_state(filter_state)
                if previous_last_action is not None:
                    self.obs_builder.set_last_action(previous_last_action)
                self._walking_active_prev = False
                self._consecutive_send_failures += 1
                if self._consecutive_send_failures >= self.send_failure_warn_count:
                    self.get_logger().error(
                        "[PARITY_FIXED][SAFE] repeated send failures="
                        f"{self._consecutive_send_failures}; gait phase remains frozen"
                    )

            if self._control_summary_log_due():
                tau_raw = np.asarray(
                    torque_limit_info["tau_raw_signed"], dtype=np.float32
                )
                tau_safe = np.asarray(
                    torque_limit_info["tau_safe_signed"], dtype=np.float32
                )
                self.get_logger().info(
                    f"[PARITY_FIXED] cmd={self.cmd.tolist()} "
                    f"phase={self.obs_builder.last_gait_phase:.3f} "
                    f"raw_action_max={float(np.max(np.abs(action_raw))):.3f} "
                    f"filtered_max={float(np.max(np.abs(filtered_action))):.3f} "
                    f"tau_raw_max={float(np.max(np.abs(tau_raw))):.2f}Nm "
                    f"tau_safe_max={float(np.max(np.abs(tau_safe))):.2f}Nm "
                    f"clipped={torque_limit_info['limited_count']}/12 "
                    f"send_ok={send_ok} max_age={max_age:.1f}ms"
                )

        except Exception as exc:
            self._restore_filter_state(filter_state)
            if previous_last_action is not None:
                self.obs_builder.set_last_action(previous_last_action)
            self.get_logger().error(
                f"[PARITY_FIXED] policy loop error; cycle not committed: {exc}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = MydogPolicyParityFixedNode()
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
