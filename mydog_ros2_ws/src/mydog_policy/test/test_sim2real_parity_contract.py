from pathlib import Path

import numpy as np

from mydog_policy.policy_contract import (
    ActionFilterConfig,
    ContractPolicyActionFilter,
    PDTorqueEquivalentLimiter,
)


def test_action_filter_matches_training_reference_step():
    control = {
        "filter_policy_actions": True,
        "policy_action_filter_alpha": 0.26,
        "policy_action_rate_limits": [8.333333, 7.073171, 11.463415] * 4,
        "policy_action_accel_limits": [288.888889, 234.146341, 400.0] * 4,
    }
    filt = ContractPolicyActionFilter.from_control(control)
    dt = 0.02
    action = np.linspace(-1.0, 1.0, 12, dtype=np.float32)

    expected_action = np.zeros(12, dtype=np.float32)
    expected_velocity = np.zeros(12, dtype=np.float32)
    rate = np.asarray(control["policy_action_rate_limits"], dtype=np.float32)
    accel = np.asarray(control["policy_action_accel_limits"], dtype=np.float32)

    for _ in range(10):
        desired = expected_action + 0.26 * (action - expected_action)
        desired_velocity = np.clip((desired - expected_action) / dt, -rate, rate)
        dv = np.clip(
            desired_velocity - expected_velocity,
            -accel * dt,
            accel * dt,
        )
        next_velocity = expected_velocity + dv
        next_action = expected_action + next_velocity * dt
        crossed = (desired - expected_action) * (desired - next_action) < 0.0
        next_action = np.where(crossed, desired, next_action)
        next_velocity = (next_action - expected_action) / dt
        expected_action = next_action.astype(np.float32)
        expected_velocity = next_velocity.astype(np.float32)

        actual = filt.step(action, dt)
        np.testing.assert_allclose(actual, expected_action, atol=1.0e-7)
        np.testing.assert_allclose(filt.action_velocity, expected_velocity, atol=1.0e-6)


def test_disabled_action_filter_is_identity():
    config = ActionFilterConfig(
        enabled=False,
        alpha=1.0,
        rate_limits=np.ones(12, dtype=np.float32),
        accel_limits=np.ones(12, dtype=np.float32),
    )
    filt = ContractPolicyActionFilter(config)
    action = np.linspace(-0.8, 0.8, 12, dtype=np.float32)
    np.testing.assert_array_equal(filt.step(action, 0.02), action)


def test_pd_equivalent_target_reconstructs_clipped_signed_torque():
    limiter = PDTorqueEquivalentLimiter()
    q = np.linspace(-0.2, 0.2, 12, dtype=np.float32)
    dq = np.linspace(-2.0, 2.0, 12, dtype=np.float32)
    q_raw = q + np.linspace(-0.4, 0.4, 12, dtype=np.float32)
    kp = np.asarray([60.0, 70.0, 70.0] * 4, dtype=np.float32)
    kd = np.asarray([1.2, 1.6, 1.6] * 4, dtype=np.float32)
    limits = np.full(12, 10.0, dtype=np.float32)

    q_safe, info = limiter.limit(
        q_raw=q_raw,
        q_current=q,
        dq_current=dq,
        kp=kp,
        kd=kd,
        torque_limits=limits,
        qd_target=np.zeros(12, dtype=np.float32),
        torque_ff=np.zeros(12, dtype=np.float32),
    )

    reconstructed = kp * (q_safe - q) - kd * dq
    expected = np.clip(kp * (q_raw - q) - kd * dq, -limits, limits)
    np.testing.assert_allclose(reconstructed, expected, atol=2.0e-5)
    np.testing.assert_allclose(info["tau_reconstructed_signed"], expected, atol=2.0e-5)
    assert info["limited_count"] > 0


def test_parity_launch_disables_legacy_closed_loop_modifiers():
    launch_file = (
        Path(__file__).parents[1]
        / "launch"
        / "sim2real_omni_parity.launch.py"
    )
    source = launch_file.read_text(encoding="utf-8")
    assert 'executable="mydog_policy_parity_node"' in source
    assert '"enable_target_smoothing": False' in source
    assert '"enable_torque_error_limit": False' in source
    assert '"enable_policy_action_cmd_gate": False' in source
    assert '"enable_velocity_ff": False' in source
    assert '"deployment_gait_phase_period_scale": LaunchConfiguration(' in source
    assert 'DeclareLaunchArgument("gait_period_scale", default_value="1.00")' in source
    assert '"gait_phase_lead_sec": LaunchConfiguration("gait_phase_lead_sec")' in source
    assert 'DeclareLaunchArgument("gait_phase_lead_sec", default_value="0.00")' in source
    assert 'DeclareLaunchArgument("motor_torque_limit_nm", default_value="10.0")' in source


def test_parity_node_applies_only_explicit_gait_period_scaling():
    node_file = (
        Path(__file__).parents[1]
        / "mydog_policy"
        / "sim2real_parity_node.py"
    )
    source = node_file.read_text(encoding="utf-8")
    assert "self.model_gait_phase_period" in source
    assert "self.contract_gait_period_scale" in source
    assert "self.model_gait_phase_period * self.contract_gait_period_scale" in source
    assert "sim2real parity gait period scale must be finite and in [1.0, 1.5]" in source
    assert 'cpg_action_info["frequency"] = 1.0 / self.gait_phase_period' in source


def test_parity_node_applies_one_shared_phase_lead_to_obs_and_gait():
    node_file = (
        Path(__file__).parents[1]
        / "mydog_policy"
        / "sim2real_parity_node.py"
    )
    source = node_file.read_text(encoding="utf-8")
    assert 'self.declare_parameter("gait_phase_lead_sec", 0.0)' in source
    assert "self.gait_phase_lead_sec / max(self.gait_phase_period, 1.0e-6)" in source
    assert "self._contract_phase + self.contract_phase_lead_cycles" in source
    assert "self.obs_builder.last_gait_phase = self._effective_contract_phase()" in source
    assert 'cpg_action_info["phase_lead_sec"]' in source
    assert "gait_phase_lead_sec must be finite and in [0.00, 0.10] seconds" in source
