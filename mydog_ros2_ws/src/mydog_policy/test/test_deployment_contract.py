import json
from pathlib import Path

import numpy as np
import onnx

from mydog_policy.semantic_mapper import JointSemanticMapper


def _contract():
    model_path = Path(__file__).parents[1] / "resource" / "policy.onnx"
    model = onnx.load(str(model_path), load_external_data=False)
    metadata = {item.key: item.value for item in model.metadata_props}
    return json.loads(metadata["fanfan_deployment_config"])


def test_policy_contract_matches_graph_and_mapper():
    contract = _contract()
    assert contract["schema_version"] == 1
    assert contract["dimensions"] == {"observations": 52, "actions": 12}

    mapper = JointSemanticMapper(
        contract["joint_names"], contract["default_joint_angles"]
    )
    assert mapper.get_policy_joint_names() == contract["joint_names"]
    assert mapper.policy_to_real_index.tolist() == [3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 10, 11]

    real_default = mapper.real_default_pose_for_motor_order()
    np.testing.assert_allclose(
        real_default,
        [0.0, 0.563, -0.95, 0.0, -0.563, 0.95,
         0.0, -0.563, 0.95, 0.0, 0.563, -0.95],
        atol=1e-6,
    )
    q_obs, dq_obs = mapper.real_to_policy_q_dq(real_default, np.zeros(12))
    np.testing.assert_allclose(q_obs, 0.0, atol=1e-6)
    np.testing.assert_allclose(dq_obs, 0.0, atol=1e-6)


def test_contract_control_period_and_action_scales():
    contract = _contract()
    control = contract["control"]
    assert control["sim_dt"] * control["decimation"] == 0.02
    np.testing.assert_allclose(
        control["action_scale"],
        [0.08, 0.18, 0.18, 0.08, 0.18, 0.18, 0.08, 0.20, 0.20, 0.08, 0.20, 0.20],
    )
    np.testing.assert_allclose(
        control["stiffness"],
        [60.0, 70.0, 70.0] * 4,
    )
    np.testing.assert_allclose(
        control["damping"],
        [0.6, 0.8, 0.8] * 4,
    )


def test_policy_gain_values_are_permuted_without_joint_signs():
    contract = _contract()
    mapper = JointSemanticMapper(
        contract["joint_names"], contract["default_joint_angles"]
    )
    real_values = mapper.policy_values_to_real_order(np.arange(12, dtype=np.float32))
    np.testing.assert_allclose(
        real_values,
        [3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 10, 11],
    )


def test_mapper_limits_match_updated_fanfan_urdf():
    contract = _contract()
    mapper = JointSemanticMapper(
        contract["joint_names"], contract["default_joint_angles"]
    )
    for i, name in enumerate(mapper.policy_joint_names):
        limits = [mapper.policy_lower_limit[i], mapper.policy_upper_limit[i]]
        if "_hip_" in name:
            np.testing.assert_allclose(limits, [-0.8, 0.8])
        elif "_thigh_" in name:
            np.testing.assert_allclose(limits, [-0.5, 4.0])
        elif "_calf_" in name:
            np.testing.assert_allclose(limits, [-2.7, -0.85])


def test_onnx_joint_order_is_fl_fr_rl_rr():
    contract = _contract()
    assert contract["joint_names"][:3] == [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    ]


def test_gait_reference_has_calf_swing_amplitude():
    contract = _contract()
    gait = contract["gait"]
    assert gait["thigh_amplitude"] == 0.0
    assert abs(float(gait["calf_amplitude"])) > 0.1
    stance_ratio = float(gait["stance_ratio"])
    phase = stance_ratio + 0.5 * (1.0 - stance_ratio)
    swing = (phase - stance_ratio) / (1.0 - stance_ratio)
    smooth = swing * swing * (3.0 - 2.0 * swing)
    calf_ref = float(gait["calf_amplitude"]) * np.sin(np.pi * smooth)
    assert abs(calf_ref) > 0.1
