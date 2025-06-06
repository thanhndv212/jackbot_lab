# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import JackbotRoughEnvCfg
from torch import pi

deg_to_rad = pi / 180.0

@configclass
class JackbotFlatEnvCfg(JackbotRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = -200.0

        # Velocity-tracking rewards (essential for walking)
        self.rewards.track_lin_vel_xy_exp.weight = 5.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5

        # Root penalties (essential for stability)
        self.rewards.lin_vel_z_l2.weight = -0.8
        self.rewards.ang_vel_xy_l2.weight = -0.6
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.base_height_exp.weight = -2.000000000000001
        self.rewards.base_height_exp.params["target_height"] = 0.828
        self.rewards.body_lin_acc_l2.weight = 0.0

        # Joint penalties (minimal for clock-based walking)\
        self.rewards.joint_deviation_hip_l1.weight = -0.1
        self.rewards.joint_deviation_knee_l1.weight = -0.1
        self.rewards.joint_deviation_ankle_l1.weight = -0.1
        self.rewards.joint_deviation_other_l1.weight = -0.4

        self.rewards.joint_acc_l2.weight = -1.25e-07

        self.rewards.joint_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                ".*_hip_.*",
                ".*_knee_.*",
                ".*_ankle_.*",
            ],
        )
        self.rewards.joint_torques_l2.weight = -1.5e-07
        self.rewards.joint_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                # ".*_hip_.*",
                ".*_knee_.*",
                # ".*_ankle_.*",
            ],
        )
        self.rewards.joint_vel_l2.weight = -0.0
        self.rewards.joint_vel_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                # ".*_pitch_hip_.*",
                ".*_pitch_knee_.*",
                ".*_ankle_.*",
            ],
        )
        self.rewards.dof_pos_limits.weight = -0.0
        self.rewards.dof_pos_limits.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                ".*_hip_.*",
                ".*_knee_.*",
                ".*_ankle_.*",
            ],
        )
        # Action penalties
        self.rewards.action_rate_l2.weight = -0.005

        # Contact sensor
        self.rewards.undesired_contacts.weight = 0.0
        self.rewards.contact_forces.weight = 0.0

        # Feet rewards (essential for clock-based walking)
        self.rewards.feet_air_time.weight = 2.0
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_contact.weight = -0.3
        self.rewards.feet_slide_exp.weight = -0.5
        self.rewards.knee_keep_distance.weight = 0.0
        self.rewards.feet_keep_distance.weight = 0.0
        self.rewards.right_foot_orientaion.weight = 0.0
        self.rewards.left_foot_orientaion.weight = 0.0

        # Clock-based rewards (core of the policy)
        self.rewards.clock_frc.weight = 0.8
        self.rewards.clock_vel.weight = 0.6
        self.rewards.leg_coordination.weight = 0.6

        # Gait symmetry and step length (essential for walking)
        self.rewards.gait_symmetry.weight = 0.5
        self.rewards.step_length.weight = 0.4
        self.rewards.step_length.params["target_step_length"] = 0.4
        self.rewards.step_length.params["min_step_length"] = 0.2

        # Joint movement rewards (minimal for clock-based walking)
        self.rewards.hip_movement.weight = 0.0
        self.rewards.hip_movement.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[".*_pitch_hip_joint"],
        )
        self.rewards.hip_movement.params["target_velocity"] = 1.0

        self.rewards.hip_extension.weight = 0.0
        self.rewards.hip_extension.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[".*_pitch_hip_joint"],
        )
        self.rewards.hip_extension.params["target_position"] = -12.0 * deg_to_rad
        self.rewards.hip_extension.params["position_range"] = 12.0 * deg_to_rad

        self.rewards.knee_extension.weight = 0.0
        self.rewards.knee_extension.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[".*_pitch_knee_joint"],
        )
        self.rewards.knee_extension.params["target_position"] = (
            45.0 * deg_to_rad
        )
        self.rewards.knee_extension.params["position_range"] = 15.0 * deg_to_rad

        # New gait-specific rewards with higher weights for flat terrain
        self.rewards.air_time_balance.weight = 0.6
        self.rewards.feet_alignment.weight = 0.5

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.1, 0.1)
        self.commands.base_velocity.heading_command = True

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "JackbotFlatEnvCfg":
            self.disable_zero_weight_rewards()
