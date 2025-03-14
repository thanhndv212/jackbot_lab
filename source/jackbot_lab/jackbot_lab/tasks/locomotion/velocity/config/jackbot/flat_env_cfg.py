# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import JackbotRoughEnvCfg


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
        self.rewards.is_terminated.weight = -200

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -0.005

        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.5

        self.rewards.base_height_l2.weight = -0.5
        self.rewards.base_height_l2.params["target_height"] = 0.81
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [
            self.base_link_name
        ]

        self.rewards.body_lin_acc_l2.weight = 0

        # Joint penalties
        self.rewards.joint_deviation.weight = -0.0
        self.rewards.create_joint_deviation_l1_rewterm(
            "joint_deviation_other_l1",
            -0.2,
            [
                ".*_yaw_hip_joint",
                ".*_roll_hip_joint",
                ".*_yaw_knee_joint",
                ".*_shoulder_joint",
                ".*_wrist_joint",
            ],
        )
        self.rewards.create_joint_deviation_l1_rewterm(
            "joint_deviation_waist_l1", -0.4, ["yaw_waist_joint"]
        )
        self.rewards.create_joint_deviation_l1_rewterm(
            "joint_deviation_elbow_l1", -0.05, [".*_pitch_elbow_joint"]
        )
        self.rewards.create_joint_deviation_l1_rewterm(
            "joint_deviation_knee_l1", -0.1, [".*_pitch_knee_joint"]
        )

        self.rewards.joint_acc_l2.weight = -1.0e-7
        self.rewards.joint_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*pitch_hip_.*", ".*pitch_knee_.*", ".*pitch_ankle_.*"]
        )
        self.rewards.joint_torques_l2.weight = -1.5e-7
        self.rewards.joint_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*pitch_hip_.*", ".*pitch_knee_.*", ".*pitch_ankle_.*"]
        )

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.005

        # Contact sensor
        self.rewards.undesired_contacts.weight = 0
        self.rewards.contact_forces.weight = 0

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5

        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5

        # Feet rewards
        self.rewards.feet_air_time.weight = 1.0

        self.rewards.feet_air_time.params["threshold"] = 0.4

        self.rewards.feet_contact.weight = 0

        self.rewards.feet_slide.weight = -0.05

        # Others
        self.rewards.joint_power.weight = 0
        self.rewards.stand_still_without_cmd.weight = 0.0

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.heading_command = True

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "JackbotFlatEnvCfg":
            self.disable_zero_weight_rewards()
