# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from jackbot_lab.tasks.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    EventCfg,
    ObservationsCfg,
)

import jackbot_lab.tasks.locomotion.velocity.mdp as mdp
from jackbot_lab.assets.jackbot import JACKBOT_MINIMAL_CFG  # isort: skip


@configclass
class JackbotRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["foot.*"]
            ),
            "threshold": 0.6,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["foot.*"]
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=["foot.*"]),
        },
    )

    # Penalize the deviation from the desired y-axis distance between the feet
    knee_keep_distance = RewTerm(
        func=mdp.feet_keep_distance,
        weight=-0.8,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names="knee.*"
            ),
        },
    )

    lowerleg_keep_distance = RewTerm(
        func=mdp.feet_keep_distance,
        weight=-0.8,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="tibia.*"),
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_ankle_joint"],
            )
        },
    )

    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_joint",
                    ".*_shoulder_joint",
                    ".*_elbow_joint",
                    ".*_knee_joint",
                    ".*_waist_joint",
                    ".*_wrist_joint",
                ],
            )
        },
    )


@configclass
class JackbotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: JackbotRewardsCfg = JackbotRewardsCfg()

    base_link_name = "pelvis"
    foot_link_name = "foot.*"

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        # Scene
        self.scene.robot = JACKBOT_MINIMAL_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.height_scanner.prim_path = (
            "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        )
        self.scene.height_scanner_base.prim_path = (
            "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        )

        # ------------------------------Events------------------------------

        # Randomization
        self.events.randomize_rigid_body_mass.params[
            "asset_cfg"
        ].body_names = [self.base_link_name]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [
            self.base_link_name
        ]
        # self.events.randomize_apply_external_force_torque.params[
        #     "asset_cfg"
        # ].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = -200

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = 0

        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -5.0

        self.rewards.base_height_l2.weight = -0.5
        self.rewards.base_height_l2.params["target_height"] = 0.832
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [
            self.base_link_name
        ]

        self.rewards.body_lin_acc_l2.weight = 0

        # Joint penalties
        self.rewards.joint_deviation.weight = -0.05
        self.rewards.joint_acc_l2.weight = -1.25e-7
        self.rewards.joint_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"]
        )
        self.rewards.joint_torques_l2.weight = -1.5e-7
        self.rewards.joint_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"]
        )

        # Action penalties
        # self.rewards.action_rate_l2.weight = -0.005
        # UNUESD self.rewards.action_l2.weight = 0.0

        # Contact sensor
        self.rewards.undesired_contacts.weight = 0
        self.rewards.contact_forces.weight = 0

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_lin_vel_xy_exp.func = (
            mdp.track_lin_vel_xy_yaw_frame_exp
        )
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5

        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5

        # Feet rewards
        self.rewards.feet_air_time.weight = 2.0
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_air_time.params["threshold"] = 0.4

        self.rewards.feet_contact.weight = 0

        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]

        # Others
        self.rewards.joint_power.weight = 0
        self.rewards.stand_still_without_cmd.weight = 0.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "JackbotRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            "pelvis.*",
            "knee.*",
            "tibia.*",
            "shoulder.*",
            "elbow.*",
            "hand.*",
        ]

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
