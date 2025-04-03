# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from jackbot_lab.tasks.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
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

    feet_contact = RewTerm(
        func=mdp.feet_contact,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["foot.*"]
            ),
            "command_name": "base_velocity",
            "expect_contact_num": 1,
        },
    )
    # Penalize the deviation from the desired y-axis distance between the feet
    knee_keep_distance = RewTerm(
        func=mdp.feet_keep_distance,
        weight=-0.8,
        params={
            "dist_min": 0.2,
            "dist_max": 0.5,
            "asset_cfg": SceneEntityCfg("robot", body_names="knee.*"),
        },
    )

    feet_keep_distance = RewTerm(
        func=mdp.feet_keep_distance,
        weight=-0.8,
        params={
            "dist_min": 0.2,
            "dist_max": 0.4,
            "asset_cfg": SceneEntityCfg("robot", body_names="foot.*"),
        },
    )

    # Penalize the deviation from the desired orientation of the feet
    right_foot_orientaion = RewTerm(
        func=mdp.feet_orientation,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="footR"
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names="footR"),
        },
    )
    left_foot_orientaion = RewTerm(
        func=mdp.feet_orientation,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="footL"
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names="footL"),
        },
    )

    # Reward for the contact force on the feet during the gait cycle
    clock_frc = RewTerm(
        func=mdp.feet_clock_frc,
        weight=0.3,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="foot.*"
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names="foot.*"),
        },
    )

    # Reward for the velocity of the feet during the gait cycle
    clock_vel = RewTerm(
        func=mdp.feet_clock_vel,
        weight=0.4,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="foot.*"),
        },
    )

    leg_coordination = RewTerm(
        func=mdp.leg_coordination_reward,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="foot.*"),
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

    # Add hip movement reward
    hip_movement = RewTerm(
        func=mdp.joint_velocity_reward,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_pitch_hip_joint", ".*_roll_hip_joint"],
            ),
            "target_velocity": 1.0,
        },
    )

    # Add hip extension reward
    hip_extension = RewTerm(
        func=mdp.joint_position_reward,
        weight=0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_pitch_hip_joint"],
            ),
            "target_position": 0.3,
            "position_range": 0.2,
        },
    )

    knee_extension = RewTerm(
        func=mdp.joint_position_reward,
        weight=0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_pitch_knee_joint"],
            ),
            "target_position": 0.3,
            "position_range": 0.2,
        },
    )

    # Add new reward terms for gait symmetry and step length
    gait_symmetry = RewTerm(
        func=mdp.gait_symmetry_reward,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="foot.*"),
        },
    )

    step_length = RewTerm(
        func=mdp.step_length_reward,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="foot.*"),
            "target_step_length": 0.4,
            "min_step_length": 0.2,
        },
    )


@configclass
class JackbotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    gait_step_num: int = 50
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
        # self.events.randomize_com_positions.params["asset_cfg"].body_names = [
        #     self.base_link_name
        # ]
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
        self.rewards.action_rate_l2.weight = -0.005
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

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "JackbotRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------ Observations ----------------------------
        self.observations.policy.gait_phase = ObsTerm(func=mdp.gait_phase)

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
