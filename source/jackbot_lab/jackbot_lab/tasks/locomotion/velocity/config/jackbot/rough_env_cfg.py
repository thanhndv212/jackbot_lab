# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    EventCfg,
    ObservationsCfg,
)

import jackbot_lab.tasks.locomotion.velocity.mdp as mdp
from jackbot_lab.robots.jackbot import JACKBOT_MINIMAL_CFG  # isort: skip


@configclass
class JackbotEvents(EventCfg):
    # Randomize the mass of the robot
    add_base_mass = EventTerm(
        func=mdp.randomize_base_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # Randomize the joint friction and amature of the robot
    robot_joint_friction_and_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.95, 1.05),
            "armature_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class JackbotRewards(RewardsCfg):

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    lin_vel_z_l2 = None

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.5,
        params={
            "target_height": 0.78,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="(right|left)_base_link"
            ),
            "threshold": 2.0,
        },
    )

    # Penalize the deviation from the desired orientation of the feet
    right_foot_orientaion = RewTerm(
        func=mdp.feet_orientation,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="right_base_link"
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names="right_base_link"),
        },
    )
    left_foot_orientaion = RewTerm(
        func=mdp.feet_orientation,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="left_base_link"
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names="left_base_link"),
        },
    )

    # Penalize the deviation from the desired y-axis distance between the feet
    feet_keep_distance = RewTerm(
        func=mdp.feet_keep_distance,
        weight=-0.8,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names="(right|left)_base_link"
            ),
        },
    )

    # Reward for the contact force on the feet during the gait cycle
    clock_frc = RewTerm(
        func=mdp.feet_clock_frc,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="(right|left)_base_link"
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names="(right|left)_base_link"
            ),
        },
    )

    # Reward for the velocity of the feet during the gait cycle
    clock_vel = RewTerm(
        func=mdp.feet_clock_vel,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names="(right|left)_base_link"
            ),
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_joint_6")
        },
    )

    # Penalize the deviation from default of the joints that are not essential for locomotion
    joint_deviation_j2_j3 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.35,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names="(right|left)_leg_joint_(2|3)"
            )
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="torso_.*"
            ),
            "threshold": 1.0,
        },
    )


@configclass
class JackbotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    gait_step_num: int = 50
    rewards: JackbotRewards = JackbotRewards()
    terminations: TerminationsCfg = TerminationsCfg()
    events: JackbotEvents = JackbotEvents()
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = JACKBOT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # Randomization
        self.events.add_base_mass.params["asset_cfg"].body_names = [".*"]
        self.events.add_base_mass.params["mass_distribution_params"] = (
            0.9,
            1.1,
        )
        self.events.add_base_mass.params["operation"] = "scale"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params[
            "asset_cfg"
        ].body_names = ["torso.*"]
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.physics_material.params["static_friction_range"] = (
            0.2,
            1.5,
        )
        self.events.physics_material.params["dynamic_friction_range"] = (
            0.2,
            1.5,
        )

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "(torso_.*|right_leg_link_1|left_leg_link_1|right_leg_link_2|left_leg_link_2|right_leg_link_3|left_leg_link_3|right_leg_link_4|left_leg_link_4|right_leg_link_5|left_leg_link_5)"
        ]

        # Rewards
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7

        # Observations
        self.observations.policy.gait_phase = ObsTerm(func=mdp.gait_phase)

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class JackbotRoughEnvCfg_PLAY(JackbotRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
