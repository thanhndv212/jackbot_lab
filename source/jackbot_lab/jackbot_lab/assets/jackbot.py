"""Configuration for Jackbot robot.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from torch import pi

deg_to_rad = pi / 180.0
##
# Configuration - Actuators.
##

JACKBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/thanh-nguyen/Downloads/robot_models/jackbot/v1_newjointlim/jackbot_v1_nobaselink_fixedtorso.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.832),
        joint_pos={
            ".*pitch_ankle_joint": -17.5 * deg_to_rad,
            ".*pitch_elbow_joint": -60.0 * deg_to_rad,
            ".*pitch_knee_joint": 45.0 * deg_to_rad,
            ".*pitch_hip_joint": -12.0 * deg_to_rad,
            ".*pitch_shoulder_joint": 15.0 * deg_to_rad,
            ".*roll_ankle_joint": 0.0,
            ".*roll_hip_joint": 0.0,
            ".*roll_shoulder_joint": 0.0,
            ".*yaw_hip_joint": 0.0,
            ".*yaw_knee_joint": 0.0,
            ".*yaw_shoulder_joint": 0.0,
            ".*yaw_wrist_joint": 0.0,
            ".*_waist_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_waist_joint",
                ".*_roll_hip_joint",
                ".*_pitch_hip_joint",
                ".*_yaw_hip_joint",
            ],
            effort_limit=180,
            velocity_limit=5,
            stiffness={
                ".*_waist_joint": 150.0,
                ".*_roll_hip_joint": 150.0,
                ".*_pitch_hip_joint": 150.0,
                ".*_yaw_hip_joint": 150.0,
            },
            damping={
                ".*_waist_joint": 8.0,
                ".*_roll_hip_joint": 8.0,
                ".*_pitch_hip_joint": 8.0,
                ".*_yaw_hip_joint": 8.0,
            },
        ),
        
        "knee_actuators": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee_joint"],
            effort_limit=90,
            velocity_limit=5,
            stiffness={
                ".*_knee_joint": 150.0,
            },
            damping={
                ".*_knee_joint": 10.0,
            },
        ),
        
        "ankle_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*pitch_ankle_joint",
                ".*roll_ankle_joint",
            ],
            effort_limit=45,
            velocity_limit=5,
            stiffness={
                ".*pitch_ankle_joint": 50.0,
                ".*roll_ankle_joint": 50.0,
            },
            damping={
                ".*pitch_ankle_joint": 10.0,
                ".*roll_ankle_joint": 10.0,
            },
        ),
        
        "arm_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_joint",
                ".*_elbow_joint",
                ".*_wrist_joint",
            ],
            effort_limit=50,
            velocity_limit=4,
            stiffness={
                ".*_shoulder_joint": 40.0,
                ".*_elbow_joint": 30.0,
                ".*_wrist_joint": 30.0,
            },
            damping={
                ".*_shoulder_joint": 8.0,
                ".*_elbow_joint": 8.0,
                ".*_wrist_joint": 8.0,
            },
        ),
    }


JACKBOT_MINIMAL_CFG = JACKBOT_CFG.copy()
JACKBOT_MINIMAL_CFG.spawn.usd_path = "/home/thanh-nguyen/Downloads/robot_models/jackbot/v1_newjointlim/jackbot_v1_nobaselink_fixedtorso.usd"
