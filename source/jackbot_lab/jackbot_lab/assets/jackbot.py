"""Configuration for Jackbot robot.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from torch import pi
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
            ".*pitch_ankle_joint": -17.5 / pi,
            ".*pitch_elbow_joint": 60.0 / pi,
            ".*pitch_knee_joint": 30.0 / pi,
            ".*pitch_hip_joint": -12.0 / pi,
            ".*pitch_shoulder_joint": 15.0 / pi,
            ".*roll_ankle_joint": 0.0,
            ".*roll_hip_joint": 0.0,
            ".*roll_shoulder_joint": 0.0,
            ".*yaw_hip_joint": 0.0,
            ".*yaw_knee_joint": 0.0,
            ".*yaw_shoulder_joint": 0.0,
            ".*yaw_wrist_joint": 0.0,
            ".*_waist_joint": 0.0,
            # "R_pitch_ankle_joint": 0.0,
            # "R_pitch_elbow_joint": 0.0,
            # "R_pitch_knee_joint": 0.0,
            # "R_pitch_hip_joint": 0.0,
            # "R_pitch_shoulder_joint": 0.0,
            # "R_roll_ankle_joint": 0.0,
            # "R_roll_hip_joint": 0.0,
            # "R_roll_shoulder_joint": 0.0,
            # "R_yaw_hip_joint": 0.0,
            # "R_yaw_knee_joint": 0.0,
            # "R_yaw_shoulder_joint": 0.0,
            # "R_yaw_wrist_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "actuators": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=2000,
            velocity_limit=100,
            stiffness={
                ".*_shoulder_joint": 100,  # 6
                ".*_elbow_joint": 100,  # 2
                ".*_wrist_joint": 300,  # 2
                ".*_waist_joint": 400,  # 2
                ".*_hip_joint": 400,  # 6
                ".*_knee_joint": 500,  # 4
                ".*pitch_ankle_joint": 400,  # 2
                ".*roll_ankle_joint": 50.0,  # 2
                # "L_pitch_ankle_joint": 20,
                # "L_pitch_elbow_joint": 200,
                # "L_pitch_knee_joint": 200,
                # "L_pitch_hip_joint": 200,
                # "L_pitch_shoulder_joint": 100,
                # "L_roll_ankle_joint": 20,
                # "L_roll_hip_joint": 200,
                # "L_roll_shoulder_joint": 100,
                # "L_yaw_hip_joint": 20,
                # "L_yaw_knee_joint": 20,
                # "L_yaw_shoulder_joint": 100,
                # "L_yaw_wrist_joint": 10,
                # "R_pitch_ankle_joint": 20,
                # "R_pitch_elbow_joint": 200,
                # "R_pitch_knee_joint": 200,
                # "R_pitch_hip_joint": 200,
                # "R_pitch_shoulder_joint": 100,
                # "R_roll_ankle_joint": 20,
                # "R_roll_hip_joint": 200,
                # "R_roll_shoulder_joint": 100,
                # "R_yaw_hip_joint": 20,
                # "R_yaw_knee_joint": 20,
                # "R_yaw_shoulder_joint": 100,
                # "R_yaw_wrist_joint": 10,
                # "pitch_waist_joint": 400,
                # "yaw_waist_joint": 400,
            },
            damping={
                ".*_shoulder_joint": 5,  # 6
                ".*_elbow_joint": 5,  # 2
                ".*_wrist_joint": 5,  # 2
                ".*_waist_joint": 5,  # 2
                ".*_hip_joint": 5,  # 6
                ".*_knee_joint": 5,  # 4
                ".*_ankle_joint": 5,  # 4
                # "L_pitch_ankle_joint": 2,
                # "L_pitch_elbow_joint": 5,
                # "L_pitch_knee_joint": 5,
                # "L_pitch_hip_joint": 5,
                # "L_pitch_shoulder_joint": 1,
                # "L_roll_ankle_joint": 2,
                # "L_roll_hip_joint": 5,
                # "L_roll_shoulder_joint": 1,
                # "L_yaw_hip_joint": 5,
                # "L_yaw_knee_joint": 5,
                # "L_yaw_shoulder_joint": 1,
                # "L_yaw_wrist_joint": 0.1,
                # "R_pitch_ankle_joint": 2,
                # "R_pitch_elbow_joint": 5,
                # "R_pitch_knee_joint": 5,
                # "R_pitch_hip_joint": 5,
                # "R_pitch_shoulder_joint": 1,
                # "R_roll_ankle_joint": 2,
                # "R_roll_hip_joint": 5,
                # "R_roll_shoulder_joint": 1,
                # "R_yaw_hip_joint": 5,
                # "R_yaw_knee_joint": 5,
                # "R_yaw_shoulder_joint": 1,
                # "R_yaw_wrist_joint": 0.1,
                # "pitch_waist_joint": 5,
                # "yaw_waist_joint": 5,
            },
        ),
    },
)


JACKBOT_MINIMAL_CFG = JACKBOT_CFG.copy()
JACKBOT_MINIMAL_CFG.spawn.usd_path = "/home/thanh-nguyen/Downloads/robot_models/jackbot/v1_newjointlim/jackbot_v1_nobaselink_fixedtorso.usd"
