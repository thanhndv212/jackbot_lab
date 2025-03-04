"""Configuration for Jackbot robot.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration - Actuators.
##

JACKBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/thanh-nguyen/Downloads/robot_models/jackbot/jackbot_v0_1.usd",
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
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.93),
        joint_pos={
            # left leg
            "pelvisL_joint": 0.0,
            "femurLroll_joint": 0.0,
            "femurLyaw_joint": -0.2618,
            "kneeLpitch_joint": 0.5236,
            "ankleLroll_joint": -0.2618,
            "ankleLpitch_joint": 0.0,
            # right leg
            "pelvisR_joint": 0.0,
            "femurRroll_joint": 0.0,
            "femurRyaw_joint": -0.2618,
            "kneeRpitch_joint": 0.5236,
            "ankleRroll_joint": -0.2618,
            "ankleRpitch_joint": 0.0,
            # waist
            "torsoRoll_joint": 0.0,
            "torsoYaw_joint": 0.0,
            # left arm
            "shoulderLroll_joint": 0.0,
            "shoulderLpitch_joint": 0.0,
            "shoulderLyaw_joint": 0.0,
            "elbowL_joint": 0.0,
            "wristL_joint": 0.0,
            # right arm
            "shoulderRroll_joint": 0.0,
            "shoulderRpitch_joint": 0.0,
            "shoulderRyaw_joint": 0.0,
            "elbowR_joint": 0.0,
            "wristR_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "actuators": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                 # left leg
                # "pelvisL_joint": 0.0,
                # "femurLroll_joint": 0.0,
                # "femurLyaw_joint": -0.2618,
                # "kneeLpitch_joint": 0.5236,
                # "ankleLroll_joint": -0.2618,
                # "ankleLpitch_joint": 0.0,
                # # right leg
                # "pelvisR_joint": 0.0,
                # "femurRroll_joint": 0.0,
                # "femurRyaw_joint": -0.2618,
                # "kneeRpitch_joint": 0.5236,
                # "ankleRroll_joint": -0.2618,
                # "ankleRpitch_joint": 0.0,
                # # waist
                # "torsoRoll_joint": 0.0,
                # "torsoYaw_joint": 0.0,
                # # left arm
                # "shoulderLroll_joint": 0.0,
                # "shoulderLpitch_joint": 0.0,
                # "shoulderLyaw_joint": 0.0,
                # "elbowL_joint": 0.0,
                # "wristL_joint": 0.0,
                # # right arm
                # "shoulderRroll_joint": 0.0,
                # "shoulderRpitch_joint": 0.0,
                # "shoulderRyaw_joint": 0.0,
                # "elbowR_joint": 0.0,
                # "wristR_joint": 0.0,
            },
            damping={
                # # left leg
                # "pelvisL_joint": 0.0,
                # "femurLroll_joint": 0.0,
                # "femurLyaw_joint": -0.2618,
                # "kneeLpitch_joint": 0.5236,
                # "ankleLroll_joint": -0.2618,
                # "ankleLpitch_joint": 0.0,
                # # right leg
                # "pelvisR_joint": 0.0,
                # "femurRroll_joint": 0.0,
                # "femurRyaw_joint": -0.2618,
                # "kneeRpitch_joint": 0.5236,
                # "ankleRroll_joint": -0.2618,
                # "ankleRpitch_joint": 0.0,
                # # waist
                # "torsoRoll_joint": 0.0,
                # "torsoYaw_joint": 0.0,
                # # left arm
                # "shoulderLroll_joint": 0.0,
                # "shoulderLpitch_joint": 0.0,
                # "shoulderLyaw_joint": 0.0,
                # "elbowL_joint": 0.0,
                # "wristL_joint": 0.0,
                # # right arm
                # "shoulderRroll_joint": 0.0,
                # "shoulderRpitch_joint": 0.0,
                # "shoulderRyaw_joint": 0.0,
                # "elbowR_joint": 0.0,
                # "wristR_joint": 0.0,
            },
        ),
    },
)



JACKBOT_MINIMAL_CFG = JACKBOT_CFG.copy()
JACKBOT_MINIMAL_CFG.spawn.usd_path = "exts/torobo_isaac_lab/data/usd/leg_v1.usd"
