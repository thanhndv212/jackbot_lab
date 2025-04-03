from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject

from isaaclab.utils.math import (
    quat_rotate_inverse,
    yaw_quat,
    quat_error_magnitude,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def exp_normalize(x: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    """Normalize a tensor using exponential kernel.
    
    Args:
        x: Input tensor
        std: Standard deviation for the exponential kernel
        
    Returns:
        Normalized tensor using exp(-x^2/2std^2)
    """
    return torch.exp(-torch.square(x) / (2 * std**2))


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
        > 0.1
    )
    return exp_normalize(reward, std=threshold)


def feet_air_time_positive_biped(
    env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[
        :, sensor_cfg.body_ids
    ]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
    )[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for small steps
    reward *= reward > 0.2
    # no reward for zero command
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
        > 0.1
    )
    return exp_normalize(reward, std=threshold)


def feet_orientation(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the difference between the orientation of the feet and the root"""
    asset = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :].reshape(-1, 4)
    root_quat = asset.data.root_quat_w
    angle_diff = torch.abs(quat_error_magnitude(body_quat, root_quat))
    return exp_normalize(angle_diff, std=0.5)


def feet_clock_frc(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    stance_mask = env.gait_phase

    # Add transition smoothing
    transition_width = 0.1
    phase = torch.sin(
        2 * torch.pi * env.episode_length_buf / env.cfg.gait_step_num
    )
    transition_mask = torch.abs(phase) < transition_width

    # Modify stance mask to include transition
    stance_mask[transition_mask] = 0.5

    swing_mask = -1 * (1 - stance_mask)
    stance_swing_mask = stance_mask + swing_mask

    asset = env.scene[asset_cfg.name]
    total_mass = torch.sum(asset.data.default_mass[0])
    max_frc = 0.5 * total_mass * 9.81
    
    # Get contact forces
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    force_norms = contact_forces.norm(p=2, dim=-1).max(dim=1)[0]
    
    # Normalize forces using exponential kernel
    force_diff = torch.abs(force_norms - max_frc)
    normed_frc = exp_normalize(force_diff, std=max_frc)
    
    # Apply stance-swing mask
    rew_normed_frc = normed_frc * stance_swing_mask
    return rew_normed_frc.mean(dim=1)


def feet_clock_vel(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    stance_mask = env.gait_phase

    # Add minimum velocity requirement for swing phase
    min_swing_vel = torch.tensor(0.1, device="cuda")
    max_vel = torch.tensor(0.3, device="cuda")

    # Modify swing mask to encourage movement
    swing_mask = -1 * (1 - stance_mask)
    stance_swing_mask = stance_mask + swing_mask
    stance_swing_mask *= -1

    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    vel_norm = body_vel.norm(p=2, dim=-1)

    # Normalize velocities using exponential kernel
    vel_diff = torch.where(
        swing_mask > 0,
        torch.abs(vel_norm - min_swing_vel),  # For swing phase
        torch.abs(vel_norm - max_vel)  # For stance phase
    )
    normed_vel = exp_normalize(vel_diff, std=max_vel)
    
    # Apply stance-swing mask
    rew_normed_vel = normed_vel * stance_swing_mask
    return rew_normed_vel.mean(dim=1)


def leg_coordination_reward(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for proper leg coordination"""
    stance_mask = env.gait_phase  # Shape: (num_envs, 2)

    # Calculate leg height difference
    asset = env.scene[asset_cfg.name]
    foot_positions = asset.data.body_state_w[
        :, asset_cfg.body_ids, 2
    ]  # z-coordinate
    # Shape: (num_envs, 2) - height of each foot

    # Calculate height difference between feet
    leg_height_diff = torch.abs(
        foot_positions[:, 0] - foot_positions[:, 1]
    )  # Shape: (num_envs,)

    # Normalize height difference using exponential kernel
    max_height_diff = torch.tensor(0.3, device="cuda")
    height_reward = exp_normalize(leg_height_diff, std=max_height_diff)

    # Expand height_reward to match swing_mask dimensions
    height_reward = height_reward.unsqueeze(-1).expand(
        -1, 2
    )  # Shape: (num_envs, 2)

    # Apply mask to reward
    swing_mask = 1 - stance_mask  # Shape: (num_envs, 2)

    # Calculate reward for each foot and take mean
    return (height_reward * swing_mask).mean(dim=1)  # Shape: (num_envs,)


def gait_symmetry_reward(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining symmetric gait between left and right legs"""
    # Get gait phase for both legs
    stance_mask = env.gait_phase  # Shape: (num_envs, 2)

    # Calculate difference between left and right leg phases
    # This penalizes when legs are in the same phase (both stance or both swing)
    phase_diff = torch.abs(stance_mask[:, 0] - stance_mask[:, 1])

    # Normalize phase difference using exponential kernel
    return exp_normalize(phase_diff, std=0.5)


def step_length_reward(
    env,
    target_step_length: float,
    min_step_length: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for maintaining appropriate step length between feet"""
    # Get robot asset
    asset = env.scene[asset_cfg.name]

    # Get indices for left and right feet
    left_foot_idx = asset.data.body_names.index("footL")
    right_foot_idx = asset.data.body_names.index("footR")

    # Get x-coordinates (forward direction) of both feet
    left_foot_x = asset.data.body_state_w[:, left_foot_idx, 0]
    right_foot_x = asset.data.body_state_w[:, right_foot_idx, 0]

    # Calculate step length (absolute difference in x-coordinates)
    step_length = torch.abs(left_foot_x - right_foot_x)

    # Calculate reward based on how close the step length is to target
    # Penalize if step length is too small
    reward = torch.where(
        step_length >= min_step_length,
        torch.exp(-torch.abs(step_length - target_step_length) / 0.1),
        torch.zeros_like(step_length),
    )

    return reward


def feet_keep_distance(
    env,
    dist_min: float,
    dist_max: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize if the distance between the two feet about the y-axis is not in the range of 0.22m to 0.32m"""
    asset = env.scene[asset_cfg.name]
    body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    root_pos = asset.data.root_pos_w[:, :]
    root_quat = yaw_quat(asset.data.root_quat_w)
    root_to_body_1 = body_pos[:, 0, :].squeeze(1) - root_pos[:, :]
    root_to_body_2 = body_pos[:, 1, :].squeeze(1) - root_pos[:, :]
    root_to_body_1_b = quat_rotate_inverse(root_quat, root_to_body_1)
    root_to_body_2_b = quat_rotate_inverse(root_quat, root_to_body_2)
    feet_y_distance = torch.abs(
        root_to_body_1_b[:, 1] - root_to_body_2_b[:, 1]
    )
    rew = torch.where(
        (feet_y_distance > dist_min) & (feet_y_distance < dist_max), 0, 1
    )
    return rew


def feet_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    expect_contact_num: int,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
        > 0.1
    )
    return exp_normalize(reward, std=0.5)


def feet_slide_exp(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return exp_normalize(reward, std=0.3)


def track_lin_vel_xy_yaw_frame_exp(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(
        yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(
        torch.square(
            env.command_manager.get_command(command_name)[:, :2]
            - vel_yaw[:, :2]
        ),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2]
        - asset.data.root_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


def feet_height_body_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[
        :, asset_cfg.body_ids, :
    ] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(
        env.num_envs, len(asset_cfg.body_ids), 3, device=env.device
    )
    cur_footvel_translated = asset.data.body_lin_vel_w[
        :, asset_cfg.body_ids, :
    ] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(
        env.num_envs, len(asset_cfg.body_ids), 3, device=env.device
    )
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_rotate_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_rotate_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    height_error = torch.square(
        footpos_in_body_frame[:, :, 2] - target_height
    ).view(env.num_envs, -1)
    foot_leteral_vel = torch.sqrt(
        torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)
    ).view(env.num_envs, -1)
    reward = torch.sum(height_error * foot_leteral_vel, dim=1)
    return torch.exp(-reward / std**2)


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if (
            torch.isnan(ray_hits).any()
            or torch.isinf(ray_hits).any()
            or torch.max(torch.abs(ray_hits)) > 1e6
        ):
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(
                ray_hits, dim=1
            )
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the exponential penalty
    height_diff = asset.data.root_pos_w[:, 2] - adjusted_target_height
    return exp_normalize(height_diff, std=0.1)


def joint_power(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(
            asset.data.joint_vel[:, asset_cfg.joint_ids]
            * asset.data.applied_torque[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    return exp_normalize(reward, std=100.0)



def joint_velocity_reward(
    env,
    target_velocity: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for maintaining target joint velocities"""
    asset = env.scene[asset_cfg.name]
    # Get joint velocities with proper batch dimension
    joint_vel = asset.data.joint_vel[
        :, asset_cfg.joint_ids
    ]  # Shape: (num_envs, num_joints)

    # Calculate reward based on how close the velocity is to target
    vel_diff = torch.abs(
        joint_vel - target_velocity
    )  # Shape: (num_envs, num_joints)
    reward = torch.exp(-vel_diff / 0.5)  # Exponential kernel

    return reward.mean(dim=1)  # Shape: (num_envs,)


def joint_position_reward(
    env,
    target_position: float,
    position_range: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for maintaining target joint positions"""
    asset = env.scene[asset_cfg.name]
    # Get joint positions with proper batch dimension
    joint_pos = asset.data.joint_pos[
        :, asset_cfg.joint_ids
    ]  # Shape: (num_envs, num_joints)

    # Calculate reward based on how close the position is to target
    pos_diff = torch.abs(
        joint_pos - target_position
    )  # Shape: (num_envs, num_joints)
    reward = torch.where(
        pos_diff < position_range,
        torch.exp(
            -pos_diff / position_range
        ),  # Exponential kernel within range
        torch.zeros_like(pos_diff),  # Zero reward outside range
    )

    return reward.mean(dim=1)  # Shape: (num_envs,)


def feet_air_time_balance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.4
) -> torch.Tensor:
    """Reward for balanced air time between left and right feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    left_foot_air = contact_sensor.data.last_air_time[:, 0]  # assuming left foot is first
    right_foot_air = contact_sensor.data.last_air_time[:, 1]  # assuming right foot is second
    
    # Calculate difference in air time
    air_time_diff = torch.abs(left_foot_air - right_foot_air)
    return exp_normalize(air_time_diff, std=0.2)


def feet_body_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_offset: float = 0.3
) -> torch.Tensor:
    """Penalize feet being too far forward or backward relative to body."""
    asset = env.scene[asset_cfg.name]
    body_pos = asset.data.root_pos_w[:, :2]  # x,y position of body
    
    # Get feet positions relative to body
    left_foot_idx = asset.data.body_names.index("footL")
    right_foot_idx = asset.data.body_names.index("footR")
    
    left_foot_pos = asset.data.body_state_w[:, left_foot_idx, :2] - body_pos
    right_foot_pos = asset.data.body_state_w[:, right_foot_idx, :2] - body_pos
    
    # Calculate how far feet are from ideal position (slightly behind body)
    left_offset = torch.abs(left_foot_pos[:, 0] + 0.1)  # ideal: 0.1m behind body
    right_offset = torch.abs(right_foot_pos[:, 0] + 0.1)
    
    # Combine offsets
    total_offset = torch.maximum(left_offset, right_offset)
    return exp_normalize(total_offset, std=max_offset)


def gait_phase_consistency(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    history_length: int = 10
) -> torch.Tensor:
    """Reward for consistent gait phase between steps."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get contact history
    contact_history = contact_sensor.data.contact_history[:, -history_length:]
    left_contacts = contact_history[:, :, 0]  # left foot contacts
    right_contacts = contact_history[:, :, 1]  # right foot contacts
    
    # Calculate phase difference
    phase_diff = torch.abs(
        torch.sum(left_contacts, dim=1) - torch.sum(right_contacts, dim=1)
    )
    return exp_normalize(phase_diff, std=2.0)


def body_velocity_consistency(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    window_size: int = 5
) -> torch.Tensor:
    """Penalize large variations in body velocity."""
    asset = env.scene[asset_cfg.name]
    vel_history = asset.data.root_lin_vel_w_history[:, -window_size:, 0]  # x-velocity
    
    # Calculate velocity variation
    vel_std = torch.std(vel_history, dim=1)
    return exp_normalize(vel_std, std=0.2)
