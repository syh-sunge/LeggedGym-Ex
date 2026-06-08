from legged_gym import *
import numpy as np
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import torch_rand_float
from .h1_config import H1RoughCfg


class H1Robot(LeggedRobot):
    """H1 humanoid locomotion task using the base LeggedRobot stack."""

    def _reset_dofs(self, env_ids):
        dof_pos = torch.zeros(
            (len(env_ids), self.num_actions),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        dof_vel = torch.zeros(
            (len(env_ids), self.num_actions),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        default = self.simulator.default_dof_pos
        dof_pos[:, :] = default + torch_rand_float(
            -0.2, 0.2, (len(env_ids), self.num_actions), self.device
        )
        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)

    def _reset_root_states(self, env_ids):
        if self.simulator.custom_origins:
            base_pos = self.simulator.base_init_pos.reshape(1, -1).repeat(len(env_ids), 1)
            base_pos += self.simulator.env_origins[env_ids]
            base_pos[:, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
        else:
            base_pos = self.simulator.base_init_pos.reshape(1, -1).repeat(len(env_ids), 1)
            base_pos += self.simulator.env_origins[env_ids]
        base_quat = self.simulator.base_init_quat.reshape(1, -1).repeat(len(env_ids), 1)
        base_lin_vel = torch_rand_float(-0.5, 0.5, (len(env_ids), 3), self.device)
        base_ang_vel = torch_rand_float(-0.5, 0.5, (len(env_ids), 3), self.device)
        self.simulator.reset_root_states(env_ids, base_pos, base_quat, base_lin_vel, base_ang_vel)

    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.0
        noise_vec[9:9 + self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9 + self.num_actions:9 + 2 * self.num_actions] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[9 + 2 * self.num_actions:9 + 3 * self.num_actions] = 0.0
        noise_vec[9 + 3 * self.num_actions:9 + 3 * self.num_actions + 2] = 0.0
        return noise_vec

    def _post_physics_step_callback(self):
        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        return super()._post_physics_step_callback()

    def compute_observations(self):
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)
        moving_mask = self._get_moving_mask().unsqueeze(1)
        sin_phase = sin_phase * moving_mask
        cos_phase = cos_phase * moving_mask

        self.obs_buf = torch.cat((
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            self.simulator.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.simulator.dof_pos - self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
            self.simulator.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            sin_phase,
            cos_phase
        ), dim=-1)
        self.privileged_obs_buf = torch.cat((
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            self.simulator.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.simulator.dof_pos - self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
            self.simulator.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            sin_phase,
            cos_phase
        ), dim=-1)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _get_moving_mask(self):
        cmd_xy = torch.norm(self.commands[:, :2], dim=1)
        cmd_yaw = torch.abs(self.commands[:, 2])
        moving = (cmd_xy > 0.15) | (cmd_yaw > 0.15)
        return moving.float()

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1),
            self.device,
        ).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        self.commands[env_ids, :3] *= (
            torch.norm(self.commands[env_ids, :3], dim=1) > 0.2
        ).unsqueeze(1)
        stand_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.commands.zero_cmd_prob
        self.commands[env_ids[stand_mask], :3] = 0.0

    def _reward_contact(self):
        moving = self._get_moving_mask()
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        feet_idx = self.simulator.feet_contact_indices
        for i in range(len(feet_idx)):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.feet_max_force_z[:, i] > 1.0
            res += (~(contact ^ is_stance)).float()
        return res * moving

    def _reward_feet_swing_height(self):
        moving = self._get_moving_mask()
        feet_pos = self.simulator.feet_pos
        contact = self.feet_force_norm > 1.0
        target_height = 0.04
        pos_error = torch.square(feet_pos[:, :, 2] - target_height) * ~contact
        return torch.sum(pos_error, dim=1) * moving

    def _reward_alive(self):
        return 1.0

    def _reward_contact_no_vel(self):
        feet_vel = self.simulator.feet_vel
        contact = self.feet_force_norm > 1.0
        contact_feet_vel = feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))

    def _reward_feet_distance(self):
        feet_pos = self.simulator.feet_pos
        feet_y_dist = torch.abs(feet_pos[:, 0, 1] - feet_pos[:, 1, 1])

        min_dist = 0.18
        max_dist = 0.45
        too_close = torch.clamp(min_dist - feet_y_dist, min=0.0)
        too_far = torch.clamp(feet_y_dist - max_dist, min=0.0)
        return torch.square(too_close) + torch.square(too_far)

    def _reward_feet_slip(self):
        contact = self.feet_force_norm > 1.0
        feet_xy_vel = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)
        return torch.sum(feet_xy_vel * contact, dim=1)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.simulator.dof_pos[:, [0, 1, 5, 6]]), dim=1)

    def _reward_default_joint_pos(self):
        joint_diff = self.simulator.dof_pos - self.simulator.default_dof_pos
        hip_yaw_roll = joint_diff[:, [0, 1, 5, 6]]
        knee_ankle = joint_diff[:, [3, 4, 8, 9]]
        return (
            torch.sum(torch.square(hip_yaw_roll), dim=1)
            + 0.25 * torch.sum(torch.square(knee_ankle), dim=1)
        )

    def _reward_stand_still(self):
        moving = self._get_moving_mask()
        stand = 1.0 - moving

        joint_error = torch.sum(
            torch.square(self.simulator.dof_pos - self.simulator.default_dof_pos),
            dim=1,
        )
        base_xy_vel = torch.sum(torch.square(self.simulator.base_lin_vel[:, :2]), dim=1)
        base_ang_vel = torch.sum(torch.square(self.simulator.base_ang_vel[:, :2]), dim=1)
        action_mag = torch.sum(torch.square(self.actions), dim=1)

        return stand * (
            joint_error
            + 2.0 * base_xy_vel
            + 0.5 * base_ang_vel
            + 0.03 * action_mag
        )

    def _reward_stand_contact(self):
        moving = self._get_moving_mask()
        stand = 1.0 - moving
        contact = (self.feet_force_norm > 1.0).float()
        return stand * torch.sum(contact, dim=1)

    def _reward_stand_action(self):
        moving = self._get_moving_mask()
        stand = 1.0 - moving
        return stand * torch.sum(torch.square(self.actions), dim=1)

    def _reward_stand_joint_pos(self):
        moving = self._get_moving_mask()
        stand = 1.0 - moving
        joint_error = self.simulator.dof_pos - self.simulator.default_dof_pos
        return stand * torch.sum(torch.square(joint_error), dim=1)

    def _reward_large_action(self):
        excess = torch.clamp(torch.abs(self.actions) - 1.0, min=0.0)
        return torch.sum(torch.square(excess), dim=1)

    def _reward_knee_action(self):
        knee_action = self.actions[:, [3, 8]]
        return torch.sum(torch.square(knee_action), dim=1)
