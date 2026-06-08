from collections import deque

from legged_gym import *
import numpy as np
import torch

from legged_gym.envs.h1.h1 import H1Robot
from legged_gym.utils.math_utils import quat_rotate_inverse, torch_rand_float
from legged_gym.utils.motion_loader import MotionLoader


class H1DeepMimic(H1Robot):
    def compute_observations(self):
        key_body_pos_relative_to_base = (
            self.simulator.key_body_pos - self.simulator.base_pos.unsqueeze(1)
        )
        key_body_pos_b = quat_rotate_inverse(
            self.simulator.base_quat.unsqueeze(1).repeat(
                1, self.simulator.key_body_pos.shape[1], 1
            ),
            key_body_pos_relative_to_base,
        )
        ref_motion_obs = self._get_ref_motion_obs()

        obs_buf = torch.cat(
            (
                self.simulator.base_ang_vel * self.obs_scales.ang_vel,
                self.simulator.base_quat,
                (self.simulator.dof_pos - self.simulator.default_dof_pos)
                * self.obs_scales.dof_pos,
                self.simulator.dof_vel * self.obs_scales.dof_vel,
                key_body_pos_b.flatten(start_dim=1),
                self.actions,
                ref_motion_obs,
            ),
            dim=-1,
        )

        domain_params = torch.cat(
            (
                self.simulator.dr_friction_values,
                self.simulator.dr_added_base_mass,
                self.simulator.dr_base_com_bias,
                self.simulator.dr_rand_push_vels,
            ),
            dim=-1,
        )
        single_critic_obs = torch.cat(
            (
                self.simulator.base_pos - self.simulator.env_origins,
                self.simulator.base_lin_vel * self.obs_scales.lin_vel,
                self.motion_loader.get_ref_base_pos(),
                obs_buf,
                domain_params,
            ),
            dim=-1,
        )

        self.critic_obs_deque.append(single_critic_obs)
        self.privileged_obs_buf = torch.cat(
            [self.critic_obs_deque[i] for i in range(self.critic_obs_deque.maxlen)],
            dim=-1,
        )

        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        self.obs_history_deque.append(obs_buf)
        self.obs_buf = torch.cat(
            [self.obs_history_deque[i] for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )

    def post_physics_step(self):
        ref_time_out_env_ids = self.motion_loader.step_frame_index()
        if len(ref_time_out_env_ids) > 0:
            _ = np.random.random()
            if (
                self.cfg.init_state.reference_state_initialization
                and _ < self.cfg.init_state.reference_state_initialization_prob
            ):
                self._reset_dofs_from_reference_motion(ref_time_out_env_ids)
                self._reset_root_states_from_reference_motion(ref_time_out_env_ids)
            else:
                self._reset_root_states(ref_time_out_env_ids)
                self._reset_dofs(ref_time_out_env_ids)
            if hasattr(self.simulator, "forward"):
                self.simulator.forward()

        super().post_physics_step()

        if self.debug:
            ref_key_body_pos = (
                self.motion_loader.get_ref_key_body_pos()
                + self.motion_loader.get_ref_base_pos().unsqueeze(1)
                + self.simulator.env_origins.unsqueeze(1)
            )
            self.simulator.draw_debug_vis(ref_key_body_pos)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        if (
            self.cfg.commands.curriculum
            and self.common_step_counter % self.max_episode_length == 0
        ):
            self._update_command_curriculum(env_ids)

        self.motion_loader.resample_frame_index(env_ids)
        self._resample_commands(env_ids)

        _ = np.random.random()
        if (
            self.cfg.init_state.reference_state_initialization
            and _ < self.cfg.init_state.reference_state_initialization_prob
        ):
            self._reset_dofs_from_reference_motion(env_ids)
            self._reset_root_states_from_reference_motion(env_ids)
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)
        self.simulator.reset_idx(env_ids)

        self.llast_actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.fail_buf[env_ids] = 0

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.simulator.terrain_levels.float()
            )
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        for i in range(self.obs_history_deque.maxlen):
            self.obs_history_deque[i][env_ids] *= 0
        for i in range(self.critic_obs_deque.maxlen):
            self.critic_obs_deque[i][env_ids] *= 0

    def _reset_dofs_from_reference_motion(self, env_ids):
        self.simulator.reset_dofs(
            env_ids,
            self.motion_loader.get_ref_dof_pos(env_ids),
            self.motion_loader.get_ref_dof_vel(env_ids),
        )

    def _reset_root_states_from_reference_motion(self, env_ids):
        root_pos = (
            self.motion_loader.get_ref_base_pos(env_ids)
            + self.simulator.env_origins[env_ids]
        )
        root_pos[:, 2] = self.simulator.base_init_pos[2]
        self.simulator.reset_root_states(
            env_ids,
            root_pos,
            self.motion_loader.get_ref_base_quat(env_ids),
            self.motion_loader.get_ref_base_lin_vel(env_ids),
            self.motion_loader.get_ref_base_ang_vel(env_ids),
        )

    def _get_noise_scale_vec(self):
        self.add_noise = self.cfg.noise.add_noise
        return torch.zeros(self.cfg.env.num_single_obs, device=self.device)

    def _init_buffers(self):
        super()._init_buffers()
        self.motion_loader = MotionLoader(
            self.num_envs,
            len(self.simulator.key_body_indices),
            self.cfg.env.motion_file,
            self.device,
        )
        self.obs_history_deque = deque(maxlen=self.cfg.env.frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history_deque.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )
        self.critic_obs_deque = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_obs_deque.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_critic_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )

    def _get_ref_motion_obs(self):
        ref_motion_obs = []
        current_frame_idx = self.motion_loader.frame_index
        for i in range(self.cfg.env.ref_motion_frame_stack):
            frame_idx = current_frame_idx + i
            over_length_env_ids = (
                frame_idx >= self.motion_loader.trajectory_num_frames
            ).nonzero(as_tuple=False).flatten()
            frame_idx[over_length_env_ids] = (
                frame_idx[over_length_env_ids] % self.motion_loader.trajectory_num_frames
            )
            ref_motion_obs.append(self._get_single_frame_ref_motion_obs(frame_idx))
        return torch.cat(ref_motion_obs, dim=-1)

    def _get_single_frame_ref_motion_obs(self, frame_idx):
        ref_base_quat = self.motion_loader.get_ref_base_quat_at_idx(frame_idx)
        ref_key_body_pos = self.motion_loader.get_ref_key_body_pos_at_idx(frame_idx)
        key_body_pos_b = quat_rotate_inverse(
            ref_base_quat.unsqueeze(1).repeat(1, ref_key_body_pos.shape[1], 1),
            ref_key_body_pos,
        )
        ref_base_lin_vel_b = quat_rotate_inverse(
            ref_base_quat,
            self.motion_loader.get_ref_base_lin_vel_at_idx(frame_idx),
        )
        ref_base_ang_vel_b = quat_rotate_inverse(
            ref_base_quat,
            self.motion_loader.get_ref_base_ang_vel_at_idx(frame_idx),
        )
        return torch.cat(
            (
                ref_base_lin_vel_b * self.obs_scales.lin_vel,
                ref_base_ang_vel_b * self.obs_scales.ang_vel,
                ref_base_quat,
                (self.motion_loader.get_ref_dof_pos_at_idx(frame_idx)
                 - self.simulator.default_dof_pos)
                * self.obs_scales.dof_pos,
                self.motion_loader.get_ref_dof_vel_at_idx(frame_idx)
                * self.obs_scales.dof_vel,
                key_body_pos_b.flatten(start_dim=1),
            ),
            dim=-1,
        )

    def _reward_tracking_ref_dof_pos(self):
        error = torch.sum(
            torch.square(self.simulator.dof_pos - self.motion_loader.get_ref_dof_pos()),
            dim=-1,
        )
        return torch.exp(-error / self.cfg.rewards.tracking_dof_pos_sigma)

    def _reward_tracking_ref_dof_vel(self):
        error = torch.sum(
            torch.abs(self.simulator.dof_vel - self.motion_loader.get_ref_dof_vel()),
            dim=-1,
        )
        return torch.exp(-error / self.cfg.rewards.tracking_dof_vel_sigma)

    def _reward_tracking_ref_base_pose(self):
        base_pos_error = torch.sum(
            torch.square(
                self.simulator.base_pos
                - (self.motion_loader.get_ref_base_pos() + self.simulator.env_origins)
            ),
            dim=-1,
        )
        base_rot_error = torch.sum(
            torch.square(
                self.simulator.base_quat - self.motion_loader.get_ref_base_quat()
            ),
            dim=-1,
        )
        return torch.exp(
            -(base_pos_error + 0.1 * base_rot_error)
            / self.cfg.rewards.tracking_ref_base_pose_sigma
        )

    def _reward_tracking_ref_base_vel(self):
        ref_base_quat = self.motion_loader.get_ref_base_quat()
        ref_base_lin_vel_b = quat_rotate_inverse(
            ref_base_quat,
            self.motion_loader.get_ref_base_lin_vel(),
        )
        ref_base_ang_vel_b = quat_rotate_inverse(
            ref_base_quat,
            self.motion_loader.get_ref_base_ang_vel(),
        )
        base_lin_vel_b = quat_rotate_inverse(
            self.simulator.base_quat,
            self.simulator.base_lin_vel,
        )
        base_ang_vel_b = quat_rotate_inverse(
            self.simulator.base_quat,
            self.simulator.base_ang_vel,
        )
        base_lin_vel_error = torch.sum(torch.square(base_lin_vel_b - ref_base_lin_vel_b), dim=-1)
        base_ang_vel_error = torch.sum(torch.square(base_ang_vel_b - ref_base_ang_vel_b), dim=-1)
        return torch.exp(
            -(base_lin_vel_error + 0.1 * base_ang_vel_error)
            / self.cfg.rewards.tracking_ref_base_vel_sigma
        )

    def _reward_tracking_ref_key_pos(self):
        key_body_pos_relative_to_base = (
            self.simulator.key_body_pos - self.simulator.base_pos.unsqueeze(1)
        )
        error = torch.sum(
            torch.square(
                key_body_pos_relative_to_base
                - self.motion_loader.get_ref_key_body_pos()
            ),
            dim=[1, 2],
        )
        return torch.exp(-error / self.cfg.rewards.tracking_ref_key_pos_sigma)
