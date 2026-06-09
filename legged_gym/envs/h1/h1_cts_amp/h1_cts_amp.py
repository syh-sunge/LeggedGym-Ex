from collections import deque

import torch

from legged_gym.envs.h1.h1 import H1Robot


class H1_CTS_AMP(H1Robot):
    """H1 concurrent teacher-student task with AMP reward."""

    def step(self, actions):
        actions = self._pre_sim_step(actions)
        self.simulator.step(actions)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.obs_history,
            self.critic_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.reset_env_ids,
            self.terminal_amp_states,
        )

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, obs_history, critic_obs, _, _, _, _, _ = self.step(
            torch.zeros(
                self.num_envs,
                self.num_actions,
                device=self.device,
                requires_grad=False,
            )
        )
        return obs, privileged_obs, obs_history, critic_obs

    def get_observations(self):
        return self.obs_buf, self.privileged_obs_buf, self.obs_history, self.critic_obs_buf

    def reset_idx(self, env_ids):
        self.reset_env_ids = env_ids
        self.terminal_amp_states = self.get_amp_observations()[env_ids]
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        for i in range(self.obs_history_deque.maxlen):
            self.obs_history_deque[i][env_ids] *= 0
        for i in range(self.critic_obs_deque.maxlen):
            self.critic_obs_deque[i][env_ids] *= 0
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["teacher_terrain_level"] = torch.mean(
                self.simulator.terrain_levels[: self.num_teacher].float()
            )
            self.extras["episode"]["student_terrain_level"] = torch.mean(
                self.simulator.terrain_levels[self.num_teacher :].float()
            )

    def get_amp_observations(self):
        key_body_pos_relative_to_base = (
            self.simulator.key_body_pos - self.simulator.base_pos.unsqueeze(1)
        )
        return torch.cat(
            (
                self.simulator.base_lin_vel,
                self.simulator.base_ang_vel,
                self.simulator.dof_pos,
                self.simulator.dof_vel,
                key_body_pos_relative_to_base.flatten(start_dim=1),
            ),
            dim=-1,
        )

    def compute_observations(self):
        super().compute_observations()

        key_body_pos_relative_to_base = (
            self.simulator.key_body_pos - self.simulator.base_pos.unsqueeze(1)
        )
        domain_randomization_info = torch.cat(
            (
                self.simulator.dr_friction_values,
                self.simulator.dr_added_base_mass,
                self.simulator.dr_rand_push_vels[:, :2],
                self.simulator.dr_base_com_bias,
                self.simulator.dr_kp_scale,
                self.simulator.dr_kd_scale,
                self.simulator.dr_ctrl_delay,
            ),
            dim=-1,
        )

        single_critic_obs = torch.cat(
            (
                self.obs_buf,
                self.simulator.base_lin_vel * self.obs_scales.lin_vel,
                key_body_pos_relative_to_base.flatten(start_dim=1),
                domain_randomization_info,
                self.simulator.feet_pos[:, :, 2],
            ),
            dim=-1,
        )
        self.critic_obs_deque.append(single_critic_obs)
        self.critic_obs_buf = torch.cat(
            [self.critic_obs_deque[i] for i in range(self.critic_obs_deque.maxlen)],
            dim=-1,
        )

        self.obs_history_deque.append(self.obs_buf)
        self.obs_history = torch.cat(
            [self.obs_history_deque[i] for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )

        self.privileged_obs_buf = torch.cat(
            (
                self.simulator.base_lin_vel * self.obs_scales.lin_vel,
                key_body_pos_relative_to_base.flatten(start_dim=1),
                domain_randomization_info,
            ),
            dim=-1,
        )

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        if self.cfg.env.num_envs > 1:
            self.num_teacher = min(self.cfg.env.num_teacher, self.cfg.env.num_envs - 1)
        else:
            self.num_teacher = 0
        self.num_history_obs = self.cfg.env.num_history_obs
        self.num_latent_dims = self.cfg.env.num_latent_dims
        self.num_critic_obs = self.cfg.env.num_critic_obs

    def _init_buffers(self):
        super()._init_buffers()
        self.reset_env_ids = None
        self.terminal_amp_states = None
        self.obs_history_deque = deque(maxlen=self.cfg.env.frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history_deque.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_observations,
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
