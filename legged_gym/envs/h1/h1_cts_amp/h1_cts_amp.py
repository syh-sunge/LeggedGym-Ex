from collections import deque

import torch

from legged_gym.envs.h1.h1 import H1Robot
from legged_gym.utils.motion_loader import AMPLoader


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
        self._prepare_reference_reset(env_ids)
        super().reset_idx(env_ids)
        self._clear_reference_reset()
        if len(env_ids) == 0:
            return
        for i in range(self.obs_history_deque.maxlen):
            self.obs_history_deque[i][env_ids] *= 0
        for i in range(self.critic_obs_deque.maxlen):
            self.critic_obs_deque[i][env_ids] *= 0
        self.step_air_time[env_ids] = 0.0
        self.step_last_contacts[env_ids] = False
        self.step_start_pos[env_ids] = self.simulator.feet_pos[env_ids]
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["teacher_terrain_level"] = torch.mean(
                self.simulator.terrain_levels[: self.num_teacher].float()
            )
            self.extras["episode"]["student_terrain_level"] = torch.mean(
                self.simulator.terrain_levels[self.num_teacher :].float()
            )
        if hasattr(self, "contact_termination_buf"):
            self.extras["episode"]["term_contact"] = torch.mean(
                self.contact_termination_buf[env_ids].float()
            )
            self.extras["episode"]["term_orientation"] = torch.mean(
                self.orientation_termination_buf[env_ids].float()
            )
            self.extras["episode"]["term_timeout"] = torch.mean(
                self.time_out_buf[env_ids].float()
            )

    def check_termination(self):
        if len(self.simulator.link_contact_forces.shape) == 4:
            self.terminated_bodies_force_norm = torch.max(
                torch.norm(
                    self.simulator.link_contact_forces[
                        :, :, self.simulator.termination_contact_indices, :
                    ],
                    dim=-1,
                ),
                dim=1,
            )[0]
            self.penalized_bodies_force_norm = torch.max(
                torch.norm(
                    self.simulator.link_contact_forces[
                        :, :, self.simulator.penalized_contact_indices, :
                    ],
                    dim=-1,
                ),
                dim=1,
            )[0]
            self.feet_force_norm = torch.max(
                torch.norm(
                    self.simulator.link_contact_forces[
                        :, :, self.simulator.feet_contact_indices, :
                    ],
                    dim=-1,
                ),
                dim=1,
            )[0]
            self.feet_max_force_z = torch.max(
                self.simulator.link_contact_forces[
                    :, :, self.simulator.feet_contact_indices, 2
                ],
                dim=1,
            )[0]
        else:
            self.terminated_bodies_force_norm = torch.norm(
                self.simulator.link_contact_forces[
                    :, self.simulator.termination_contact_indices, :
                ],
                dim=-1,
            )
            self.penalized_bodies_force_norm = torch.norm(
                self.simulator.link_contact_forces[
                    :, self.simulator.penalized_contact_indices, :
                ],
                dim=-1,
            )
            self.feet_force_norm = torch.norm(
                self.simulator.link_contact_forces[
                    :, self.simulator.feet_contact_indices, :
                ],
                dim=-1,
            )
            self.feet_max_force_z = self.simulator.link_contact_forces[
                :, self.simulator.feet_contact_indices, 2
            ]

        self.contact_termination_buf = torch.any(
            self.terminated_bodies_force_norm > 10.0, dim=1
        )
        self.orientation_termination_buf = (
            self.simulator.projected_gravity[:, 2] > self.cfg.env.max_projected_gravity
        )
        fail_buf = self.contact_termination_buf | self.orientation_termination_buf
        self.fail_buf = torch.where(
            fail_buf,
            self.fail_buf + 1,
            torch.zeros_like(self.fail_buf),
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
            | self.time_out_buf
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
                self._domain_rand_info("dr_friction_values", 1),
                self._domain_rand_info("dr_added_base_mass", 1),
                self._domain_rand_info("dr_rand_push_vels", 3)[:, :2],
                self._domain_rand_info("dr_base_com_bias", 3),
                self._domain_rand_info("dr_kp_scale", self.num_actions),
                self._domain_rand_info("dr_kd_scale", self.num_actions),
                self._domain_rand_info("dr_ctrl_delay", 1),
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

    def _domain_rand_info(self, name, dim):
        try:
            return getattr(self.simulator, name)
        except AttributeError:
            return torch.zeros(self.num_envs, dim, dtype=torch.float, device=self.device)

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
        self.reference_reset_env_ids = None
        self.reference_reset_frames = None
        if self.cfg.init_state.reference_state_initialization:
            self.amp_loader = AMPLoader(
                motion_files=self.cfg.env.amp_motion_files,
                device=self.device,
                time_between_frames=self.dt,
                num_dof=self.num_actions,
                num_key_bodies=len(self.simulator.key_body_indices),
            )
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
        num_feet = len(self.simulator.feet_contact_indices)
        self.step_air_time = torch.zeros(
            self.num_envs, num_feet, dtype=torch.float, device=self.device
        )
        self.step_last_contacts = torch.zeros(
            self.num_envs, num_feet, dtype=torch.bool, device=self.device
        )
        self.step_start_pos = torch.zeros(
            self.num_envs, num_feet, 3, dtype=torch.float, device=self.device
        )

    def _prepare_reference_reset(self, env_ids):
        if (
            len(env_ids) == 0
            or not self.cfg.init_state.reference_state_initialization
            or self.cfg.init_state.reference_state_initialization_prob <= 0.0
        ):
            self._clear_reference_reset()
            return

        ref_mask = (
            torch.rand(len(env_ids), device=self.device)
            < self.cfg.init_state.reference_state_initialization_prob
        )
        ref_env_ids = env_ids[ref_mask]
        if len(ref_env_ids) == 0:
            self._clear_reference_reset()
            return

        self.reference_reset_env_ids = ref_env_ids
        self.reference_reset_frames = self.amp_loader.get_full_frame_batch(len(ref_env_ids))

    def _clear_reference_reset(self):
        self.reference_reset_env_ids = None
        self.reference_reset_frames = None

    def _split_reference_reset_envs(self, env_ids):
        if self.reference_reset_env_ids is None or len(self.reference_reset_env_ids) == 0:
            return env_ids, None

        ref_mask = torch.isin(env_ids, self.reference_reset_env_ids)
        regular_env_ids = env_ids[~ref_mask]
        return regular_env_ids, self.reference_reset_env_ids

    def _reset_dofs(self, env_ids):
        regular_env_ids, ref_env_ids = self._split_reference_reset_envs(env_ids)
        if len(regular_env_ids) > 0:
            super()._reset_dofs(regular_env_ids)
        if ref_env_ids is not None:
            ref_dof_pos = self.amp_loader.get_dof_pos_batch(self.reference_reset_frames)
            ref_dof_vel = self.amp_loader.get_dof_vel_batch(self.reference_reset_frames)
            self.simulator.reset_dofs(ref_env_ids, ref_dof_pos, ref_dof_vel)

    def _reset_root_states(self, env_ids):
        regular_env_ids, ref_env_ids = self._split_reference_reset_envs(env_ids)
        if len(regular_env_ids) > 0:
            super()._reset_root_states(regular_env_ids)
        if ref_env_ids is not None:
            ref_base_pos = self.amp_loader.get_base_pos_batch(self.reference_reset_frames)
            ref_base_pos[:, 2] = self.simulator.base_init_pos[2]
            base_pos = ref_base_pos + self.simulator.env_origins[ref_env_ids]
            ref_base_rot = self.amp_loader.get_base_rot_batch(self.reference_reset_frames)
            ref_base_lin_vel = self.amp_loader.get_base_lin_vel_batch(self.reference_reset_frames)
            ref_base_ang_vel = self.amp_loader.get_base_ang_vel_batch(self.reference_reset_frames)
            self.simulator.reset_root_states(
                ref_env_ids,
                base_pos,
                ref_base_rot,
                ref_base_lin_vel,
                ref_base_ang_vel,
            )

    def _moving_mask(self):
        return torch.norm(self.commands[:, :3], dim=1) > 0.2

    def _reward_step_length(self):
        contact = self.feet_max_force_z > 10.0
        takeoff = (~contact) & self.step_last_contacts
        first_contact = contact & (~self.step_last_contacts) & (self.step_air_time > 0.0)

        self.step_start_pos = torch.where(
            takeoff.unsqueeze(-1),
            self.simulator.feet_pos,
            self.step_start_pos,
        )

        step_length = torch.norm(
            self.simulator.feet_pos[:, :, :2] - self.step_start_pos[:, :, :2],
            dim=-1,
        )
        target = torch.clamp(
            torch.abs(self.commands[:, 0]).unsqueeze(1) * 0.4,
            min=self.cfg.rewards.min_step_length,
            max=self.cfg.rewards.max_step_length,
        )
        step_reward = torch.exp(
            -torch.square(step_length - target) / self.cfg.rewards.step_length_sigma
        )
        reward = torch.sum(step_reward * first_contact, dim=1) * self._moving_mask()

        self.step_air_time = torch.where(
            contact,
            torch.zeros_like(self.step_air_time),
            self.step_air_time + self.dt,
        )
        self.step_last_contacts = contact
        return reward

    def _reward_single_support(self):
        contacts = self.feet_max_force_z > 10.0
        return (torch.sum(contacts.float(), dim=1) == 1).float() * self._moving_mask()
