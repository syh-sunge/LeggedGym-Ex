from legged_gym import SIMULATOR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.h1.h1_config import H1RoughCfg


class H1DeepMimicCfg(H1RoughCfg):
    class env(H1RoughCfg.env):
        frame_stack = 5
        ref_motion_frame_stack = 2
        ref_motion_single_obs = 36
        num_single_obs = 43 + int(ref_motion_single_obs * ref_motion_frame_stack)
        num_observations = int(num_single_obs * frame_stack)
        c_frame_stack = 5
        num_single_critic_obs = num_single_obs + 17
        num_privileged_obs = int(num_single_critic_obs * c_frame_stack)
        motion_file = f"unitree_h1/{SIMULATOR}_run/C1_-_stand_to_run_stageii_{SIMULATOR}.pkl"
        episode_length_s = 10
        debug_draw_key_body_points = True
        max_projected_gravity = -0.3

    class asset(H1RoughCfg.asset):
        key_bodies = ["left_ankle_link", "right_ankle_link"]

    class init_state(H1RoughCfg.init_state):
        reference_state_initialization = True
        reference_state_initialization_prob = 0.7

    class rewards(H1RoughCfg.rewards):
        soft_dof_pos_limit = 0.9
        tracking_dof_pos_sigma = 4.0
        tracking_dof_vel_sigma = 100.0
        tracking_ref_base_pose_sigma = 0.2
        tracking_ref_base_vel_sigma = 1.0
        tracking_ref_key_pos_sigma = 0.1
        only_positive_rewards = False

        class scales(H1RoughCfg.rewards.scales):
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            alive = 0.0
            contact = 0.0
            stand_action = 0.0
            stand_joint_pos = 0.0
            stand_contact = 0.0
            large_action = 0.0
            default_joint_pos = 0.0

            tracking_ref_dof_pos = 1.0
            tracking_ref_dof_vel = 0.2
            tracking_ref_base_pose = 1.0
            tracking_ref_base_vel = 0.2
            tracking_ref_key_pos = 0.3

            ang_vel_xy = -0.05
            dof_acc = -2.5e-7
            collision = -1.0
            action_rate = -0.01
            feet_slip = -0.5

    class domain_rand(H1RoughCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.0, 2.0]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 0.5
        randomize_com_displacement = True
        com_pos_x_range = [-0.05, 0.05]
        com_pos_y_range = [-0.05, 0.05]
        com_pos_z_range = [-0.05, 0.05]

    class normalization(H1RoughCfg.normalization):
        clip_actions = 100.0


class H1DeepMimicCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        clip_actions = H1DeepMimicCfg.normalization.clip_actions
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [1024, 256, 128]
        activation = "elu"

    class runner(LeggedRobotCfgPPO.runner):
        num_steps_per_env = 32
        max_iterations = 3000
        run_name = f"h1_deepmimic_{SIMULATOR}"
        experiment_name = "h1_deepmimic"
        save_interval = 500
