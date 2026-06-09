import glob

from legged_gym import LEGGED_GYM_ROOT_DIR, SIMULATOR
from legged_gym.envs.h1.h1_config import H1RoughCfg
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO


MOTION_FILES = glob.glob(
    LEGGED_GYM_ROOT_DIR + f"/resources/reference_motion/unitree_h1/{SIMULATOR}_run/*"
)


class H1_CTS_AMPCfg(H1RoughCfg):
    class env(H1RoughCfg.env):
        num_envs = 4096
        num_teacher = num_envs // 4 * 3
        frame_stack = 5
        num_observations = 41
        num_history_obs = num_observations * frame_stack
        c_frame_stack = 5
        num_single_critic_obs = 80
        num_critic_obs = num_single_critic_obs * c_frame_stack
        num_privileged_obs = 37
        num_latent_dims = num_privileged_obs
        num_actions = 10
        amp_motion_files = MOTION_FILES

    class asset(H1RoughCfg.asset):
        key_bodies = ["left_ankle_link", "right_ankle_link"]

    class init_state(H1RoughCfg.init_state):
        reference_state_initialization = False
        reference_state_initialization_prob = 0.0

    class domain_rand(H1RoughCfg.domain_rand):
        randomize_com_displacement = True
        com_pos_x_range = [-0.05, 0.05]
        com_pos_y_range = [-0.05, 0.05]
        com_pos_z_range = [-0.05, 0.05]
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        randomize_ctrl_delay = True
        ctrl_delay_step_range = [0, 2]


class H1_CTS_AMPCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = "CTS_AMP_Runner"

    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [1024, 512, 256]
        privilege_encoder_hidden_dims = [256, 128]
        history_encoder_hidden_dims = [256, 128]
        activation = "elu"
        init_noise_std = 0.8

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        encoder_lr = 4.0e-4
        num_encoder_epochs = 1
        amp_replay_buffer_size = (
            H1_CTS_AMPCfg.env.num_envs * LeggedRobotCfgPPO.runner.num_steps_per_env * 10
        )
        disc_lr = 1.0e-4
        symmetry_cfg = None

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticCTS"
        algorithm_class_name = "PPO_CTS_AMP"
        amp_reward_coef = 1.0 * H1_CTS_AMPCfg.control.dt
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = (
            H1_CTS_AMPCfg.env.num_envs * LeggedRobotCfgPPO.runner.num_steps_per_env * 10
        )
        amp_discr_hidden_dims = [1024, 512]
        amp_task_reward_lerp = 0.5
        num_steps_per_env = 24
        max_iterations = 5000
        save_interval = 500
        run_name = f"h1_cts_amp_{SIMULATOR}"
        experiment_name = "h1_cts_amp"
