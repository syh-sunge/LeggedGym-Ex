from legged_gym import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class H1RoughCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.0]
        default_joint_angles = {
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.1,
            "left_knee_joint": 0.3,
            "left_ankle_joint": -0.2,
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.1,
            "right_knee_joint": 0.3,
            "right_ankle_joint": -0.2,
        }

    class env(LeggedRobotCfg.env):
        num_observations = 41
        num_privileged_obs = 44
        num_actions = 10

    class viewer(LeggedRobotCfg.viewer):
        rendered_envs_idx = [0]

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {
            "hip_yaw": 150,
            "hip_roll": 150,
            "hip_pitch": 150,
            "knee": 200,
            "ankle": 40,
        }
        damping = {
            "hip_yaw": 2,
            "hip_roll": 2,
            "hip_pitch": 2,
            "knee": 4,
            "ankle": 2,
        }
        action_scale = 0.18
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        name = "h1"
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1.urdf"
        xml_file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1.xml"
        foot_name = "ankle"
        base_link_name = "pelvis"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0
        flip_visual_attachments = False
        dof_names = [
            "left_hip_yaw_joint",
            "left_hip_roll_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_joint",
            "right_hip_yaw_joint",
            "right_hip_roll_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_joint",
        ]
        dof_armature = [0.01] * 10
        links_to_keep = ["left_ankle_link", "right_ankle_link"]
        dof_vel_limits = [23, 23, 23, 14, 9, 23, 23, 23, 14, 9]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class commands(LeggedRobotCfg.commands):
        zero_cmd_prob = 0.3

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 1.08

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.2
            tracking_ang_vel = 0.4

            lin_vel_z = -2.5
            ang_vel_xy = -0.2
            orientation = -2.5
            base_height = -18.0

            dof_acc = -2.5e-7
            action_rate = -0.05
            dof_pos_limits = -5.0
            collision = -1.0

            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.5

            contact = 0.02
            feet_swing_height = -2.0

            feet_slip = -0.4
            feet_distance = -1.5

            stand_action = -0.5
            stand_joint_pos = -2.0
            stand_contact = 0.8
            large_action = -0.2
            default_joint_pos = -0.05

            torques = 0.0
            feet_air_time = 0.0


class H1RoughCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = "elu"
        rnn_type = "lstm"
        rnn_hidden_size = 64
        rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ""
        experiment_name = "h1"
