from legged_gym.envs.h1.h1_config import H1RoughCfg


class H1MotionVisCfg(H1RoughCfg):
    class env(H1RoughCfg.env):
        episode_length_s = 10
        debug_draw_key_body_points = True

    class asset(H1RoughCfg.asset):
        key_bodies = ["left_ankle_link", "right_ankle_link"]
