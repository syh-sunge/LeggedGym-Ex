import argparse
import os
import shutil
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legged-gym-root", type=str, default="~/LeggedGym-Ex")
    parser.add_argument("--task", type=str, default="h1")
    parser.add_argument("--load-run", type=str, required=True)
    parser.add_argument("--ckpt", type=int, default=-1)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    root = os.path.abspath(os.path.expanduser(args.legged_gym_root))
    sys.path.insert(0, root)

    from legged_gym import SIMULATOR
    import legged_gym.envs  # noqa: F401
    from legged_gym.utils.task_registry import task_registry
    from legged_gym.utils.helpers import PolicyExporterLSTM

    if SIMULATOR == "genesis":
        import genesis as gs
        gs.init(
            backend=gs.cpu if args.cpu else gs.gpu,
            logging_level="warning",
        )

    runner_args = SimpleNamespace(
        task=args.task,
        headless=True,
        cpu=args.cpu,
        num_envs=1,
        max_iterations=None,
        resume=True,
        sync_wandb=False,
        export_onnx=False,
        debug=False,
        load_run=args.load_run,
        ckpt=args.ckpt,
        use_joystick=False,
        joystick_type="xbox",
        follow_robot=False,
        motion_file=None,
        motion_out_dir=None,
        distill=False,
        teacher_model_path=None,
        num_student=None,
    )

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = 1

    if hasattr(env_cfg, "terrain"):
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 1
        env_cfg.terrain.curriculum = False

    env, env_cfg = task_registry.make_env(
        name=args.task,
        args=runner_args,
        env_cfg=env_cfg,
    )

    runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=runner_args,
        log_root="default",
    )

    actor_critic = runner.alg.actor_critic

    if not getattr(actor_critic, "is_recurrent", False):
        raise RuntimeError(
            "当前 checkpoint 不是 recurrent policy。请确认 h1_config.py 里 "
            "policy_class_name = 'ActorCriticRecurrent'，并且训练时确实使用了 LSTM。"
        )

    output_path = os.path.abspath(os.path.expanduser(args.output))
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    tmp_dir = os.path.join(output_dir, "_tmp_lstm_export")
    os.makedirs(tmp_dir, exist_ok=True)

    exporter = PolicyExporterLSTM(actor_critic)
    exporter.export(tmp_dir)

    src = os.path.join(tmp_dir, "policy_lstm_1.pt")
    shutil.move(src, output_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"Exported LSTM policy to: {output_path}")


if __name__ == "__main__":
    main()
