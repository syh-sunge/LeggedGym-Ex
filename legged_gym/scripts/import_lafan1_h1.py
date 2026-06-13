from __future__ import annotations

import argparse
import math
import os
import pickle
from pathlib import Path

import numpy as np

from legged_gym import LEGGED_GYM_ROOT_DIR


H1_DOF_NAMES = [
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

H1_DOF_LIMITS = np.array(
    [
        [-0.43, 0.43],
        [-0.43, 0.43],
        [-1.57, 1.57],
        [-0.26, 2.05],
        [-0.87, 0.52],
        [-0.43, 0.43],
        [-0.43, 0.43],
        [-1.57, 1.57],
        [-0.26, 2.05],
        [-0.87, 0.52],
    ],
    dtype=np.float32,
)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., :3] *= -1.0
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    x, y, z, w = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    mat = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
    mat[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    mat[..., 0, 1] = 2.0 * (xy - wz)
    mat[..., 0, 2] = 2.0 * (xz + wy)
    mat[..., 1, 0] = 2.0 * (xy + wz)
    mat[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    mat[..., 1, 2] = 2.0 * (yz - wx)
    mat[..., 2, 0] = 2.0 * (xz - wy)
    mat[..., 2, 1] = 2.0 * (yz + wx)
    mat[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return mat


def axis_angle_matrix(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    axis = axis.astype(np.float32)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c
    mat = np.empty(angle.shape + (3, 3), dtype=np.float32)
    mat[..., 0, 0] = c + x * x * one_c
    mat[..., 0, 1] = x * y * one_c - z * s
    mat[..., 0, 2] = x * z * one_c + y * s
    mat[..., 1, 0] = y * x * one_c + z * s
    mat[..., 1, 1] = c + y * y * one_c
    mat[..., 1, 2] = y * z * one_c - x * s
    mat[..., 2, 0] = z * x * one_c - y * s
    mat[..., 2, 1] = z * y * one_c + x * s
    mat[..., 2, 2] = c + z * z * one_c
    return mat


def fk_leg(dof_pos: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        start = 0
        translations = [
            [0.0, 0.0875, -0.1742],
            [0.039468, 0.0, 0.0],
            [0.0, 0.11536, 0.0],
            [0.0, 0.0, -0.4],
            [0.0, 0.0, -0.4],
        ]
    elif side == "right":
        start = 5
        translations = [
            [0.0, -0.0875, -0.1742],
            [0.039468, 0.0, 0.0],
            [0.0, -0.11536, 0.0],
            [0.0, 0.0, -0.4],
            [0.0, 0.0, -0.4],
        ]
    else:
        raise ValueError(side)

    axes = [
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    ]
    pos = np.zeros((dof_pos.shape[0], 3), dtype=np.float32)
    rot = np.broadcast_to(np.eye(3, dtype=np.float32), (dof_pos.shape[0], 3, 3)).copy()
    for i, trans in enumerate(translations):
        pos += np.einsum("nij,j->ni", rot, np.asarray(trans, dtype=np.float32))
        joint_rot = axis_angle_matrix(axes[i], dof_pos[:, start + i])
        rot = np.einsum("nij,njk->nik", rot, joint_rot)
    return pos


def angular_velocity_from_quat(root_rot: np.ndarray, fps: float) -> np.ndarray:
    root_rot = quat_normalize(root_rot.astype(np.float32))
    out = np.zeros((root_rot.shape[0], 3), dtype=np.float32)
    if root_rot.shape[0] < 2:
        return out
    delta = quat_mul(root_rot[1:], quat_conj(root_rot[:-1]))
    delta = quat_normalize(delta)
    sign = np.where(delta[:, 3:4] < 0.0, -1.0, 1.0)
    delta *= sign
    xyz = delta[:, :3]
    w = np.clip(delta[:, 3], -1.0, 1.0)
    sin_half = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, w)
    axis = xyz / np.clip(sin_half[:, None], 1e-8, None)
    out[:-1] = axis * angle[:, None] * fps
    out[-1] = out[-2]
    return out


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    if values.shape[0] < 2:
        return out
    out[:-1] = (values[1:] - values[:-1]) * fps
    out[-1] = out[-2]
    return out


def convert_file(csv_path: Path, raw_dir: Path, processed_dirs: dict[str, Path], clip: bool) -> tuple[int, int]:
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != 26:
        raise ValueError(f"{csv_path} has {data.shape[1]} columns, expected 26")

    fps = 30.0
    root_pos = data[:, 0:3].astype(np.float32)
    root_rot = quat_normalize(data[:, 3:7].astype(np.float32))
    dof_pos_raw = data[:, 7:17].astype(np.float32)
    if clip:
        dof_pos = np.clip(dof_pos_raw, H1_DOF_LIMITS[:, 0], H1_DOF_LIMITS[:, 1])
        clipped = int(np.count_nonzero(np.abs(dof_pos - dof_pos_raw) > 1e-6))
    else:
        dof_pos = dof_pos_raw
        clipped = 0

    root_lin_vel = finite_difference(root_pos, fps)
    root_ang_vel = angular_velocity_from_quat(root_rot, fps)
    dof_vel = finite_difference(dof_pos, fps)

    local_key_body_pos = np.stack([fk_leg(dof_pos, "left"), fk_leg(dof_pos, "right")], axis=1)
    root_matrix = quat_to_matrix(root_rot)
    key_body_pos_relative_to_base = np.einsum("nij,nkj->nki", root_matrix, local_key_body_pos)

    raw_motion = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "source": str(csv_path),
        "dof_names": H1_DOF_NAMES,
    }
    processed_motion = {
        "fps": fps,
        "root_pos": root_pos,
        "root_lin_vel": root_lin_vel,
        "root_rot": root_rot,
        "root_ang_vel": root_ang_vel,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "key_body_pos_relative_to_base": key_body_pos_relative_to_base.astype(np.float32),
        "source": str(csv_path),
        "dof_names": H1_DOF_NAMES,
        "key_body_names": ["left_ankle_link", "right_ankle_link"],
    }

    stem = f"lafan1_{csv_path.stem}"
    with (raw_dir / f"{stem}.pkl").open("wb") as f:
        pickle.dump(raw_motion, f)
    for simulator, out_dir in processed_dirs.items():
        with (out_dir / f"{stem}_{simulator}.pkl").open("wb") as f:
            pickle.dump(processed_motion, f)
    return int(data.shape[0]), clipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path.home() / "datasets/LAFAN1_Retargeting_Dataset/h1",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(LEGGED_GYM_ROOT_DIR) / "resources/reference_motion/unitree_h1",
    )
    parser.add_argument("--no-clip", action="store_true", help="keep joint values outside H1 URDF limits")
    args = parser.parse_args()

    csv_files = sorted(args.dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {args.dataset_dir}")

    raw_dir = args.out_root / "raw_run"
    processed_dirs = {
        "genesis": args.out_root / "genesis_run",
        "isaacgym": args.out_root / "isaacgym_run",
        "isaaclab": args.out_root / "isaaclab_run",
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    for out_dir in processed_dirs.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    total_clipped = 0
    for csv_path in csv_files:
        frames, clipped = convert_file(csv_path, raw_dir, processed_dirs, clip=not args.no_clip)
        total_frames += frames
        total_clipped += clipped
        print(f"converted {csv_path.name}: {frames} frames, clipped_values={clipped}")

    print(
        f"converted {len(csv_files)} files, {total_frames} frames, "
        f"total_clipped_values={total_clipped}, out_root={args.out_root}"
    )


if __name__ == "__main__":
    main()
