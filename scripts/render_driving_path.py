#!/usr/bin/env python
"""
Field2Scene P2 — 주행 시점 합성 데이터 생성기.

학습된 3DGS 씬(토마토 온실)에서 로봇 주행 시점의 카메라 경로를 만들어
novel-view 프레임을 렌더한다. 학습 카메라(온실 통로를 따라간 dolly 샷)의
궤적을 부드럽게 재보간(위치 lerp + 회전 slerp)해 "주행 경로"의 척추로 삼고,
도메인 랜덤화(시점 흔들기 + 광학 변형)를 얹어 Sim-to-Real 비교(P3)용
합성 데이터셋을 만든다.

INRIA gaussian-splatting 렌더러를 재사용한다(별도 학습/재구성 없음).
결과물(렌더·데이터)은 data/·output/ 아래로 나가며 리포에 커밋하지 않는다.

사용:
  python render_driving_path.py -m <model_dir> -o <out_dir> \
      [--iteration 30000] [--width 1280] [--nominal 150] [--dr 200] [--seed 0]
"""
import os
import sys
import json
import math
import argparse
from types import SimpleNamespace

import numpy as np
import torch
import torchvision

# --- INRIA gaussian-splatting 리포를 import 경로에 추가 ---
GS_REPO = os.environ.get(
    "GS_REPO",
    r"c:/coding/my-github-repository/HonglabSplatting/gaussian-splatting",
)
sys.path.insert(0, GS_REPO)

from scene.gaussian_model import GaussianModel          # noqa: E402
from scene.cameras import MiniCam                        # noqa: E402
from gaussian_renderer import render                     # noqa: E402
from utils.graphics_utils import getProjectionMatrix, focal2fov  # noqa: E402


# ----------------------------- 회전 유틸 (numpy, scipy 없이) -----------------------------
def mat2quat(R):
    """3x3 회전행렬 -> 쿼터니언 [w,x,y,z] (Shepperd)."""
    m = R
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat2mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def slerp(q0, q1, t):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta0 * t
    s0 = math.sin(theta0 - theta) / math.sin(theta0)
    s1 = math.sin(theta) / math.sin(theta0)
    return s0 * q0 + s1 * q1


def axis_angle_mat(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float64)


# ----------------------------- 카메라 로드/경로 -----------------------------
def load_cameras(model_dir):
    with open(os.path.join(model_dir, "cameras.json"), "r") as f:
        cams = json.load(f)
    # img_name 숫자 순으로 정렬 = 촬영(주행) 순서
    cams.sort(key=lambda c: int("".join(ch for ch in c["img_name"] if ch.isdigit())))
    centers = np.array([c["position"] for c in cams], dtype=np.float64)
    rots = np.array([c["rotation"] for c in cams], dtype=np.float64)  # C2W 회전
    ref = cams[len(cams) // 2]
    intr = dict(fx=ref["fx"], fy=ref["fy"], w=ref["width"], h=ref["height"])
    return centers, rots, intr


def moving_average(arr, win):
    if win <= 1:
        return arr
    pad = win // 2
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones(win) / win
    return np.stack([np.convolve(padded[:, d], ker, mode="valid") for d in range(arr.shape[1])], axis=1)


def resample_path(centers, rots, n_out, smooth_win=5):
    """N개 제어 카메라를 부드럽게 n_out개 waypoint로 재보간."""
    centers_s = moving_average(centers, smooth_win)
    quats = np.array([mat2quat(R) for R in rots])
    n_ctrl = len(centers_s)
    ts = np.linspace(0, n_ctrl - 1, n_out)
    out_pos, out_rot = [], []
    for t in ts:
        i0 = int(math.floor(t))
        i1 = min(i0 + 1, n_ctrl - 1)
        f = t - i0
        pos = centers_s[i0] * (1 - f) + centers_s[i1] * f
        q = slerp(quats[i0], quats[i1], f)
        out_pos.append(pos)
        out_rot.append(quat2mat(q))
    return np.array(out_pos), np.array(out_rot)


def build_minicam(rot_c2w, pos, fovx, fovy, width, height, znear=0.01, zfar=100.0):
    C2W = np.eye(4, dtype=np.float64)
    C2W[:3, :3] = rot_c2w
    C2W[:3, 3] = pos
    W2C = np.linalg.inv(C2W)
    world_view = torch.tensor(W2C, dtype=torch.float32).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0, 1).cuda()
    full = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    return MiniCam(width, height, fovy, fovx, znear, zfar, world_view, full)


def perturb_pose(rot, pos, rng, unit, cfg):
    """도메인 랜덤화: 카메라 로컬 좌표계에서 위치/자세 흔들기.
    이동량은 씬 공간 크기(unit=카메라 궤적 bbox 대각)에 비례 → 프레임 수와 무관."""
    right = rot[:, 0]
    up = -rot[:, 1]  # 카메라 y는 아래 → up은 -y
    d_lat = rng.uniform(-cfg["lateral"], cfg["lateral"]) * unit
    d_up = rng.uniform(-cfg["height"], cfg["height"]) * unit
    d_fwd = rng.uniform(-cfg["forward"], cfg["forward"]) * unit
    new_pos = pos + d_lat * right + d_up * up + d_fwd * rot[:, 2]
    yaw = math.radians(rng.uniform(-cfg["yaw_deg"], cfg["yaw_deg"]))
    pitch = math.radians(rng.uniform(-cfg["pitch_deg"], cfg["pitch_deg"]))
    new_rot = rot @ axis_angle_mat([0, 1, 0], yaw) @ axis_angle_mat([1, 0, 0], pitch)
    return new_rot, new_pos, dict(d_lat=d_lat, d_up=d_up, d_fwd=d_fwd,
                                  yaw_deg=math.degrees(yaw), pitch_deg=math.degrees(pitch))


def photometric(img, rng, cfg):
    """렌더 텐서(3,H,W)에 조도/화이트밸런스/감마 변형 = 조명 랜덤화 흉내."""
    bright = rng.uniform(*cfg["brightness"])
    gamma = rng.uniform(*cfg["gamma"])
    wb = torch.tensor([rng.uniform(*cfg["wb"]) for _ in range(3)],
                      device=img.device).view(3, 1, 1)
    out = torch.clamp(img * bright * wb, 0, 1) ** gamma
    return torch.clamp(out, 0, 1), dict(brightness=bright, gamma=gamma,
                                        wb=[float(x) for x in wb.flatten().cpu()])


def render_pass(name, poses, gaussians, pipe, bg, intr_fov, out_res, out_dir,
                unit, rng=None, pose_cfg=None, photo_cfg=None):
    pass_dir = os.path.join(out_dir, name)
    os.makedirs(pass_dir, exist_ok=True)
    fovx, fovy = intr_fov
    W, H = out_res
    meta = []
    positions, rotations = poses
    step = float(np.median(np.linalg.norm(np.diff(positions, axis=0), axis=1))) if len(positions) > 1 else 1.0
    for idx in range(len(positions)):
        rot, pos = rotations[idx], positions[idx]
        prec = {}
        if pose_cfg is not None:
            rot, pos, prec = perturb_pose(rot, pos, rng, unit, pose_cfg)
        cam = build_minicam(rot, pos, fovx, fovy, W, H)
        with torch.no_grad():
            img = render(cam, gaussians, pipe, bg, separate_sh=True)["render"]
        photo = {}
        if photo_cfg is not None:
            img, photo = photometric(img, rng, photo_cfg)
        fname = f"{name}_{idx:04d}.png"
        torchvision.utils.save_image(img, os.path.join(pass_dir, fname))
        C2W = np.eye(4); C2W[:3, :3] = rot; C2W[:3, 3] = pos
        meta.append(dict(frame=fname, c2w=C2W.tolist(), fovx=fovx, fovy=fovy,
                         width=W, height=H, perturb=prec, photometric=photo))
    with open(os.path.join(pass_dir, "metadata.json"), "w") as f:
        json.dump(dict(pass_name=name, count=len(meta), step_median=step, frames=meta),
                  f, indent=1)
    print(f"[{name}] {len(meta)} frames -> {pass_dir}")
    return len(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required=True)
    ap.add_argument("-o", "--out_dir", required=True)
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--nominal", type=int, default=150, help="nominal 패스 프레임 수")
    ap.add_argument("--dr", type=int, default=200, help="도메인 랜덤화 패스 프레임 수")
    ap.add_argument("--smooth", type=int, default=5, help="경로 위치 이동평균 창")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sh_degree", type=int, default=3)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    centers, rots, intr = load_cameras(args.model_dir)
    extent = float(np.linalg.norm(centers.max(0) - centers.min(0)))  # 궤적 bbox 대각 = 흔들기 단위
    print(f"loaded {len(centers)} training cams; intr fx={intr['fx']:.1f} "
          f"w={intr['w']} h={intr['h']}; trajectory extent={extent:.3f}")

    # FoV는 원본 intrinsics에서 (해상도만 바꿔도 FoV 보존)
    fovx = focal2fov(intr["fx"], intr["w"])
    fovy = focal2fov(intr["fy"], intr["h"])
    aspect = intr["w"] / intr["h"]
    W = args.width
    H = int(round(W / aspect))

    # 모델 로드
    gaussians = GaussianModel(args.sh_degree)
    ply = os.path.join(args.model_dir, "point_cloud",
                       f"iteration_{args.iteration}", "point_cloud.ply")
    gaussians.load_ply(ply)
    print(f"loaded {gaussians.get_xyz.shape[0]:,} splats from {ply}")

    pipe = SimpleNamespace(debug=False, antialiasing=False,
                           compute_cov3D_python=False, convert_SHs_python=False)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = dict(model_dir=args.model_dir, iteration=args.iteration,
                    resolution=[W, H], fovx_deg=math.degrees(fovx),
                    fovy_deg=math.degrees(fovy), trajectory_extent=extent,
                    seed=args.seed, passes=[])

    # 1) nominal 주행 패스 (깨끗한 경로, 데모/기준)
    base = resample_path(centers, rots, args.nominal, smooth_win=args.smooth)
    n1 = render_pass("nominal", base, gaussians, pipe, bg, (fovx, fovy), (W, H),
                     args.out_dir, extent)
    manifest["passes"].append(dict(name="nominal", count=n1, perturb=False, photo=False))

    # 2) 도메인 랜덤화 패스 (시점 흔들기 + 조도/WB/감마). 이동량은 extent 비율.
    pose_cfg = dict(lateral=0.03, height=0.02, forward=0.02, yaw_deg=7.0, pitch_deg=3.5)
    photo_cfg = dict(brightness=(0.7, 1.3), gamma=(0.8, 1.25), wb=(0.9, 1.1))
    dr = resample_path(centers, rots, args.dr, smooth_win=args.smooth)
    n2 = render_pass("domain_random", dr, gaussians, pipe, bg, (fovx, fovy), (W, H),
                     args.out_dir, extent, rng=rng, pose_cfg=pose_cfg, photo_cfg=photo_cfg)
    manifest["passes"].append(dict(name="domain_random", count=n2, perturb=pose_cfg,
                                   photo=photo_cfg))

    manifest["total_frames"] = n1 + n2
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"DONE total={n1 + n2} frames; manifest -> {args.out_dir}/manifest.json")


if __name__ == "__main__":
    main()
