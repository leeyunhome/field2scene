#!/usr/bin/env python
"""
Field2Scene P3 — Sim-to-Real 갭 리포트.

동일 카메라 pose에서 (실촬영 프레임) vs (3DGS 렌더)에 같은 오픈보캡 검출기를
돌려, 검출기가 합성 이미지에서 실사 대비 얼마나 다르게 반응하는지 정량화한다.

비교쌍 = 학습에서 제외한 test held-out 뷰:
  <model>/test/ours_<it>/gt/NNNNN.png       (실촬영 = 원본 영상 프레임)
  <model>/test/ours_<it>/renders/NNNNN.png  (같은 pose의 3DGS 렌더)
이 쌍은 시점이 동일하므로 화질/아티팩트 차이만 분리해서 볼 수 있다.

수동 라벨이 없으므로 "실촬영 검출"을 기준(pseudo-GT)으로 삼아:
  - 검출 수 비율 (render/real)
  - 평균 신뢰도 (real vs render)
  - IoU>=thr 매칭률: 실촬영 박스가 렌더에서도 잡히는 비율(recall 유사),
                     렌더 박스가 실촬영에도 있는 비율(precision 유사)
을 프레임별·전체로 집계한다. 라벨 정확도가 아니라 "검출기 반응의 도메인 갭"임을 명시.

사용:
  python gap_report.py --pairs <model>/test/ours_30000 \
      [--synth data/synth_.../nominal] [--conf 0.10] [--iou 0.5] \
      [--model yolov8s-worldv2.pt] [--classes tomato] -o report/
"""
import os
import csv
import json
import glob
import argparse

import numpy as np
import cv2


def load_detector(model_name, classes):
    from ultralytics import YOLOWorld
    m = YOLOWorld(model_name)
    m.set_classes(classes)
    return m


def detect(model, path, conf):
    r = model.predict(path, conf=conf, verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
    confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))
    return boxes, confs


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def greedy_match(iou, thr):
    """실촬영 박스 i가 렌더 박스 j와 IoU>=thr로 1:1 매칭된 수."""
    matched = 0
    if iou.size == 0:
        return 0
    iou = iou.copy()
    while True:
        i, j = np.unravel_index(np.argmax(iou), iou.shape)
        if iou[i, j] < thr:
            break
        matched += 1
        iou[i, :] = -1
        iou[:, j] = -1
    return matched


def draw(path, boxes, confs, color):
    img = cv2.imread(path)
    for (x1, y1, x2, y2), c in zip(boxes.astype(int), confs):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{c:.2f}", (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="test/ours_<it> 디렉토리 (gt/ 와 renders/ 포함)")
    ap.add_argument("--synth", default=None, help="합성 주행 프레임 디렉토리 (선택, 검출률만)")
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    ap.add_argument("--classes", nargs="+", default=["tomato"])
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("-o", "--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    viz_dir = os.path.join(args.out_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    model = load_detector(args.model, args.classes)

    gt_dir = os.path.join(args.pairs, "gt")
    ren_dir = os.path.join(args.pairs, "renders")
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(gt_dir, "*.png")))
    print(f"{len(names)} paired views; detector={args.model} classes={args.classes} "
          f"conf={args.conf} iou={args.iou}")

    rows = []
    agg = dict(real_dets=0, render_dets=0, matched=0,
               real_conf=[], render_conf=[])
    for k, name in enumerate(names):
        rp, rc = detect(model, os.path.join(gt_dir, name), args.conf)     # real
        sp, sc = detect(model, os.path.join(ren_dir, name), args.conf)    # synthetic render
        m = greedy_match(iou_matrix(rp, sp), args.iou)
        rows.append(dict(view=name, real_dets=len(rp), render_dets=len(sp),
                         matched=m,
                         real_conf_mean=float(np.mean(rc)) if len(rc) else 0.0,
                         render_conf_mean=float(np.mean(sc)) if len(sc) else 0.0))
        agg["real_dets"] += len(rp)
        agg["render_dets"] += len(sp)
        agg["matched"] += m
        agg["real_conf"] += list(map(float, rc))
        agg["render_conf"] += list(map(float, sc))
        # 앞쪽 3쌍은 검출 시각화 저장
        if k < 3:
            vr = draw(os.path.join(gt_dir, name), rp, rc, (0, 200, 0))
            vs = draw(os.path.join(ren_dir, name), sp, sc, (0, 165, 255))
            h = min(vr.shape[0], vs.shape[0])
            combo = np.hstack([cv2.resize(vr, (int(vr.shape[1] * h / vr.shape[0]), h)),
                               cv2.resize(vs, (int(vs.shape[1] * h / vs.shape[0]), h))])
            cv2.imwrite(os.path.join(viz_dir, f"pair_{name.replace('.png','')}.jpg"), combo)

    # per-frame CSV
    with open(os.path.join(args.out_dir, "per_frame.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(names)
    rc_arr, sc_arr = np.array(agg["real_conf"]), np.array(agg["render_conf"])
    summary = dict(
        n_pairs=n, conf_thr=args.conf, iou_thr=args.iou,
        model=args.model, classes=args.classes,
        real_dets_total=agg["real_dets"], render_dets_total=agg["render_dets"],
        real_dets_per_frame=round(agg["real_dets"] / n, 2),
        render_dets_per_frame=round(agg["render_dets"] / n, 2),
        detection_count_ratio=round(agg["render_dets"] / max(agg["real_dets"], 1), 3),
        real_conf_mean=round(float(rc_arr.mean()) if rc_arr.size else 0, 3),
        render_conf_mean=round(float(sc_arr.mean()) if sc_arr.size else 0, 3),
        conf_delta=round(float((sc_arr.mean() if sc_arr.size else 0) -
                               (rc_arr.mean() if rc_arr.size else 0)), 3),
        matched_total=agg["matched"],
        recall_render_vs_real=round(agg["matched"] / max(agg["real_dets"], 1), 3),
        precision_render_vs_real=round(agg["matched"] / max(agg["render_dets"], 1), 3),
    )

    # 합성 주행 프레임 검출률 (실촬영 짝 없음 — 보조 지표)
    if args.synth:
        sframes = sorted(glob.glob(os.path.join(args.synth, "*.png")))
        tot = 0
        for p in sframes:
            _, c = detect(model, p, args.conf)
            tot += len(c)
        summary["synth_frames"] = len(sframes)
        summary["synth_dets_per_frame"] = round(tot / max(len(sframes), 1), 2)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # ---- 차트 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(args.conf, 1.0, 18)
    ax[0].hist(rc_arr, bins=bins, alpha=0.6, label="real", color="#2c7")
    ax[0].hist(sc_arr, bins=bins, alpha=0.6, label="3DGS render", color="#f83")
    ax[0].set_title("Detection confidence distribution")
    ax[0].set_xlabel("confidence"); ax[0].set_ylabel("# detections"); ax[0].legend()

    per_real = [r["real_dets"] for r in rows]
    per_ren = [r["render_dets"] for r in rows]
    x = np.arange(n)
    ax[1].bar(x - 0.2, per_real, width=0.4, label="real", color="#2c7")
    ax[1].bar(x + 0.2, per_ren, width=0.4, label="3DGS render", color="#f83")
    ax[1].set_title("Detections per paired view")
    ax[1].set_xlabel("paired view idx"); ax[1].set_ylabel("# tomato detections"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "gap_charts.png"), dpi=110)

    print("SUMMARY:", json.dumps(summary, indent=1))
    print(f"-> {args.out_dir}/summary.json, per_frame.csv, gap_charts.png, viz/")


if __name__ == "__main__":
    main()
