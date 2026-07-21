# P3 — Sim-to-Real 갭 리포트

> 완료 2026-07-20. 동일 카메라 pose의 실촬영 vs 3DGS 렌더에 같은 오픈보캡 검출기를 돌려, 합성 이미지에서 검출기 반응이 실사 대비 얼마나 벌어지는지 정량화.

## 실험 설계

- **비교쌍**: 학습에서 제외한 test held-out 24뷰. 각 뷰마다
  - 실촬영 = 원본 영상 프레임 (`test/ours_30000/gt/`)
  - 3DGS 렌더 = **같은 pose**의 novel-view 렌더 (`test/ours_30000/renders/`)
  - 시점이 동일하므로 화질·아티팩트 차이만 분리 관찰 가능.
- **검출기**: YOLO-World (`yolov8s-world.pt`) 오픈보캡, 프롬프트 `"tomato"` (토마토는 COCO 클래스에 없어 오픈보캡 사용 — 재학습 없음).
- **동작점**: conf ≥ 0.10, IoU 매칭 ≥ 0.5.
- **지표**(수동 라벨 없음 → 실촬영 검출을 pseudo-GT로): 검출 수 비율, 평균 신뢰도, IoU 매칭 기반 recall/precision. **라벨 정확도가 아니라 "검출기 반응의 도메인 갭"**임을 명시.

## 결과 (24쌍)

| 지표 | 실촬영 | 3DGS 렌더 | 갭 |
|---|---|---|---|
| 프레임당 검출 수 | 7.79 | 7.29 | ×0.936 (렌더가 6.4%↓) |
| 평균 신뢰도 | 0.196 | 0.195 | −0.001 (거의 동일) |
| 검출 총계 | 187 | 175 | −12 |

- **IoU≥0.5 매칭**: recall(실촬영 박스가 렌더에도 잡힘) **0.909**, precision(렌더 박스가 실촬영에도 있음) **0.971**.
- 신뢰도 분포가 거의 겹침(`assets/p3_gap_charts.png` 좌), 뷰별 검출 수도 밀접하게 추종(우).
- 검출 시각화(`assets/p3_detection_pair.jpg`, 좌 실촬영·초록 / 우 렌더·주황): 같은 토마토가 같은 위치에 검출됨.
- 보조: 합성 주행 프레임(nominal 150장)도 프레임당 6.77개 검출 — 주행 경로 전체가 검출 가능.

## 해석 (정직)

- **갭이 작다**: 3DGS 렌더는 이 온실 씬에서 YOLO 검출 관점으로 실사와 거의 동등(count 0.94, 신뢰도 동일, precision 0.97). 이는 P1의 test PSNR 30.7dB와 일관 — 근·중경 화질이 검출에 충분.
- **남은 6% 갭의 원인**: 렌더에서 놓친 검출은 대부분 **원거리·통로 안쪽의 작은 토마토** — P1에서 지적한 "관측 적은 원거리 blur" 한계가 그대로 downstream 검출 손실로 나타남. recall(0.909) < precision(0.971)이 이를 뒷받침(렌더는 있는 걸 지어내기보단 어려운 걸 놓치는 방향).
- **한계**: (1) 수동 라벨이 없어 절대 정확도가 아닌 상대 갭. (2) 단일 씬·단일 검출기·오픈보캡 프롬프트 의존. (3) test 뷰는 학습 궤적 위 → 큰 시점 이탈에서의 갭은 미측정(로드맵).

## 재현

```bash
# env: yolo-edge (ultralytics 8.4.75). 최초 실행 시 world 가중치 + CLIP 텍스트 인코더(338MB) 다운로드.
python scripts/gap_report.py \
    --pairs output/tomato_greenhouse/test/ours_30000 \
    --synth data/synth_tomato_greenhouse/nominal \
    --model yolov8s-world.pt --classes tomato --conf 0.10 --iou 0.5 \
    -o report/gap_tomato
# 산출: summary.json, per_frame.csv, gap_charts.png, viz/pair_*.jpg
```

per-frame CSV·summary JSON은 `report/gap_tomato/`에 커밋(소용량, 재현 검증용).
