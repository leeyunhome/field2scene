# Field2Scene

실외 환경을 3D Gaussian Splatting으로 재구성하고, 그 씬에서 합성 데이터를 생성해 Sim-to-Real 갭을 검증하고, ROS2 심 카메라로 퍼블리시하는 Real2Sim 파이프라인.

```
촬영/공개 영상 → COLMAP → 3DGS 학습 → 주행 시점 렌더(합성 데이터)
→ YOLO 렌더 vs 실촬영 갭 리포트 → ROS2 sensor_msgs 퍼블리시 → (선택) 메시 추출·Gazebo
```

> 진행 중 — 단계별 현황은 커밋 히스토리 참조.

## P1 — 씬 재구성 (완료)

상업용 **토마토 온실** 씬을 3DGS로 재구성했다. 소스는 Pexels 공개 영상 "Tomato farm in greenhouse" (by Mehdi Alaoui, [Pexels License](https://www.pexels.com/license/), [영상 링크](https://www.pexels.com/video/tomato-farm-in-greenhouse-10179855/)) — 직접 촬영이 아니며 출처를 밝힌다.

| 항목 | 값 |
|---|---|
| COLMAP 등록 | 190 / 217 프레임 (sequential matching) |
| 3DGS | 1.33M splat, 30k iter, RTX 4060 Laptop 8GB |
| **held-out(test) PSNR** | **30.73 dB** (train 31.08 — 과적합 아님) |

좌: 원본 프레임(GT) / 우: 3DGS 렌더(학습에 쓰지 않은 held-out 뷰).

![근경 비교](assets/p1_compare_00005.jpg)
![통로 비교](assets/p1_compare_00020.jpg)

정직 노트: 근·중경은 원본과 거의 구별되지 않으나, 관측이 적은 **통로 안쪽 원거리는 흐릿하게** 재현된다(P3 갭 분석에서 정량화 예정). 재현 절차·파라미터는 [references/p1_result.md](references/p1_result.md) 참조.

## P2 — 합성 데이터 생성 (완료)

학습된 씬에서 **로봇 주행 시점 카메라 경로**를 만들어 novel-view 프레임 **350장**을 렌더했다. 학습 카메라(통로 dolly 궤적)를 SLERP로 재보간해 주행 경로로 쓰고, 도메인 랜덤화(시점 흔들기 + 조도/화이트밸런스/감마)로 Sim-to-Real 비교용 데이터셋을 구성한다. 재학습·재구성 없이 렌더만 — 스크립트 [scripts/render_driving_path.py](scripts/render_driving_path.py), 상세 [references/p2_result.md](references/p2_result.md).

| 패스 | 프레임 | 내용 |
|---|---|---|
| nominal | 150 | 변형 없는 주행 경로 (기준) |
| domain_random | 200 | 시점 ±(횡0.03·높이0.02·yaw7°·pitch3.5°) + 조도×[0.7,1.3]·감마·WB |

프레임별 pose·변형 파라미터를 metadata.json에 기록(seed 고정 → 완전 재현).

![주행 경로 렌더](assets/p2_driving_path.gif)

도메인 랜덤화 다양성 (시점·조도·위치):

![도메인 랜덤화](assets/p2_domain_random.jpg)

## P3 — Sim-to-Real 갭 리포트 (완료)

**동일 카메라 pose**의 실촬영 프레임 vs 3DGS 렌더(test held-out 24뷰)에 같은 오픈보캡 검출기(YOLO-World, `"tomato"` 프롬프트)를 돌려 검출기 반응의 도메인 갭을 정량화했다. 스크립트 [scripts/gap_report.py](scripts/gap_report.py), 상세 [references/p3_result.md](references/p3_result.md).

| 지표 (24쌍, conf≥0.10) | 실촬영 | 3DGS 렌더 |
|---|---|---|
| 프레임당 검출 수 | 7.79 | 7.29 (×0.94) |
| 평균 신뢰도 | 0.196 | 0.195 |
| IoU≥0.5 매칭 | recall 0.909 · precision 0.971 | |

**갭이 작다** — 3DGS 렌더는 YOLO 검출 관점에서 실사와 거의 동등하며, 남은 6% 검출 손실은 대부분 **원거리 작은 토마토**(P1의 원거리 blur 한계가 downstream으로 이어짐). 라벨 정확도가 아니라 검출기 반응의 상대 갭임을 명시한다.

![검출 갭 차트](assets/p3_gap_charts.png)

좌: 실촬영(초록) / 우: 3DGS 렌더(주황) — 같은 토마토가 같은 위치에 검출됨.

![검출 비교](assets/p3_detection_pair.jpg)

## 구조 (예정)

```
field2scene/
├── scripts/          # 프레임 추출 · 주행 경로 렌더 · 갭 리포트
├── ros2_ws/          # 3DGS 심 카메라 ROS2 패키지
├── report/           # Sim-to-Real 갭 분석
└── assets/           # 데모 GIF · 결과 그래프
```
