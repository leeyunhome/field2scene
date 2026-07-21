# P2 — 합성 데이터 생성 결과

> 완료 2026-07-20. 학습된 토마토 온실 3DGS 씬에서 로봇 주행 시점 카메라 경로를 만들어 novel-view 프레임을 렌더 = 합성 데이터셋.

## 방법

학습 카메라 190대는 온실 통로를 따라간 dolly 샷의 궤적을 이룬다. 이를 **주행 경로의 척추**로 삼아:

1. **경로 재보간**: 190개 학습 카메라 pose를 위치 이동평균(창 5)으로 흔들림 제거 후, 위치 lerp + 회전 **쿼터니언 SLERP**로 부드럽게 재샘플링. → 학습 프레임 사이의 진짜 novel view.
2. **nominal 패스 (150장)**: 변형 없는 깨끗한 주행 경로. 데모/기준용.
3. **domain_random 패스 (200장)**: Sim-to-Real 대비 도메인 랜덤화 —
   - **시점 흔들기**(카메라 로컬 좌표계): 횡이동 ±0.03·extent, 높이 ±0.02·extent, 전후 ±0.02·extent, yaw ±7°, pitch ±3.5° (extent = 궤적 bbox 대각 = 11.13). 로봇이 이랑에서 다른 위치·자세로 지나가는 상황 흉내.
   - **광학 변형**(렌더 후): 밝기 ×[0.7,1.3], 감마 [0.8,1.25], 채널별 화이트밸런스 [0.9,1.1]. 조명·카메라 편차 흉내.

렌더러는 INRIA gaussian-splatting을 재사용(`MiniCam` + `render(separate_sh=True)`). **재학습·재구성 없음** — 학습된 씬에서 렌더만.

## 산출물

| 항목 | 값 |
|---|---|
| 총 프레임 | **350장** (nominal 150 + domain_random 200) |
| 해상도 | 1280×719, FoV 60.2° (학습 intrinsics 유지) |
| 씬 | tomato_greenhouse (1.33M splat, test PSNR 30.73) |
| 데이터 위치 | `data/synth_tomato_greenhouse/` (gitignore, 429MB) |
| 메타데이터 | 프레임별 `c2w`·FoV·perturb·photometric 파라미터 → `<pass>/metadata.json`; 전체 요약 `manifest.json` |

프레임별 카메라 pose와 변형 파라미터를 metadata.json에 남겨 **완전 재현 가능**(seed 고정). P3에서 프레임 조건별(거리·조도) 검출률 분석에 활용.

## 스크립트

`scripts/render_driving_path.py` — 단일 파일, numpy만 추가 의존(scipy 없이 쿼터니언 SLERP 직접 구현).

```bash
GS_REPO=<INRIA repo> python scripts/render_driving_path.py \
    -m output/tomato_greenhouse -o data/synth_tomato_greenhouse \
    --width 1280 --nominal 150 --dr 200 --seed 0
```

## 검증

- nominal 주행 경로: `assets/p2_driving_path.gif` (통로를 따라 전진하는 로봇 시점).
- 도메인 랜덤화 다양성: `assets/p2_domain_random.jpg` (시점·조도·위치 변형 6종).
- 품질: 경로가 학습 궤적 근처에 머물러(P1 한계 반영) 근·중경 선명 유지. 통로 원거리·좌측 벽 밖은 여전히 흐릿 → P3 갭 소재.

## 교훈 / 정직 노트

1. 이 rasterizer 빌드(newer, dc/shs 분리)는 `render(separate_sh=True)`만 정상 — `False`는 CUDA illegal access. 학습에 쓴 호출과 일치시켜야 함.
2. 도메인 랜덤화 이동량은 **씬 extent 비율**로 정의해 프레임 수·씬 스케일과 무관하게 일정. (COLMAP 좌표는 임의 스케일이므로 절대값 하드코딩 금지)
3. "합성 데이터"는 **한 실촬영 씬의 novel view + 증강**이지, 새로운 씬을 만든 게 아니다. P3 갭 분석·README에 이 정의를 명확히 밝힌다.
