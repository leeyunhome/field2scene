# 엔지니어링 로그 — 시행착오와 해결

파이프라인을 만들며 실제로 막혔던 지점과 원인·해결을 기록한다. 결과 수치는 각 `p{1,2,3}_result.md`에 있고, 여기서는 "왜 처음엔 안 됐는가"를 남긴다.

## 1. COLMAP: 전진(dolly) 영상에서 등록률 붕괴

- **증상**: `convert.py` 기본값(exhaustive matcher)으로 113프레임 중 **4프레임만 등록**되고 재구성 종료.
- **원인**: 소스 영상이 통로를 따라 앞으로 빠지는 dolly 샷이라 (1) 프레임 간 삼각측량 각이 작고, (2) 근접 물체가 프레임마다 크게 움직여 exhaustive 전역 매칭이 대응점을 못 이음. 15fps 샘플링이라 baseline이 더 벌어진 것도 악화 요인.
- **해결**:
  - 30fps로 전체 재추출(217장)해 프레임 간 겹침 확보.
  - `sequential_matcher --SequentialMatching.overlap 25` (인접 프레임 위주 매칭) + mapper 완화(`init_min_tri_angle 4`, `abs_pose_min_num_inliers 20`, `min_num_matches 10`).
  - 결과 **190/217 등록**, 재투영오차 0.67px.
- **추가 함정**: 전진 모션은 SfM이 여러 sub-model로 쪼개진다(여기선 4개). `model_analyzer`로 등록 수 최대 모델(190장짜리)을 골라 undistort해야 함 — 기본 `sparse/0`이 최대가 아닐 수 있다.

## 2. 8GB VRAM에서 3DGS 학습이 과도하게 느림

- **증상**: 30k iter 학습이 **4시간 45분** 소요(7k까지 19분인데 이후 급감속).
- **원인**: 조밀한 온실 씬(잎·줄기·수백 개 열매)이라 densification이 splat을 **1.33M(315MB)**까지 불렸고, VRAM이 96%까지 차 후반 iter가 크게 느려짐.
- **판단**: 품질(test PSNR 30.73dB)은 만족스러워 v1은 그대로 채택. 다만 8GB에서 기본 30k는 비효율.
- **다음에 할 것**: 해상도 다운스케일(`-r 2`)이나 densification 억제(`--densify_grad_threshold`↑, `--densify_until_iter`↓)로 splat 수를 줄이면 30분대 완주 가능. 웹 뷰어 공개 전에는 스플랫 경량화 필요(315MB는 웹 로딩 부담).

## 3. novel-view 렌더 시 CUDA illegal memory access

- **증상**: 커스텀 카메라 pose로 `render()` 호출 시 rasterizer가 access violation으로 크래시. 반면 `render.py`(학습 카메라 렌더)는 동일 모델에서 정상.
- **디버깅**: pose 계산을 먼저 의심 → C2W→W2C=inv(C2W) 방식이 `getWorld2View2(R,T)`와 **수치적으로 일치**(diff 3e-8)함을 확인해 pose는 배제. 이어 `render.py`와의 호출 차이를 비교.
- **원인**: 이 rasterizer 빌드는 dc/SH를 분리 전달하는 newer 버전이라 `render(separate_sh=True)`만 정상. `separate_sh=False`(combined SH 경로)는 텐서 shape 불일치로 illegal access.
- **해결**: `render.py`가 쓰던 것과 동일하게 `separate_sh=True`로 호출.

## 4. 도메인 랜덤화 세기가 프레임 수에 의존

- **증상**: 초기 구현에서 시점 흔들기 크기를 "출력 waypoint 간 이동거리"에 비례시켰더니, 프레임 수를 늘리면(간격이 좁아짐) 흔들기가 무의미하게 작아짐.
- **해결**: 흔들기 단위를 **카메라 궤적 bbox 대각(extent)**에 비례하도록 변경 → 프레임 수·씬 스케일과 무관하게 일정. (COLMAP 좌표는 임의 스케일이라 절대값 하드코딩은 금물.)

## 5. YOLO-World가 멈춘 것처럼 보임

- **증상**: 오픈보캡 검출 첫 실행이 5분 넘게 출력 없이 정지한 듯 보임.
- **원인**: `set_classes(["tomato"])` 시 CLIP 텍스트 인코더(**338MB**)를 처음 한 번 다운로드하는데, `conda run`의 출력 버퍼링 때문에 진행 표시가 안 보였을 뿐 실제로는 다운로드 중.
- **해결**: `python -u`(unbuffered) 직접 실행으로 진행률 확인. 이후 캐시되어 재실행은 즉시. 검출 자체는 정상(토마토는 COCO에 없어 오픈보캡 프롬프트로 접근).
