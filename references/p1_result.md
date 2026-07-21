# P1 — 씬 재구성 결과 (토마토 온실)

> 완료 2026-07-19. 씬 확보 → COLMAP → 3DGS 학습까지 완주. 검증: held-out(test) PSNR + 육안 비교.

## 파이프라인 실측

| 단계 | 도구 | 결과 |
|---|---|---|
| 소스 | Pexels "Tomato farm in greenhouse" (ID 10179855, Mehdi Alaoui, Pexels License) | 1920×1080 30fps 9s 원테이크 달리샷 |
| 프레임 추출 | ffmpeg (t=1.8s부터 전프레임) | 217장 |
| SfM | COLMAP (feature_extractor OPENCV + **sequential_matcher** overlap 25 + mapper 완화) | **190/217장 등록(88%)**, 20,641 pts, 재투영오차 0.67px |
| 3DGS 학습 | INRIA gaussian-splatting, `train.py --eval --iterations 30000` | 1,330,294 splat, 314.6MB, 4h45m (RTX 4060 8GB) |

## 품질 (held-out test = 학습 미사용 프레임, --eval llffhold=8)

| iter | test PSNR | test L1 | train PSNR |
|---|---|---|---|
| 7000 | 27.71 | 0.0268 | 27.79 |
| **30000** | **30.73** | **0.0198** | 31.08 |

- 실사 씬에서 test 30.7dB는 우수. train-test 격차 0.35dB로 일반화 양호(과적합 아님).
- 육안 비교(`assets/p1_compare_00005.jpg`, `p1_compare_00020.jpg`, 좌 GT / 우 렌더): 근·중경(토마토·물방울·줄기·바닥 멀칭)은 원본과 거의 구별 불가.

## 알려진 한계 (숨기지 않음 — P3 갭 분석 소재)

- **원거리·통로 끝 뭉개짐**: 관측 프레임이 적은 먼 영역(통로 안쪽, GT의 서 있는 사람)이 흐릿한 blob으로 재현됨. 카메라 경로 근처만 신뢰.
- **얇은 기하 노이즈**: 줄기·잎 경계에 소량 floater 가능.
- → P2 카메라 경로는 관측이 조밀한 통로 근접 구간으로 한정, P3에서 "거리별 검출률 저하"로 정량화 예정.

## 재현 절차 (요약)

```bash
# env: conda gaussian_splatting / repo: HonglabSplatting/gaussian-splatting
ffmpeg -ss 1.8 -i pexels_10179855.mp4 -q:v 2 input/%04d.jpg
colmap feature_extractor --database_path distorted/database.db --image_path input \
  --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --SiftExtraction.use_gpu 1
colmap sequential_matcher --database_path distorted/database.db --SequentialMatching.overlap 25 --SiftMatching.use_gpu 1
colmap mapper --database_path distorted/database.db --image_path input --output_path distorted/sparse \
  --Mapper.init_min_tri_angle 4 --Mapper.abs_pose_min_num_inliers 20 --Mapper.min_num_matches 10
# 가장 큰 모델(여기선 sparse/3, 190장) 선택 후:
colmap image_undistorter --image_path input --input_path distorted/sparse/3 --output_path . --output_type COLMAP
#   → sparse/ 안의 3파일을 sparse/0/ 으로 이동 (INRIA 포맷)
python train.py -s <scene> -m <out> --eval --iterations 30000 --save_iterations 7000 30000
python render.py -m <out> --skip_train   # test held-out 렌더
```

## P1 교훈 (다음 씬/재학습 시 반영)

1. **전진(달리) 모션 영상은 exhaustive 매칭으로 실패** — convert.py 기본이 4장만 등록. **sequential_matcher 필수**.
2. 전진 모션은 SfM이 여러 sub-model로 쪼개짐 — model_analyzer로 최대 모델 선택 필요.
3. 이 씬은 8GB에서 기본 30k가 **4h45m로 과도하게 느림**(1.33M splat, VRAM 96%). 시간 급하면 `-r 2`(해상도↓)나 densification 억제로 30분대 완주 가능. 품질(30.7dB)엔 만족하므로 v1은 이대로 채택.
4. 결과물 315MB — 웹 공개(splatting-viewer) 전 경량화 필요(Mechander clean_scan 파이프라인 재사용, P6).
