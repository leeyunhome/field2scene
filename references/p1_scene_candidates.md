# P1 — 씬 소스 선정 (공개 자료 트랙)

> 결정 2026-07-19: 직접 촬영 여유 없음 → 공개 자료 사용. 낱장 풍경 사진은 불가(다시점 겹침 필요) — **한 씬을 연속으로 훑은 영상 1개**가 소스 단위.

## ✅ 확정: 주 씬 — 토마토 온실 (Pexels 10179855)

- **"Tomato farm in greenhouse"** by **Mehdi Alaoui** — https://www.pexels.com/video/tomato-farm-in-greenhouse-10179855/
- 라이선스: Pexels License (자유 사용·수정 허용) — README에 링크+작가명 명시
- 1920×1080(가로) · 30fps · 9.0s · **원테이크 확인**(scene-cut 검출 0개) · 토마토 클로즈업→통로 전경으로 빠지는 달리샷
- **선정 이유: 상업용 토마토 온실 통로 = 농업 수확 로봇 도메인(토마토)과 잘 맞음.** 작물 열 + 주행 통로 구조라 P2 "주행 시점 렌더" 스토리가 자연스러움
- 전처리: 초반 1.5s는 매크로 아웃포커스(배경 블러)라 제외. t=1.5s부터 2프레임 간격(실효 15fps) 추출 → **113장**
- 리스크 메모: 얇은 줄기·잎 기하는 3DGS 아티팩트 가능성, 통로 원거리는 재구성 품질 저하 예상(카메라 경로 근처만 사용)

### P3 파급: 토마토는 COCO 클래스에 없음

렌더 vs 실프레임 YOLO 비교 시 기본 COCO 모델로는 tomato 검출 불가. 대안:
1. 오픈보캡 검출(YOLO-World 등)에 "tomato" 프롬프트 — 학습 불필요, 1순위
2. 공개 토마토 데이터셋 사전학습 가중치(laboro-tomato 등) — 실데이터 사전학습이므로 합성 데이터로 재학습하지 않는 이 프로젝트 방침과도 맞음

## 백업: 식물원 온실 통로 (Pexels 34918438)

- **"Lush Botanical Greenhouse with Pathway"** by **Alex Ohan** — https://www.pexels.com/video/lush-botanical-greenhouse-with-pathway-34918438/
- 1080×1920(세로) · 30fps · 21.2s · 원테이크 확인(컷 0). 통로·다리·연못·화분 다수 → COCO potted plant로 P3 가능
- 단점: 세로 화면(주행 카메라 스토리에 부자연), 연못 반사(아티팩트 요인). 주 씬 실패 시에만 사용

## 처리 절차 (검증된 INRIA 환경 — 이 PC)

```
# 1) 프레임 추출 (data/scenes/<scene>/input/)
ffmpeg -ss 1.5 -i <영상.mp4> -vf "select='not(mod(n,2))'" -vsync vfr -q:v 2 input/%04d.jpg

# 2) COLMAP SfM (INRIA 리포에서, gaussian_splatting conda env)
python convert.py -s <scene경로> --camera OPENCV --colmap_executable <colmap.exe>

# 3) 3DGS 학습 (--eval: 8장당 1장 test 홀드아웃 → P3의 "실촬영 역할" 프레임)
python train.py -s <scene경로> --eval --iterations 30000
```

- nerfstudio 아님 — 기존에 검증된 INRIA gaussian-splatting 환경 재사용
- **held-out 분리**: `--eval`(llffhold=8)로 학습 제외 프레임 확보 → P3에서 렌더 vs held-out 비교
- 산출물(data/·output/)은 커밋 금지 — .gitignore 적용 확인

## P3 방법론 메모

직접 촬영이 없으므로 갭 리포트의 비교쌍은 **"3DGS 렌더 vs 원본 영상 held-out 프레임(학습 미사용)"**. train/test split 표준 방식이며 README·리포트에 이 정의를 명시(과장 없음).
