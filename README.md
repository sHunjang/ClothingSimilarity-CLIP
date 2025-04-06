
# 👕 CLIP 기반 의류 이미지 유사도 측정 실험

사용자가 저장한 옷 이미지들과 스타일 참조 이미지(Query) 간의 유사도를  
[OpenAI CLIP](https://github.com/openai/CLIP) 모델을 활용해 측정하고,  
Top-K 유사도 결과를 기반으로 추천 정확도를 평가하는 실험 코드

---

## ✅ 주요 기능

- CLIP ViT-B/32 모델 기반 이미지 임베딩
- 쿼리 이미지와 데이터셋 이미지 간 cosine similarity 계산
- Top-K 유사 클래스 결과 추출 및 정답 포함 여부 평가
- 실험 결과를 CSV 파일로 저장

---

## 📁 폴더 구조 예시

```
project/
├── dataset/         # 실험용 의류 이미지 데이터셋 (업로드 제외 대상)
│   └── tops/
│       ├── hoodie/
│       ├── tshirt/
├── queries/         # 스타일 참조 이미지 (쿼리용)
│   └── tops/
├── clip_similarity.py
├── clip_results.csv # 실험 결과
├── README.md
└── .gitignore
```

---

## ⚙️ 환경 구성 (Windows + GPU 기준)

```bash
conda create -n clip-env python=3.10 -y
conda activate clip-env

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/openai/CLIP.git
pip install numpy pillow tqdm
```

> Mac 환경에서는 `pytorch`만 설치하면 자동으로 MPS(GPU) 지원.
> `--index-url` 옵션은 Windows/CUDA 환경에서만 사용.

---

## 🚀 실행 방법

```bash
python clip_similarity.py
```

> 결과는 `clip_results.csv`로 저장됨.

---

## 🧪 쿼리 파일명 규칙

쿼리 이미지는 `정답클래스_번호.jpg` 형식으로 저장되어 있어야 함.  
예시: `hoodie_01.jpg`, `tshirt_02.jpg`

---

## 📊 출력 예시

```
🔍 Query: hoodie_01.jpg | 정답: hoodie
  1. hoodie (0.9123)
  2. hoodie (0.8991)
  3. tshirt (0.8342)
  ...

✅ Top-5 Accuracy: 90.00%
```

---

## 🧠 향후 계획

- [ ] DINOv2 모델과의 성능 비교 실험
- [ ] Triplet Loss 기반 유사도 모델 직접 학습
- [ ] 시각화 기능 추가 (query + top-K 이미지 출력)

---

## 📄 라이선스

MIT License
