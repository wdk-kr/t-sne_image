# t-sne_image


> **AI는 이미지를 어떻게 분류하고 이상을 감지할까?**  
> 이 프로젝트는 사전 라벨 없이도 이미지 간 유사성을 분석하고, 이상 이미지를 자동으로 감지하는 비지도 이미지 분석 시스템입니다.  

---

## 🧠 프로젝트 소개

이 프로젝트는 다음과 같은 흐름으로 구성되어 있습니다:

1. 사전 학습된 CNN 모델(ResNet50)로 이미지의 Feature Vector 추출  
2. t-SNE를 이용해 고차원 벡터를 2차원으로 시각화  
3. 유클리드 거리 기반 유사 이미지/비유사 이미지 쌍 자동 탐색  
4. (옵션) 오토인코더(AutoEncoder)로 이상 이미지 감지  
5. Streamlit 웹 앱을 통해 대화형 시각화 제공


---


## ▶️ 실행 방법

파이썬 3.12.9 버전을 권장합니다.

### 1. 설치

```bash
pip install -r requirements.txt
```

### 2. 이미지 준비
- images/ 폴더(만드세요..)에 .jpg, .png 이미지 파일을 넣어주세요.
- 기본적으로는 급식 이미지를 예시로 사용하였지만, 어떤 이미지든 적용 가능합니다.


### 3. Streamlit 앱 실행

```bash
streamlit run app.py
```

