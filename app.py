import streamlit as st
import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.preprocessing import normalize


st.set_page_config(layout="wide")
st.title("급식 사진 유사/비유사 이미지 쌍 찾기")

import pathlib
image_dir = "images"

features_path = pathlib.Path("features.npy")
filenames_path = pathlib.Path("filenames.npy")

# 전처리(Feature 추출) 버튼 UI
with st.expander("전처리(Feature 추출) 실행", expanded=not (features_path.exists() and filenames_path.exists())):
    if st.button("전처리 및 Feature 추출 실행", help="이미지 폴더에서 features.npy, filenames.npy를 새로 생성합니다."):
        st.info("전처리 및 feature 추출을 실행합니다...")
        try:
            from extract_features import extract_and_save_features
            extract_and_save_features(image_dir=image_dir, features_path="features.npy", filenames_path="filenames.npy")
        except ImportError:
            import subprocess
            subprocess.run(["python", "extract_features.py"], check=True)
        st.success("전처리 및 feature 추출 완료! 새로고침 해주세요.")
        st.stop()

# 파일이 없으면 안내 메시지 및 실행 차단
if not (features_path.exists() and filenames_path.exists()):
    st.warning("features.npy, filenames.npy가 없습니다. 위의 버튼으로 전처리를 먼저 실행하세요.")
    st.stop()

features = np.load("features.npy")
features = normalize(features, norm='l2')
filenames = np.load("filenames.npy", allow_pickle=True)

# 썸네일 생성 함수
def get_thumbnail(path, size=(128, 128)):
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(size)
        return img
    except Exception:
        return Image.new("RGB", size, (255,255,255))


# 유클리드 유사도로 가장 비슷/다른 쌍 찾기 (upper triangle만 사용, 중복 완전 제거)
from scipy.spatial.distance import pdist, squareform
dist_matrix = squareform(pdist(features, metric="euclidean"))

# 거리 행렬의 upper triangle만 flatten해서 인덱스 추출
triu_indices = np.triu_indices_from(dist_matrix, k=1)
flat_dist = dist_matrix[triu_indices]

# 가장 가까운 쌍 2개
sim_idx = np.argsort(flat_dist)[:2]
similar_pairs = [(triu_indices[0][k], triu_indices[1][k]) for k in sim_idx]

# 가장 먼 쌍 2개
dissim_idx = np.argsort(flat_dist)[-2:]
dissimilar_pairs = [(triu_indices[0][k], triu_indices[1][k]) for k in dissim_idx]

# 산점도용 t-SNE 좌표 불러오기 (없으면 features의 첫 2차원 사용)
import plotly.express as px
from sklearn.manifold import TSNE

# t-SNE 결과 shape가 다르면 자동 재계산
tsne_path = "tsne_result.npy"
need_tsne = True
if os.path.exists(tsne_path):
    try:
        tsne_result = np.load(tsne_path)
        if tsne_result.shape[0] == features.shape[0]:
            coords = tsne_result
            need_tsne = False
    except Exception:
        pass
if need_tsne:
    st.info("t-SNE 결과를 새로 계산합니다. (이미지 개수에 맞게)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, (features.shape[0]-1)//3)))
    coords = tsne.fit_transform(features)
    np.save(tsne_path, coords)
    st.success("t-SNE 계산 및 저장 완료!")

df = pd.DataFrame(coords, columns=["x", "y"])
df["filename"] = filenames


# 산점도와 쌍 비교 이미지를 좌우로 배치
col1, col2 = st.columns([1.2, 1.8])
with col1:
    st.header("급식 사진 t-SNE 산점도 (원본 유사도 1위 쌍 강조)")
    # 원본 feature 공간에서 가장 유사한 쌍을 t-SNE 공간에서도 강조
    highlight_idx = set(similar_pairs[0])
    df["highlight"] = ["highlight" if idx in highlight_idx else "normal" for idx in range(len(df))]
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="highlight",
        hover_data=["filename"],
        color_discrete_map={"highlight": "red", "normal": "blue"},
        height=600,
        title="t-SNE 산점도 (가장 비슷한 쌍 1위만 빨간색)"
    )
    fig.update_traces(marker=dict(size=12, opacity=0.7))
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("가장 비슷한 급식 사진 TOP 2 쌍 (1위, 2위)")
    for rank, (i, j) in enumerate(similar_pairs, 1):
        st.markdown(f"**유사도 {rank}위**")
        c1, c2 = st.columns(2)
        with c1:
            st.image(get_thumbnail(os.path.join(image_dir, filenames[i])), caption=f"{filenames[i]}")
        with c2:
            st.image(get_thumbnail(os.path.join(image_dir, filenames[j])), caption=f"{filenames[j]}")
        st.write(f"유사도(거리): {dist_matrix[i, j]:.4f}")

    st.header("가장 다른 급식 사진 TOP 2 쌍 (1위, 2위)")
    for rank, (i, j) in enumerate(dissimilar_pairs, 1):
        st.markdown(f"**비유사도 {rank}위**")
        c1, c2 = st.columns(2)
        with c1:
            st.image(get_thumbnail(os.path.join(image_dir, filenames[i])), caption=f"{filenames[i]}")
        with c2:
            st.image(get_thumbnail(os.path.join(image_dir, filenames[j])), caption=f"{filenames[j]}")
        st.write(f"비유사도(거리): {dist_matrix[i, j]:.4f}")