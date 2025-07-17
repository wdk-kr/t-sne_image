import os
import numpy as np
import cv2
import time
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
from tensorflow.keras.models import Model # type: ignore

def load_images_from_folder(folder):
    images = []
    filenames = []
    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(fpath):
            try:
                img = cv2.imread(fpath)
                if img is not None:
                    images.append(img)
                    filenames.append(fname)
            except Exception:
                pass
    return images, np.array(filenames)

def extract_and_save_features(image_dir="images", features_path="features.npy", filenames_path="filenames.npy"):
    # 사전 학습된 ResNet50 모델 로딩
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)

    images, filenames = load_images_from_folder(image_dir)
    print(f"{len(images)}개의 이미지를 불러왔습니다.")

    features = []
    start = time.time()
    for img in images:
        # BGR to RGB 변환 (중요!)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = cv2.resize(img_rgb, (224, 224))
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x, verbose=0)
        features.append(feature.flatten())

    features = np.array(features)
    print("벡터 추출 소요 시간:", round(time.time() - start, 2), "초")
    print(f"{len(features)}개 이미지의 feature 벡터 shape:", features.shape)

    np.save(features_path, features)
    np.save(filenames_path, filenames)
    print(f"{features_path} / {filenames_path} 저장 완료")

if __name__ == "__main__":
    extract_and_save_features()