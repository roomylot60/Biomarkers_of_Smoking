import pandas as pd
from models import SmokingPredictionModels
import os

base_dir = os.getcwd()

# 데이터 로드
x_train = pd.read_csv(os.path.join(base_dir,"data/Smoking_raw/competition_format/x_train.csv")).drop(columns=["ID"], errors="ignore")
x_test = pd.read_csv(os.path.join(base_dir,"data/Smoking_raw/competition_format/x_test.csv")).drop(columns=["ID"], errors="ignore")
y_train = pd.read_csv(os.path.join(base_dir,"data/Smoking_raw/competition_format/y_train.csv"))
y_test = pd.read_csv(os.path.join(base_dir,"data/Smoking_raw/competition_format/y_test.csv"))

# 모델 객체 생성 (최적화 옵션 활성화)
model_manager = SmokingPredictionModels(optimize=True)

# 데이터 전처리
x_train_scaled, x_test_scaled = model_manager.preprocess_data(x_train, x_test)

# 모델 학습
model_manager.train_models(x_train_scaled, y_train)

# 모델 평가
results = model_manager.evaluate_models(x_test_scaled, y_test)

# 결과 출력
print("\n 모델별 평가 결과:")
for model, scores in results.items():
    print(f"\n {model} 성능 평가:")
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")