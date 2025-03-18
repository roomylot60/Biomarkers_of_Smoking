import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
import os
import optuna
from preprocessing import preprocessing, scaling  # 전처리 모듈 추가

class SmokingPredictionModels:
    def __init__(self, scaler="MinMaxScaler", optimize=False):
        self.scaler = scaler
        self.optimize = optimize  # 최적화 여부
        self.models = {
            "LightGBM": lgb.LGBMClassifier(n_estimators=1000, max_depth=12, learning_rate=0.01, random_state=42),
            "XGBoost": xgb.XGBClassifier(n_estimators=1000, max_depth=12, learning_rate=0.01, random_state=42),
            "SVM": SVC(kernel='rbf', probability=True, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=500, max_depth=12, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
        }
        self.ensemble_model = None  # 앙상블 모델

    def get_hparam_path(self, model_name):
        return f"./saved_models/{model_name}_hparam.json"

    def save_best_params(self, model_name, params):
        """ 최적의 하이퍼파라미터를 JSON 파일에 저장 """
        os.makedirs("./saved_models/", exist_ok=True)
        with open(self.get_hparam_path(model_name), "w") as f:
            json.dump(params, f, indent=4)
        print(f"✅ {model_name} 최적의 하이퍼파라미터 저장 완료!")

    def load_best_params(self, model_name):
        """ JSON 파일에서 최적의 하이퍼파라미터 불러오기 """
        path = self.get_hparam_path(model_name)
        if os.path.exists(path):
            with open(path, "r") as f:
                params = json.load(f)
            print(f"✅ 저장된 {model_name} 하이퍼파라미터 불러오기 완료!")
            return params
        return None

    def optimize_model(self, model_name, x_train, y_train):
        """ Optuna를 사용하여 모든 모델의 하이퍼파라미터 최적화 """
        param_spaces = {
            "LightGBM": {
                'num_leaves': (10, 50),
                'max_depth': (5, 20),
                'learning_rate': (0.01, 0.1),
                'n_estimators': (500, 1000),
                'boosting_type': ['gbdt'],
                'num_threads': [4],
                'max_bin': [255]
            },
            "XGBoost": {
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.1),
                'n_estimators': (500, 1500),
                'tree_method': ['hist']
            },
            "SVM": {
                'C': (0.1, 10),
                'gamma': (0.001, 1)
            },
            "RandomForest": {
                'n_estimators': (100, 1000),
                'max_depth': (5, 20)
            },
            "LogisticRegression": {
                'C': (0.1, 10)
            }
        }

        if model_name not in param_spaces:
            return None
        
        def objective(trial):
            try:
                param = {}
                for key, value in param_spaces[model_name].items():
                    if isinstance(value, tuple) and len(value) == 2:
                        param[key] = trial.suggest_int(key, *value) if isinstance(value[0], int) else trial.suggest_float(key, *value, log=True)
                    elif isinstance(value, list):
                        param[key] = trial.suggest_categorical(key, value)
                
                # LightGBM 예외 처리 추가
                if model_name == "LightGBM":
                    param["num_leaves"] = min(param["num_leaves"], 2 ** param["max_depth"])
                    param.pop("tree_method", None)  # LightGBM에서는 tree_method가 필요 없음
                
                print(f"🔹 {model_name} 학습 시작 (파라미터: {param})")
                model = self.models[model_name].__class__(**param, random_state=42)
                model.fit(x_train, y_train.values.ravel())
                return f1_score(y_train, model.predict(x_train))
            except Exception as e:
                print(f"❌ {model_name} 최적화 중 오류 발생: {e}")
                return float(0.0)
        
        best_params = self.load_best_params(model_name)
        if best_params:
            return best_params
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10, timeout=300)
        best_params = study.best_params
        self.save_best_params(model_name, best_params)
        return best_params

    def preprocess_data(self, x_train, x_test):
        """ 전처리 수행 """
        x_train, x_test = preprocessing(x_train), preprocessing(x_test)
        x_train_scaled, x_test_scaled = scaling(x_train, x_test, scaler=self.scaler)
        return x_train_scaled, x_test_scaled

    def train_models(self, x_train, y_train):
        """ 모든 모델 학습 + 최적화 옵션 적용 """
        for name, model in self.models.items():
            if self.optimize:
                print(f"🔍 {name} 최적화 중...")
                best_params = self.optimize_model(name, x_train, y_train)
                if best_params:
                    self.models[name] = self.models[name].__class__(**best_params, random_state=42)

            print(f"Training {name}...")

            # LightGBM 학습 시 메모리 절약 옵션 적용
            if name == "LightGBM":
                print(f"⚡ LightGBM 학습 시 메모리 절약 모드 활성화")
                model.set_params(device="cpu", max_bin=255, num_leaves=31, max_depth=10)

                # 배치 학습 적용
                batch_size = min(len(x_train) // 10, 1000)  # 배치 크기 설정
                for i in range(0, len(x_train), batch_size):
                    x_batch = x_train.iloc[i:min(i + batch_size, len(x_train))]  # x_train 범위 조정
                    y_batch = y_train.iloc[i:min(i + batch_size, len(y_train))]  # y_train 범위 조정
                    
                    print(f"🛠 LightGBM 배치 학습 ({i} ~ {i+batch_size}), 실제 배치 크기: {len(x_batch)}, {len(y_batch)}")
                    
                    if len(x_batch) != len(y_batch):  # 데이터 크기 불일치 방지
                        print(f"❌ 데이터 크기 불일치: x_batch={len(x_batch)}, y_batch={len(y_batch)} → 학습 건너뜀")
                        continue
                    
                    model.fit(x_batch, y_batch.values.ravel(), init_model=model if i > 0 else None)
            else:
                try:
                    model.fit(x_train, y_train.values.ravel())
                except Exception as e:
                    import traceback
                    print(f"❌ {name} 학습 중 오류 발생: {e}")
                    traceback.print_exc()
                    continue

    def evaluate_models(self, x_test, y_test):
        """ 모든 모델 평가 """
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(x_test)
            results[name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "ROC AUC": roc_auc_score(y_test, y_pred),
            }
        return results