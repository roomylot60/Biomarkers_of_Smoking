import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

def preprocessing(dataframe):
    # 불필요한 ID 컬럼 제거 및 Oral(=구강검사 여부) 특성값은 모두 Y 값이므로 삭제.
    dataframe = dataframe.drop(columns=["ID","oral"], errors="ignore")

    # 2개의 카테고리를 가지는 특성값을 0, 1로 변환
    dataframe["gender"] = dataframe["gender"].replace({"M": 1, "F": 0}).infer_objects(copy=False)
    dataframe["tartar"] = dataframe["tartar"].replace({"Y": 1, "N": 0}).infer_objects(copy=False)
    
    # Urine protein categorizing
    dataframe['Urine protein'] = dataframe['Urine protein'].replace({1.0: 0, 2.0: 1})
    
    # hearing feature converting values 1, 2 => 1, 0
    dataframe['hearing(left)'] = dataframe['hearing(left)'].replace({2.0: 0})
    dataframe['hearing(right)'] = dataframe['hearing(right)'].replace({2.0: 0})

    # BMI 지수 계산 : bmi = kg/m^2
    dataframe['bmi'] = dataframe['weight(kg)']/((dataframe['height(cm)']*0.01)**2)
    # wwi(비만 지수) 지수 계산 : wwi = cm/sqrt(kg)
    dataframe['wwi'] = dataframe['waist(cm)']/ np.sqrt(dataframe['weight(kg)'])

    return dataframe

def scaling(train_data, test_data, scaler="MinMaxScaler"):
    """
    Scaler를 이용하여 데이터를 변환하는 함수
    scaler: StandardScaler(), RobustScaler(), MinMaxScaler(), logScaler 가능
    """
    train_data = preprocessing(train_data)
    test_data = preprocessing(test_data)

    # 카테고리형 변수 분리
    categorical_features = ['gender', 'tartar', 'hearing(right)', 'hearing(left)', 'dental caries']
    train_cat = train_data[categorical_features]
    test_cat = test_data[categorical_features]
    
    train_num = train_data.drop(columns=categorical_features)
    test_num = test_data.drop(columns=categorical_features)

    # 문자열 옵션 또는 직접 scaler 객체 입력 가능하도록 처리
    scaler_dict = {
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler(),
        "MinMaxScaler": MinMaxScaler()
    }
    if isinstance(scaler, str):
        if scaler in scaler_dict:
            scaler = scaler_dict[scaler]
        elif scaler == "logScaler":
            # log 변환 적용
            train_scaled = np.log1p(train_num)
            test_scaled = np.log1p(test_num)
        else:
            raise ValueError(f"지원되지 않는 Scaler 옵션입니다: {scaler}")
    elif not hasattr(scaler, "fit") or not hasattr(scaler, "transform"):
        raise ValueError(f"올바른 Scaler 객체가 아닙니다: {scaler}")

    # Scaler 적용
    if scaler != "logScaler":
        scaler.fit(train_num)
        train_scaled = pd.DataFrame(scaler.transform(train_num), columns=train_num.columns)
        test_scaled = pd.DataFrame(scaler.transform(test_num), columns=test_num.columns)

    # 카테고리형 데이터 추가
    train_scaled[categorical_features] = train_cat.reset_index(drop=True)
    test_scaled[categorical_features] = test_cat.reset_index(drop=True)

    return train_scaled, test_scaled

if __name__ == "__main__":
    base_path = os.getcwd()
    data_path = os.path.join(base_path, "data/Smoking_raw/smoking.csv")
    # Load dataset
    smoking = pd.read_csv(data_path)
    df = preprocessing(smoking)
    pd.set_option('display.max_columns',30)
    print(df.head())
    
