# utils/model/model.py

import os
import pickle
import numpy as np
import pandas as pd
from skfda import FDataGrid
from fts_data_utils import pad_with_last_row

def infer_new_data(model_path: str, input_dir, fixed_rows: int = 300):
    """
    Load a saved Rocket+FDA model and predict on CSV segments.
    
    Parameters:
      - model_path: path to rocket_fda_model.pkl
      - input_dir: folder containing .csv segment files
      - fixed_rows: time‐series length for padding & smoothing
    
    Returns:
      List of (filename, predicted_label)
    """
    # 1) 모델 번들 로드
    with open(model_path, "rb") as f:
        model_bundle = pickle.load(f)
    rocket     = model_bundle["rocket"]
    scaler     = model_bundle["scaler"]
    classifier = model_bundle["classifier"]
    smoother   = model_bundle["smoother"]
    grid       = model_bundle["grid"]
    
    results = []
    # 2) 모든 CSV 파일 순회
    for fn in os.listdir(input_dir):
        if not fn.endswith(".csv"):
            continue
        path = os.path.join(input_dir, fn)
        df   = pd.read_csv(path)
        
        # 3) 원본 시계열 (T × F) 추출 & 패딩
        ts = df.iloc[:, 3:].values               # shape: (T, F)
        ts = pad_with_last_row(ts, fixed_rows)    # shape: (fixed_rows, F)
        
        # 4) 훈련 때와 동일한 형태로 3D 배열 만들기
        #    (n_samples=1, timepoints, features)
        ts_3d = ts[np.newaxis, :, :]             # shape: (1, T, F)
        
        # 5) FDataGrid → 스무딩
        fd = FDataGrid(data_matrix=ts_3d, grid_points=[grid])  # multivariate
        fd = smoother.fit_transform(fd)                        # still shape (1, T, F)
        
        # 6) ROCKET 입력 형식: (n_samples, channels, timepoints)
        X = fd.data_matrix                        # shape: (1, T, F)
        X = np.transpose(X, (0, 1, 2))            # shape: (1, F, T)
        
        # 7) 변환 → 정규화 → 예측
        Z        = rocket.transform(X)           # (1, K)
        Z_scaled = scaler.transform(Z)           # (1, K)
        pred     = classifier.predict(Z_scaled)[0]
        
        results.append((fn, pred))
    
    return results

# backward‐compatible alias
inference = infer_new_data


# if __name__ == "__main__":
#     # Example usage
#     model_path = "lateralraise_fin.pkl"
#     input_dir = "peak_detection_results/segments2"
#     results = infer_new_data(model_path, input_dir, fixed_rows=11, apply_smoothing=False)
#     label_map = {
#         '377': '올바른 사이드 레터럴 레이즈 자세',
#         '378': '무릎 반동 있는 자세',
#         '379': '어깨 으쓱 자세',
#         '380': '상완/전완 각도 불량 자세',
#         '381': '손목 각도 불량 자세',
#         '382': '상체 반동 자세'
#     }
#     for fname, label in results:
#         print(f"{fname} -> {label} ({label_map.get(label)})")
