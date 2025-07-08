from ezc3d import c3d
import numpy as np


def check_nan_inf_in_c3d(path):
    c = c3d(path)
    point_data = c["data"]["points"]  # shape: (4, num_markers, num_frames)
    data = point_data[0:3, :, :]  # X, Y, Z only

    contains_nan = np.isnan(data).any()
    contains_inf = np.isinf(data).any()

    print(f"ファイル: {path}")
    if contains_nan:
        print("⚠️ NaN を含んでいます")
    else:
        print("✅ NaN は含まれていません")

    if contains_inf:
        print("⚠️ inf を含んでいます")
    else:
        print("✅ inf は含まれていません")

    # どこに含まれているか位置を確認（任意）
    if contains_nan or contains_inf:
        nan_locs = np.argwhere(np.isnan(data))
        inf_locs = np.argwhere(np.isinf(data))
        if len(nan_locs) > 0:
            print(f"NaNの位置（[軸, マーカー, フレーム]）: {nan_locs}")
        if len(inf_locs) > 0:
            print(f"infの位置（[軸, マーカー, フレーム]）: {inf_locs}")


path = "../dataset/000004_000103_75_236_R_003_972.c3d"
check_nan_inf_in_c3d(path)
