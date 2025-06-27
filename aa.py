def auto_canny(img, sigma=0.33):
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lower, upper)
    
    
    
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_line_angles(img, canny_thresh1=50, canny_thresh2=150, hough_thresh=100):
    # グレースケール化（カラー対応）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # エッジ検出
    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2)

    # ハフ変換で直線検出
    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_thresh)

    if lines is None:
        print("線が検出されませんでした。")
        return None

    # 角度を取得（-90～+90度に正規化）
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.rad2deg(theta)
        if angle > 90:
            angle -= 180  # -90〜+90 の範囲に変換
        angles.append(angle)

    # 結果表示
    plt.hist(angles, bins=180, range=(-90, 90))
    plt.title("Detected Line Angles")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

    # 最頻値を回転角とする
    hist, bins = np.histogram(angles, bins=180, range=(-90, 90))
    peak_angle = (bins[np.argmax(hist)] + bins[np.argmax(hist)+1]) / 2

    print(f"推定された回転角（最頻値）: {peak_angle:.2f}°")
    return peak_angle
    
    import cv2
import numpy as np

def estimate_rotation_ecc(img1, img2, warp_mode=cv2.MOTION_AFFINE):
    # グレースケール変換
    if img1.ndim == 3: img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3: img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1 = np.float32(img1)
    img2 = np.float32(img2)

    # 初期の変換行列
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # 停止条件
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)

    # ECCによる整列
    cc, warp_matrix = cv2.findTransformECC(img1, img2, warp_matrix, warp_mode, criteria)

    # アフィンなら回転角を抽出
    if warp_mode == cv2.MOTION_AFFINE:
        dx = warp_matrix[0, 0]
        dy = warp_matrix[1, 0]
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.rad2deg(angle_rad)
        print(f"回転角（ECC推定）: {angle_deg:.3f}°")
        return angle_deg
    else:
        print("射影変換では回転角は直接得られません")
        return None