

import numpy as np
import cv2
from scipy.signal.windows import hann
import matplotlib.pyplot as plt

def create_hanning_window(size):
    """2D Hanning window を生成"""
    hann_y = hann(size[0], sym=False)
    hann_x = hann(size[1], sym=False)
    window = np.outer(hann_y, hann_x)
    return window.astype(np.float32)

def phase_correlation_like_opencv(img1, img2):
    """OpenCVのphaseCorrelateと同等の処理"""
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    h, w = img1.shape
    win = create_hanning_window((h, w))
    img1_win = img1 * win
    img2_win = img2 * win

    H = cv2.getOptimalDFTSize(h)
    W = cv2.getOptimalDFTSize(w)
    padded1 = np.zeros((H, W), dtype=np.float32)
    padded2 = np.zeros((H, W), dtype=np.float32)
    padded1[:h, :w] = img1_win
    padded2[:h, :w] = img2_win

    dft1 = cv2.dft(padded1, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft2 = cv2.dft(padded2, flags=cv2.DFT_COMPLEX_OUTPUT)

    # conj(dft2)をかける（複素共役）
    conj_dft2 = dft2.copy()
    conj_dft2[:, :, 1] *= -1

    numerator = cv2.mulSpectrums(dft1, conj_dft2, 0)
    mag = cv2.magnitude(numerator[:, :, 0], numerator[:, :, 1])
    mag[mag == 0] = 1e-10  # ゼロ除算防止

    cross_power_spectrum = numerator / mag[:, :, np.newaxis]

    # 相関マップ（逆FFT）
    corr = cv2.idft(cross_power_spectrum, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    corr = np.fft.fftshift(corr)

    # ピーク位置を検出
    _, _, _, peak_loc = cv2.minMaxLoc(corr)
    peak_y, peak_x = peak_loc[1], peak_loc[0]

    # レスポンス計算（ピーク周囲5x5）
    window_size = 5
    half_win = window_size // 2
    minr = max(0, peak_y - half_win)
    maxr = min(H - 1, peak_y + half_win)
    minc = max(0, peak_x - half_win)
    maxc = min(W - 1, peak_x + half_win)
    response = np.sum(corr[minr:maxr+1, minc:maxc+1]) / (H * W)

    # 中心からのずれとしてシフト量を算出
    center = np.array([W / 2, H / 2])
    shift = center - np.array([peak_x, peak_y])

    # グラフ表示
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(corr, cmap='hot')
    ax.set_title(f"Correlation Map\nPeak at ({peak_x}, {peak_y}), Response: {response:.5f}")
    ax.plot(peak_x, peak_y, 'bo')
    plt.tight_layout()
    plt.show()

    return shift, response

# テスト用：円をずらして比較
if __name__ == "__main__":
    img_base = np.zeros((128, 128), dtype=np.float32)
    cv2.circle(img_base, (64, 64), 10, 1, -1)
    img_shifted = np.roll(img_base, shift=(5, -3), axis=(0, 1))

    shift, response = phase_correlation_like_opencv(img_base, img_shifted)
    print("Detected shift:", shift)
    print("Response:", response)

def compute_response_like_opencv(corr, peak_loc):
    """
    OpenCVのphaseCorrelateのようなresponseを再現する
    """
    h, w = corr.shape
    M, N = h, w
    win_size = 5
    half_win = win_size // 2

    y, x = peak_loc
    minr = max(0, y - half_win)
    maxr = min(h - 1, y + half_win)
    minc = max(0, x - half_win)
    maxc = min(w - 1, x + half_win)

    region = corr[minr:maxr+1, minc:maxc+1]
    response = np.sum(region) / (M * N)

    return response


def mask_corr_near_shift(corr, prev_shift, max_distance=10):
    """
    corr の中で、prev_shift から max_distance 以内の部分だけ残す

    Parameters:
        corr: np.ndarray
        prev_shift: tuple of float (dx, dy)
        max_distance: float, 距離しきい値（ピクセル）

    Returns:
        masked_corr: np.ndarray（同サイズ、マスク以外ゼロ）
    """
    h, w = corr.shape
    dx, dy = prev_shift
    # wrap-around を考慮して、位置を整数でマッピング
    cx = int(round((dx + w) % w))
    cy = int(round((dy + h) % h))

    Y, X = np.ogrid[:h, :w]
    dist = np.hypot(X - cx, Y - cy)

    mask = dist <= max_distance
    masked_corr = np.zeros_like(corr)
    masked_corr[mask] = corr[mask]
    return masked_corr


import numpy as np
from scipy.ndimage import maximum_filter

def find_peak_candidates(
    corr,
    threshold=0.2,
    min_distance=3,
    top_k=None,
    sort_by="response"
):
    """
    相関マップからピーク候補を抽出

    Parameters:
        corr: np.ndarray 相関マップ（float32）
        threshold: float ピーク強度のしきい値（0～1）
        min_distance: int 最小距離（近接ピークを除外）
        top_k: int or None 上位k個に制限（Noneなら制限なし）
        sort_by: str 'response' または 'distance_to_center'

    Returns:
        List of (y, x) tuples
    """

    h, w = corr.shape
    center_y, center_x = h // 2, w // 2

    # 局所最大検出
    local_max = (corr == maximum_filter(corr, size=3))
    threshold_mask = (corr > threshold)
    peak_mask = local_max & threshold_mask

    candidates = np.argwhere(peak_mask)
    if candidates.size == 0:
        return []

    # スコア計算
    scored = []
    for y, x in candidates:
        val = corr[y, x]
        dist_to_center = np.hypot(x - center_x, y - center_y)
        scored.append((val, dist_to_center, y, x))

    # 並べ替え
    if sort_by == "response":
        scored.sort(key=lambda t: -t[0])  # 相関値降順
    elif sort_by == "distance_to_center":
        scored.sort(key=lambda t: t[1])   # 中心に近い順

    # 最小距離フィルタ
    selected = []
    for _, _, y, x in scored:
        if all(np.hypot(x - px, y - py) >= min_distance for py, px in selected):
            selected.append((y, x))
        if top_k and len(selected) >= top_k:
            break

    return selected

##上下・左右対称のピークをペアで除外
def filter_symmetric_peaks(peaks, corr, tolerance=5):
    h, w = corr.shape
    kept = []

    for y, x in peaks:
        mirror_y = (h - y) % h
        mirror_x = (w - x) % w
        val = corr[y, x]
        mirror_val = corr[mirror_y, mirror_x]

        if abs(val - mirror_val) > 0.05:  # 似すぎていたら対称とみなす
            kept.append((y, x))

    return kept
    
    

###corr の中心から遠いピークを除外する
def filter_symmetric_peaks(peaks, corr, tolerance=5):
    h, w = corr.shape
    kept = []

    for y, x in peaks:
        mirror_y = (h - y) % h
        mirror_x = (w - x) % w
        val = corr[y, x]
        mirror_val = corr[mirror_y, mirror_x]

        if abs(val - mirror_val) > 0.05:  # 似すぎていたら対称とみなす
            kept.append((y, x))

    return kept

def filter_by_distance_to_center(peaks, corr_shape, max_distance_ratio=0.5):
    h, w = corr_shape
    cx, cy = w / 2, h / 2
    max_dist = np.hypot(w * max_distance_ratio, h * max_distance_ratio)
    return [(y, x) for y, x in peaks if np.hypot(x - cx, y - cy) <= max_dist]

def filter_symmetric_peaks(peaks, corr, tolerance=0.05):
    h, w = corr.shape
    kept = []
    for y, x in peaks:
        mirror_y = (h - y) % h
        mirror_x = (w - x) % w
        if abs(corr[y, x] - corr[mirror_y, mirror_x]) > tolerance:
            kept.append((y, x))
    return kept
    
    

##切り替えつき
def phase_correlation_with_subpixel_with_tracking(
    img1, img2, prev_shift=None, peak_threshold=0.2, use_hanning=True,
    peak_filter_mode="nearest_to_prev", max_dist_ratio=0.5
):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    if use_hanning:
        win = cv2.createHanningWindow(img1.shape[::-1], cv2.CV_32F)
        img1 *= win
        img2 *= win

    dft1 = cv2.dft(img1, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft2 = cv2.dft(img2, flags=cv2.DFT_COMPLEX_OUTPUT)
    cs = cv2.mulSpectrums(dft1, dft2, flags=0, conjB=True)

    mag = cv2.magnitude(cs[..., 0], cs[..., 1])
    mag[mag == 0] = 1e-6
    cs[..., 0] /= mag
    cs[..., 1] /= mag

    corr = cv2.idft(cs, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    h, w = corr.shape
    peaks = find_peak_candidates(corr, threshold=peak_threshold)

    # 🔁 ピークフィルタリング処理（モードに応じて）
    if peak_filter_mode == "distance_to_center":
        peaks = filter_by_distance_to_center(peaks, corr.shape, max_distance_ratio=max_dist_ratio)
    elif peak_filter_mode == "symmetric":
        peaks = filter_symmetric_peaks(peaks, corr)
    # "nearest_to_prev" は何もしない（後で prev_shift 使って選択）

    # 📌 ピーク選択
    if len(peaks) == 0:
        return (0, 0), 0.0, corr

    if peak_filter_mode == "nearest_to_prev" and prev_shift is not None:
        best_yx, _ = select_nearest_peak(peaks, corr, prev_shift, w, h)
    else:
        best_yx = max(peaks, key=lambda p: corr[p[0], p[1]])  # 最も相関が高い点

    # 🎯 サブピクセル補間
    y_sub, x_sub = subpixel_peak_2d(corr, best_yx[1], best_yx[0])
    shift = correct_shift(x_sub, y_sub, w, h)
    response = compute_response(corr, shift, w, h)

    return shift, response, corr




import numpy as np
import cv2
from scipy.ndimage import maximum_filter

def phase_correlation_with_subpixel_with_tracking(img1, img2, prev_shift=None, peak_threshold=0.2, use_hanning=True):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    if use_hanning:
        win = cv2.createHanningWindow(img1.shape[::-1], cv2.CV_32F)
        img1 *= win
        img2 *= win

    # FFT & cross power spectrum
    dft1 = cv2.dft(img1, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft2 = cv2.dft(img2, flags=cv2.DFT_COMPLEX_OUTPUT)
    cs = cv2.mulSpectrums(dft1, dft2, flags=0, conjB=True)

    mag = cv2.magnitude(cs[..., 0], cs[..., 1])
    mag[mag == 0] = 1e-6  # avoid divide-by-zero
    cs[..., 0] /= mag
    cs[..., 1] /= mag

    # Inverse DFT to get phase correlation map
    corr = cv2.idft(cs, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    h, w = corr.shape
    peaks = find_peak_candidates(corr, threshold=peak_threshold)

    # Choose the best peak
    if prev_shift is not None and len(peaks) > 0:
        best_yx, _ = select_nearest_peak(peaks, corr, prev_shift, w, h)
    else:
        _, _, _, max_loc = cv2.minMaxLoc(corr)
        best_yx = (max_loc[1], max_loc[0])  # (y, x)

    # --- サブピクセル補間 ---
    y_sub, x_sub = subpixel_peak_2d(corr, best_yx[1], best_yx[0])

    # --- Wrap-around 補正 ---
    shift = correct_shift(x_sub, y_sub, w, h)

    # 相関強度（正規化済み）
    response = compute_response(corr, shift, w, h)

    return shift, response, corr
    
def find_peak_candidates(corr, threshold=0.2):
    local_max = (corr == maximum_filter(corr, size=3))
    return np.argwhere(local_max & (corr > threshold))

def correct_shift(x, y, width, height):
    dx = x if x <= width / 2 else x - width
    dy = y if y <= height / 2 else y - height
    return dx, dy

def select_nearest_peak(peaks, corr, prev_shift, width, height):
    best_peak = None
    min_distance = float('inf')
    for y, x in peaks:
        dx, dy = correct_shift(x, y, width, height)
        dist = np.hypot(dx - prev_shift[0], dy - prev_shift[1])
        if dist < min_distance:
            min_distance = dist
            best_peak = (y, x)
    return best_peak, min_distance

def subpixel_peak_2d(corr, x, y):
    # 2次補間でサブピクセル補正（x, y: int）
    if not (1 <= x < corr.shape[1] - 1 and 1 <= y < corr.shape[0] - 1):
        return y, x  # 端はそのまま

    dx = (corr[y, x+1] - corr[y, x-1]) / 2
    dxx = corr[y, x+1] - 2 * corr[y, x] + corr[y, x-1]

    dy = (corr[y+1, x] - corr[y-1, x]) / 2
    dyy = corr[y+1, x] - 2 * corr[y, x] + corr[y-1, x]

    sub_x = x - dx / dxx if dxx != 0 else x
    sub_y = y - dy / dyy if dyy != 0 else y

    return sub_y, sub_x  # 注意: (y, x) で返す

def compute_response(corr, shift, width, height):
    x = int(round(shift[0])) % width
    y = int(round(shift[1])) % height
    peak_val = corr[y, x]
    norm = np.sqrt(np.sum(corr ** 2))
    return float(peak_val) / norm if norm != 0 else 0.0
    
    
prev_shift = (0.0, 0.0)  # 初回
shift, response, corr = phase_correlation_with_subpixel_with_tracking(img1, img2, prev_shift)
print(f"サブピクセル精度のシフト: {shift}, 相関強度: {response:.4f}")
prev_shift = shift


=+=============








import numpy as np
import cv2
from scipy.ndimage import maximum_filter

def phase_correlation_with_subpixel_with_tracking(img1, img2, prev_shift=None, peak_threshold=0.2, use_hanning=True):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    if use_hanning:
        win = cv2.createHanningWindow(img1.shape[::-1], cv2.CV_32F)
        img1 *= win
        img2 *= win

    dft1 = cv2.dft(img1, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft2 = cv2.dft(img2, flags=cv2.DFT_COMPLEX_OUTPUT)

    cs = cv2.mulSpectrums(dft1, dft2, flags=0, conjB=True)
    mag = cv2.magnitude(cs[..., 0], cs[..., 1])
    mag[mag == 0] = 1e-6
    cs[..., 0] /= mag
    cs[..., 1] /= mag

    corr = cv2.idft(cs, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    h, w = corr.shape
    peaks = find_peak_candidates(corr, threshold=peak_threshold)

    if prev_shift is not None and len(peaks) > 0:
        best_shift, _ = select_nearest_peak(peaks, corr, prev_shift, w, h)
    else:
        _, _, _, max_loc = cv2.minMaxLoc(corr)
        x0, y0 = max_loc
        best_shift = correct_shift(x0, y0, w, h)

    response = compute_response(corr, best_shift, w, h)

    return best_shift, response, corr


def find_peak_candidates(corr, threshold=0.2):
    local_max = (corr == maximum_filter(corr, size=3))
    return np.argwhere(local_max & (corr > threshold))


def correct_shift(x, y, width, height):
    dx = x if x <= width / 2 else x - width
    dy = y if y <= height / 2 else y - height
    return dx, dy


def select_nearest_peak(peaks, corr, prev_shift, width, height):
    best_peak = None
    min_distance = float('inf')

    for y, x in peaks:
        dx, dy = correct_shift(x, y, width, height)
        dist = np.hypot(dx - prev_shift[0], dy - prev_shift[1])
        if dist < min_distance:
            min_distance = dist
            best_peak = (dx, dy)

    return best_peak, min_distance


def compute_response(corr, shift, width, height):
    x = int(round(shift[0])) % width
    y = int(round(shift[1])) % height
    peak_val = corr[y, x]
    norm = np.sqrt(np.sum(corr ** 2))
    return float(peak_val) / norm if norm != 0 else 0.0
    
使用方法

prev_shift = (0.0, 0.0)  # 初回は (0, 0) または None
shift, response, corr = phase_correlation_with_subpixel_with_tracking(img1, img2, prev_shift)
print("シフト:", shift, "信頼度:", response)
prev_shift = shift  # 次回の比較に使う

