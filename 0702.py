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

