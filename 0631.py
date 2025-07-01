def compute_response(corr, peak_pos):
    y, x = int(round(peak_pos[1])), int(round(peak_pos[0]))
    peak_val = corr[y, x]
    energy = np.sqrt(np.sum(corr ** 2))
    if energy == 0:
        return 0.0
    return float(peak_val) / energy
    
peak_subpixel = subpixel_peak_2d(corr)
response = compute_response(corr, peak_subpixel)


def subpixel_peak_2d(corr):
    _, _, max_loc, _ = cv2.minMaxLoc(corr)
    x0, y0 = max_loc

    if not (1 <= x0 < corr.shape[1] - 1 and 1 <= y0 < corr.shape[0] - 1):
        return (float(x0), float(y0))  # 周囲が取れない場合は整数返す

    # 3x3 近傍を取り出す
    patch = corr[y0-1:y0+2, x0-1:x0+2]

    # フィッティングのための座標と値
    dx = np.array([-1, 0, 1])
    dy = np.array([-1, 0, 1])
    X, Y = np.meshgrid(dx, dy)
    Z = patch.flatten()

    A = np.stack([
        np.ones_like(X).flatten(),
        X.flatten(),
        Y.flatten(),
        X.flatten()**2,
        X.flatten()*Y.flatten(),
        Y.flatten()**2
    ], axis=1)

    # 最小二乗で係数を求める
    coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)

    # 2次関数の極値（頂点）を求める
    a, b, c, d, e, f = coeffs
    denom = 4*d*f - e**2
    if denom == 0:
        return (float(x0), float(y0))

    x_peak = (e*c - 2*f*b) / denom
    y_peak = (e*b - 2*d*c) / denom

    return (x0 + x_peak, y0 + y_peak)



import numpy as np

def apply_bandpass_filter(dft, low=10, high=100):
    h, w = dft.shape[:2]
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)

    mask = np.logical_and(dist > low, dist < high).astype(np.float32)
    mask = np.repeat(mask[:, :, np.newaxis], 2, axis=2)  # 2ch (real, imag)

    return dft * mask

import cv2

# グレースケール画像
img1_f = img1.astype(np.float32)
img2_f = img2.astype(np.float32)

# ハニング窓でエッジを抑制（任意）
win = cv2.createHanningWindow(img1.shape[::-1], cv2.CV_32F)
img1_f *= win
img2_f *= win

# FFT（2ch: real + imag）
dft1 = cv2.dft(img1_f, flags=cv2.DFT_COMPLEX_OUTPUT)
dft2 = cv2.dft(img2_f, flags=cv2.DFT_COMPLEX_OUTPUT)

# バンドパスフィルタ適用
dft1_filt = apply_bandpass_filter(dft1, low=10, high=120)
dft2_filt = apply_bandpass_filter(dft2, low=10, high=120)

# クロスパワースペクトルの正規化
num = cv2.mulSpectrums(dft1_filt, dft2_filt, flags=0, conjB=True)
mag = cv2.magnitude(num[..., 0], num[..., 1])
mag[mag == 0] = 1e-6  # divide by zero防止
num[..., 0] /= mag
num[..., 1] /= mag

# 逆FFT → 相関マップ
corr = cv2.idft(num, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

# ピーク検出（ズレ量）
_, _, _, max_loc = cv2.minMaxLoc(corr)
shift = [float(max_loc[0]), float(max_loc[1])]

# wrap-around補正
h, w = corr.shape
if shift[0] > w / 2: shift[0] -= w
if shift[1] > h / 2: shift[1] -= h

print(f"推定シフト: {shift}")

low=10, high=60
柔らかい特徴だけ、細かい模様を除外（ノイズ除去）
low=30, high=120
中程度の模様にフォーカス（繰り返し抑制と特徴強調のバランス）
low=0, high=20
高速変化の除去、大域的な構造のみ強調


import cv2
import numpy as np
import matplotlib.pyplot as plt

def phase_correlation_with_subpixel(img1, img2, low=10, high=100):
    # 1. 前処理
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # 2. ハニング窓（エッジ抑制）
    win = cv2.createHanningWindow(img1.shape[::-1], cv2.CV_32F)
    img1 *= win
    img2 *= win

    # 3. DFT（複素数2チャンネル）
    dft1 = cv2.dft(img1, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft2 = cv2.dft(img2, flags=cv2.DFT_COMPLEX_OUTPUT)

    # 4. バンドパスフィルター
    dft1 = apply_bandpass_filter(dft1, low, high)
    dft2 = apply_bandpass_filter(dft2, low, high)

    # 5. クロスパワースペクトルの正規化
    num = cv2.mulSpectrums(dft1, dft2, flags=0, conjB=True)
    mag = cv2.magnitude(num[..., 0], num[..., 1])
    mag[mag == 0] = 1e-6  # 0除算防止
    num[..., 0] /= mag
    num[..., 1] /= mag

    # 6. 相関マップ（逆FFT）
    corr = cv2.idft(num, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # 7. サブピクセル補間
    peak = subpixel_peak(corr)

    # 8. wrap-around補正
    h, w = corr.shape
    dx, dy = peak
    if dx > w / 2: dx -= w
    if dy > h / 2: dy -= h

    return (dx, dy), corr, peak

def apply_bandpass_filter(dft, low=10, high=100):
    h, w = dft.shape[:2]
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    mask = np.logical_and(dist > low, dist < high).astype(np.float32)
    mask = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
    return dft * mask

def subpixel_peak(corr):
    _, _, max_loc, _ = cv2.minMaxLoc(corr)
    x, y = max_loc

    def parabolic(fm1, f0, fp1):
        denom = (fm1 - 2*f0 + fp1)
        if denom == 0:
            return 0
        return 0.5 * (fm1 - fp1) / denom

    if 1 <= x < corr.shape[1] - 1:
        dx = parabolic(corr[y, x-1], corr[y, x], corr[y, x+1])
    else:
        dx = 0
    if 1 <= y < corr.shape[0] - 1:
        dy = parabolic(corr[y-1, x], corr[y, x], corr[y+1, x])
    else:
        dy = 0

    return (x + dx, y + dy)

# ✅ 使用例
# img1, img2 は同サイズのグレースケール画像
# 例: img1 = cv2.imread('img1.png', 0)

# dxdy, corrmap, peakraw = phase_correlation_with_subpixel(img1, img2)
# print(f"推定ズレ量（サブピクセル）: dx={dxdy[0]:.4f}, dy={dxdy[1]:.4f}")









✅ 前提（一般化）

紙の高さを h、
固定点（支点）の位置を 0〜1の相対値（上=0、下=1、中=0.5）で表すとします：
	•	anchor = 0.0 → 上端固定
	•	anchor = 1.0 → 下端固定
	•	anchor = 0.5 → 中央固定

⸻

✅ 回転後のずれ（Δx, Δy）の計算式

\begin{align*}
L_{\text{top}} &= -h \cdot \text{anchor} \\
L_{\text{bottom}} &= h \cdot (1 - \text{anchor})
\end{align*}

それぞれが、支点から上端／下端までの距離です。

回転角度を a（ラジアン）としたとき：

\begin{align*}
\Delta x_{\text{top}} &= L_{\text{top}} \cdot \sin(a) \\
\Delta y_{\text{top}} &= L_{\text{top}} \cdot (1 - \cos(a)) \\
\Delta x_{\text{bottom}} &= L_{\text{bottom}} \cdot \sin(a) \\
\Delta y_{\text{bottom}} &= L_{\text{bottom}} \cdot (1 - \cos(a))
\end{align*}


解法：回転後の位置から差分を求める

数学的に：

\text{Q}{\text{rot}} = R\theta \cdot (Q - P) + P

つまり：
	1.	点Qを 支点P中心にずらす（Q - P）
	2.	回転行列で回す
	3.	元の位置Pに戻す

⸻

✅ 実装：座標ベースの支点を指定して、任意点の回転差分を求める


import numpy as np

def rotate_point_around_center(Q, P, angle_deg):
    """
    Q: 回転させたい点（x, y）
    P: 回転の支点（x, y）
    angle_deg: 回転角（度）
    """
    angle_rad = np.deg2rad(angle_deg)
    xq, yq = Q
    xp, yp = P

    # 支点を原点に移動して回転
    x_shifted = xq - xp
    y_shifted = yq - yp

    x_rot =  x_shifted * np.cos(angle_rad) - y_shifted * np.sin(angle_rad)
    y_rot =  x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad)

    # 元の位置に戻す
    x_new = x_rot + xp
    y_new = y_rot + yp

    dx = x_new - xq
    dy = y_new - yq

    return dx, dy  # ずれ量（Δx, Δy）





def bandpass_filter_fft(img, low=10, high=100):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    mask = (r > low) & (r < high)  # 帯域指定
    fshift[~mask] = 0

    f_ishift = np.fft.ifftshift(fshift)
    filtered = np.fft.ifft2(f_ishift)
    return np.abs(filtered)
    


angles = []  # 例：分割ブロックで求めた角度群
for block in blocks:
    angle = compute_phase_angle(block)
    if angle is not None:
        angles.append(angle)
        
        
        
import scipy.stats as stats

def remove_outliers(data, z_thresh=2.5):
    z = stats.zscore(data)
    return np.array(data)[np.abs(z) < z_thresh]

clean_angles = remove_outliers(angles)
stable_angle = np.mean(clean_angles)

import scipy.stats as stats

def remove_outliers(data, z_thresh=2.5):
    z = stats.zscore(data)
    return np.array(data)[np.abs(z) < z_thresh]

clean_angles = remove_outliers(angles)
stable_angle = np.mean(clean_angles)



import cv2
import numpy as np

def phase_correlation_fft(img1, img2, window=True):
    # グレースケールかつ float32 に変換
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    # オプション：ハニング窓をかけるとピークが安定（境界抑制）
    if window:
        hann = cv2.createHanningWindow(img1.shape[::-1], cv2.CV_32F)
        img1 *= hann
        img2 *= hann

    # DFT（FFT）を取得（2チャネルの複素数：real + imag）
    dft1 = cv2.dft(img1, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft2 = cv2.dft(img2, flags=cv2.DFT_COMPLEX_OUTPUT)

    # conj(D2)
    dft2_conj = dft2.copy()
    dft2_conj[..., 1] *= -1  # 虚数部の符号を反転して共役

    # R = (F1 * conj(F2)) / |F1 * conj(F2)|
    numerator = cv2.mulSpectrums(dft1, dft2_conj, 0, conjB=False)
    mag = cv2.magnitude(numerator[..., 0], numerator[..., 1])
    mag[mag == 0] = 1e-10  # 0除算防止
    numerator[..., 0] /= mag
    numerator[..., 1] /= mag

    # 逆FFT → 相関画像
    corr = cv2.idft(numerator, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # ピーク位置（＝ずれ量）
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr)
    shift = np.array(max_loc, dtype=np.float32)

    # wrap-around補正
    h, w = img1.shape
    if shift[0] > w / 2:
        shift[0] -= w
    if shift[1] > h / 2:
        shift[1] -= h

    return shift, corr  # シフト量と相関画像
    
    

import cv2
import numpy as np
import matplotlib.pyplot as plt

def phase_correlation_fft_with_debug(img1, img2, window=True, debug=False):
    # --- 入力画像の前処理 ---
    # グレースケール化 + float32型に変換
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    # --- ハニング窓を適用（端の影響を減らす） ---
    if window:
        hann = cv2.createHanningWindow(img1.shape[::-1], cv2.CV_32F)
        img1 *= hann
        img2 *= hann

    # --- DFT（FFT） ---
    dft1 = cv2.dft(img1, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft2 = cv2.dft(img2, flags=cv2.DFT_COMPLEX_OUTPUT)

    # --- DFT2の複素共役を取る ---
    dft2_conj = dft2.copy()
    dft2_conj[..., 1] *= -1  # 虚部の符号を反転

    # --- クロスパワースペクトルの計算 ---
    numerator = cv2.mulSpectrums(dft1, dft2_conj, 0, conjB=False)
    mag = cv2.magnitude(numerator[..., 0], numerator[..., 1])
    mag[mag == 0] = 1e-10  # 0除算防止
    numerator[..., 0] /= mag
    numerator[..., 1] /= mag

    # --- 逆DFT（相関マップの計算）---
    corr = cv2.idft(numerator, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # --- 相関マップから最大値の位置（シフト量）を取得 ---
    _, _, _, max_loc = cv2.minMaxLoc(corr)
    shift = np.array(max_loc, dtype=np.float32)
    h, w = img1.shape
    if shift[0] > w / 2:
        shift[0] -= w
    if shift[1] > h / 2:
        shift[1] -= h

    # --- debug=True なら途中結果をグラフ表示 ---
    if debug:
        fft_mag1 = np.log(cv2.magnitude(dft1[..., 0], dft1[..., 1]) + 1)
        fft_mag2 = np.log(cv2.magnitude(dft2[..., 0], dft2[..., 1]) + 1)
        cross_power = np.log(cv2.magnitude(numerator[..., 0], numerator[..., 1]) + 1)

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs[0, 0].imshow(img1, cmap='gray')
        axs[0, 0].set_title("Input Image 1")
        axs[0, 1].imshow(img2, cmap='gray')
        axs[0, 1].set_title("Input Image 2")
        axs[0, 2].imshow(corr, cmap='hot')
        axs[0, 2].set_title("Phase Correlation Map")

        axs[1, 0].imshow(fft_mag1, cmap='gray')
        axs[1, 0].set_title("FFT Magnitude 1")
        axs[1, 1].imshow(fft_mag2, cmap='gray')
        axs[1, 1].set_title("FFT Magnitude 2")
        axs[1, 2].imshow(cross_power, cmap='gray')
        axs[1, 2].set_title("Cross Power Spectrum")

        for ax in axs.ravel():
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    return shift, corr


