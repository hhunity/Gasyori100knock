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


