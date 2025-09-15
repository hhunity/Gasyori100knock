
import cv2
import numpy as np

def bandpass_phase_correlation(img1, img2, low_cut=0.05, high_cut=0.5):
    # フーリエ変換
    dft1 = cv2.dft(np.float32(img1), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft2 = cv2.dft(np.float32(img2), flags=cv2.DFT_COMPLEX_OUTPUT)

    # シフト（低周波が中央に来るように）
    dft1_shift = np.fft.fftshift(dft1, axes=(0, 1))
    dft2_shift = np.fft.fftshift(dft2, axes=(0, 1))

    # バンドパスマスクを作成
    rows, cols = img1.shape
    crow, ccol = rows // 2, cols // 2
    radius_low = int(min(rows, cols) * low_cut)
    radius_high = int(min(rows, cols) * high_cut)
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius_high, 1, thickness=-1)
    cv2.circle(mask, (ccol, crow), radius_low, 0, thickness=-1)
    mask = mask.astype(np.float32)

    # 実部・虚部にマスクを適用
    mask_2ch = cv2.merge([mask, mask])
    dft1_filt = dft1_shift * mask_2ch
    dft2_filt = dft2_shift * mask_2ch

    # 逆シフト → 逆DFT
    dft1_filtered = np.fft.ifftshift(dft1_filt, axes=(0, 1))
    dft2_filtered = np.fft.ifftshift(dft2_filt, axes=(0, 1))
    img1_filtered = cv2.idft(dft1_filtered, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    img2_filtered = cv2.idft(dft2_filtered, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # Hanning窓を適用して位相相関
    win = cv2.createHanningWindow(img1.shape[::-1], cv2.CV_32F)
    shift, _ = cv2.phaseCorrelate(img1_filtered * win, img2_filtered * win)
    return shift[1]
    
    
def refine_shift_with_template(img1, img2, initial_shift_y, search_range=5, template_height=50):
    h, w = img1.shape

    # img1 からテンプレートを抽出（中央部など）
    top = h // 2 - template_height // 2
    template = img1[top:top+template_height, :]

    # img2 の中で ±search_range だけ上下にずらしてテンプレートマッチング
    best_y = None
    best_score = -1

    for offset in range(-search_range, search_range+1):
        y = int(round(top + initial_shift_y + offset))
        if y < 0 or y + template_height > h:
            continue
        region = img2[y:y+template_height, :]
        result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)
        if score > best_score:
            best_score = score
            best_y = y

    refined_shift = best_y - top if best_y is not None else initial_shift_y
    return refined_shift
    
    



def stable_phase_shift(img1, img2):
    h, w = img1.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_64F)

    # 上半分と下半分で計算して平均
    shifts = []
    for y1, y2 in [(0, h//2), (h//2, h)]:
        sub1 = img1[y1:y2, :] * win[y1:y2, :]
        sub2 = img2[y1:y2, :] * win[y1:y2, :]
        shift, _ = cv2.phaseCorrelate(sub1, sub2)
        shifts.append(shift[1])

    return np.mean(shifts)
    
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_frequency_spectrum(img):
    # グレースケール化 & float変換
    f = np.float32(img)

    # フーリエ変換（2チャネル: 実部・虚部）
    dft = cv2.dft(f, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 振幅スペクトルを計算
    real = dft_shift[:, :, 0]
    imag = dft_shift[:, :, 1]
    magnitude = np.sqrt(real**2 + imag**2)

    # 対数スケールにして視覚化しやすく
    magnitude_log = np.log1p(magnitude)

    # 描画用データ作成（密度制限のため間引き推奨）
    step = 4  # 画素を間引いて表示（重い場合は値を大きく）
    X = np.arange(0, magnitude.shape[1], step)
    Y = np.arange(0, magnitude.shape[0], step)
    X, Y = np.meshgrid(X, Y)
    Z = magnitude_log[::step, ::step]

    # 3Dプロット
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)
    ax.set_title("Frequency Magnitude Spectrum (log scale)")
    ax.set_xlabel("Frequency X")
    ax.set_ylabel("Frequency Y")
    ax.set_zlabel("Amplitude (log)")
    plt.tight_layout()
    plt.show()
    
    
    
    




img1, img2
 ↓ DFT
F1, F2
 ↓ conj & normalize
R(u,v) = (F1 * conj(F2)) / |F1 * conj(F2)|
 ↓ inverse DFT
corr = IDFT(R) → ピークの位置 = ずれ


import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_phase_correlation_peak(img1, img2, window=True, step=1):
    f1 = np.float32(img1)
    f2 = np.float32(img2)

    if window:
        win = cv2.createHanningWindow(img1.shape[::-1], cv2.CV_32F)
        f1 *= win
        f2 *= win

    # フーリエ変換
    dft1 = cv2.dft(f1, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft2 = cv2.dft(f2, flags=cv2.DFT_COMPLEX_OUTPUT)

    # クロスパワースペクトルの計算
    conj2 = cv2.mulSpectrums(dft1, dft2, 0, conjB=True)
    mag = cv2.magnitude(conj2[:, :, 0], conj2[:, :, 1])
    mag[mag == 0] = 1e-8
    cps = conj2 / mag[:, :, np.newaxis]

    # 相関マップを得る（逆DFT）
    corr = cv2.idft(cps, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # ピーク位置
    _, _, _, max_loc = cv2.minMaxLoc(corr)
    print(f"Correlation peak at: {max_loc} (x, y)")

    # === 2D 表示（ヒートマップ） ===
    plt.figure(figsize=(6, 5))
    plt.imshow(corr, cmap='jet')
    plt.title("Phase Correlation Peak Map (2D)")
    plt.colorbar(label='Correlation strength')
    plt.scatter(max_loc[0], max_loc[1], color='white', s=50, label='Peak')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === 3D 表示（相関強度の地形図） ===
    X = np.arange(0, corr.shape[1], step)
    Y = np.arange(0, corr.shape[0], step)
    X, Y = np.meshgrid(X, Y)
    Z = corr[::step, ::step]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)
    ax.set_title("Phase Correlation Peak Map (3D)")
    ax.set_xlabel("X offset")
    ax.set_ylabel("Y offset")
    ax.set_zlabel("Correlation")
    plt.tight_layout()
    plt.show()
    
    


