import cv2
import numpy as np

# 入力画像の読み込み（グレースケール）
img = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)

# ===== ① Sobelフィルターでエッジ検出 =====
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)
sobel = cv2.convertScaleAbs(sobel)

# ===== ② Cannyエッジ検出 =====
canny = cv2.Canny(img, threshold1=50, threshold2=150)

# ===== ③ HoughCirclesで円検出 =====
# ノイズ除去（中央値フィルター）
blurred = cv2.medianBlur(img, 5)

# 円の検出
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                           param1=100, param2=30, minRadius=20, maxRadius=100)

# 元画像をカラーに変換して描画
output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# 円の描画
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]
        cv2.circle(output, center, radius, (0, 255, 0), 2)
        cv2.circle(output, center, 2, (0, 0, 255), 3)

# ===== ④ 表示（必要に応じて保存も可能） =====
cv2.imshow("Original", img)
cv2.imshow("Sobel", sobel)
cv2.imshow("Canny", canny)
cv2.imshow("Hough Circles", output)

cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # 移動平均のために使用

# --- 移動平均関数 ---
def moving_average(data, window_size=5):
    return pd.Series(data).rolling(window=window_size, center=True).mean().to_numpy()

# --- アルゴリズム定義（省略: 前回と同じ） ---
# shift_by_phase_correlation
# shift_by_template_matching
# shift_by_hough_circle_center

# --- 比較用画像ペア ---
image_pairs = [("img1.png", "img2.png"), ("img2.png", "img3.png"), ("img3.png", "img4.png")]
# もっと多くてももちろんOK

shift_phase = []
shift_template = []
shift_hough = []

for imgA_path, imgB_path in image_pairs:
    imgA = cv2.imread(imgA_path, cv2.IMREAD_GRAYSCALE)
    imgB = cv2.imread(imgB_path, cv2.IMREAD_GRAYSCALE)
    if imgA is None or imgB is None:
        print(f"読み込みエラー: {imgA_path}, {imgB_path}")
        continue
    shift_phase.append(shift_by_phase_correlation(imgA, imgB))
    shift_template.append(shift_by_template_matching(imgA, imgB))
    shift_hough.append(shift_by_hough_circle_center(imgA, imgB))

# --- NaN補完（HoughなどでNoneが出たとき） ---
shift_phase = [np.nan if v is None else v for v in shift_phase]
shift_template = [np.nan if v is None else v for v in shift_template]
shift_hough = [np.nan if v is None else v for v in shift_hough]

# --- 移動平均の計算 ---
ma_phase = moving_average(shift_phase, window_size=5)
ma_template = moving_average(shift_template, window_size=5)
ma_hough = moving_average(shift_hough, window_size=5)

# --- グラフ表示 ---
x = list(range(len(shift_phase)))

plt.figure(figsize=(12, 6))

# オリジナルデータ（点線）
plt.plot(x, shift_phase, 'o--', label='Phase Corr.', alpha=0.4)
plt.plot(x, shift_template, 's--', label='Template Match', alpha=0.4)
plt.plot(x, shift_hough, '^--', label='Hough Center', alpha=0.4)

# 移動平均（実線）
plt.plot(x, ma_phase, '-', label='Phase Corr. (MA)', linewidth=2)
plt.plot(x, ma_template, '-', label='Template Match (MA)', linewidth=2)
plt.plot(x, ma_hough, '-', label='Hough Center (MA)', linewidth=2)

plt.title("Vertical Shift with Moving Averages")
plt.xlabel("Image Pair Index")
plt.ylabel("Vertical Shift (pixels)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- 移動平均関数 ---
def moving_average(data, window_size=5):
    return pd.Series(data).rolling(window=window_size, center=True).mean().to_numpy()

# --- 各アルゴリズム関数（以前の定義をそのまま使えます） ---
def shift_by_phase_correlation(img1, img2):
    win = cv2.createHanningWindow(img1.shape[::-1], cv2.CV_64F)
    shift, _ = cv2.phaseCorrelate(img1 * win, img2 * win)
    return shift[1]

def shift_by_template_matching(img1, img2):
    template = img1[0:100, :]
    res = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    return max_loc[1]

def shift_by_hough_circle_center(img1, img2):
    def get_center_y(img):
        blurred = cv2.medianBlur(img, 5)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 30,
                                   param1=100, param2=30, minRadius=20, maxRadius=100)
        if circles is None:
            return None
        circles = np.uint16(np.around(circles))
        return np.mean(circles[0, :, 1])
    y1 = get_center_y(img1)
    y2 = get_center_y(img2)
    return y2 - y1 if y1 is not None and y2 is not None else None

# --- アルゴリズム辞書を定義 ---
algorithms = {
    "Phase Correlation": shift_by_phase_correlation,
    "Template Matching": shift_by_template_matching,
    "Hough Circles": shift_by_hough_circle_center
}

# --- 入力画像ペアのリスト ---
image_pairs = [("img1.png", "img2.png"), ("img2.png", "img3.png"), ("img3.png", "img4.png")]

# --- 結果格納用の辞書（アルゴ名 → シフト値リスト） ---
shift_dict = {name: [] for name in algorithms}

# --- 各アルゴでずれ量を計算して辞書に格納 ---
for imgA_path, imgB_path in image_pairs:
    imgA = cv2.imread(imgA_path, cv2.IMREAD_GRAYSCALE)
    imgB = cv2.imread(imgB_path, cv2.IMREAD_GRAYSCALE)
    if imgA is None or imgB is None:
        print(f"読み込みエラー: {imgA_path}, {imgB_path}")
        continue

    for name, func in algorithms.items():
        try:
            shift = func(imgA, imgB)
        except Exception as e:
            shift = None
            print(f"{name} でエラー: {e}")
        shift_dict[name].append(np.nan if shift is None else shift)

# --- グラフ描画（移動平均含む） ---
x = list(range(len(image_pairs)))
plt.figure(figsize=(12, 6))

for name, values in shift_dict.items():
    ma = moving_average(values, window_size=5)
    plt.plot(x, values, linestyle='--', marker='o', alpha=0.4, label=f"{name}")
    plt.plot(x, ma, linewidth=2, label=f"{name} (MA)")

plt.title("Vertical Shift by Algorithm with Moving Average")
plt.xlabel("Image Pair Index")
plt.ylabel("Vertical Shift (pixels)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



from tqdm import tqdm  # プログレスバー用

# --- 結果格納用辞書 ---
shift_dict = {name: [] for name in algorithms}

# --- tqdmで画像ペアのループをラップ ---
for idx, (imgA_path, imgB_path) in enumerate(tqdm(image_pairs, desc="Processing image pairs")):
    imgA = cv2.imread(imgA_path, cv2.IMREAD_GRAYSCALE)
    imgB = cv2.imread(imgB_path, cv2.IMREAD_GRAYSCALE)
    if imgA is None or imgB is None:
        print(f"読み込みエラー: {imgA_path}, {imgB_path}")
        continue

    for name, func in algorithms.items():
        try:
            shift = func(imgA, imgB)
        except Exception as e:
            shift = None
            print(f"{name} エラー: {e}")
        shift_dict[name].append(np.nan if shift is None else shift)

