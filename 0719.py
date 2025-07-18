import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像を読み込み（グレースケール前提）
img1 = cv2.imread('img1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('img2.png', cv2.IMREAD_GRAYSCALE)

# 画像サイズ確認
assert img1.shape == img2.shape, "画像サイズが一致していません"

# 差分を計算する x 座標を指定
x = 100

# 各画像の x=100 における縦方向の画素値を取得
col1 = img1[:, x]
col2 = img2[:, x]

# 差分計算
diff = col1.astype(np.int16) - col2.astype(np.int16)

# グラフ表示
plt.figure(figsize=(10, 5))
plt.plot(diff, label='Difference (img1 - img2)')
plt.xlabel('Y-coordinate (row)')
plt.ylabel('Pixel Difference')
plt.title(f'Vertical Difference at x = {x}')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
import numpy as np

def count_vertical_runs(diff_image, thresh):
    """
    各列ごとに、y方向に沿って絶対差が閾値を超える連続カウントを取得

    Parameters:
        diff_image: 2D numpy array (H x W)
        thresh: float（しきい値）

    Returns:
        counts: 2D numpy array（H x W） 各位置における連続カウント
    """
    H, W = diff_image.shape
    bin_image = (np.abs(diff_image) > thresh).astype(np.uint8)
    counts = np.zeros_like(diff_image, dtype=np.int32)

    for x in range(W):
        count = 0
        for y in range(H):
            if bin_image[y, x]:
                count += 1
            else:
                count = 0
            counts[y, x] = count
    return counts
