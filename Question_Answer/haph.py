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