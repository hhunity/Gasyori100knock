import cv2
import numpy as np

# 特徴点検出
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# マッチング
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.match(des1, des2)

# 対応点抽出
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# ホモグラフィ行列推定（RANSACで外れ値除去）
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# X方向の移動量
dx = H[0, 2]
print(f"X方向の平行移動: dx = {dx:.2f}")





#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int main() {
    // 画像読み込み（グレースケール）
    Mat img1 = imread("fixed.png", IMREAD_GRAYSCALE);
    Mat img2 = imread("moved.png", IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        cerr << "画像が読み込めませんでした" << endl;
        return -1;
    }

    // ORBで特徴点と記述子を抽出
    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> kp1, kp2;
    Mat desc1, desc2;
    orb->detectAndCompute(img1, noArray(), kp1, desc1);
    orb->detectAndCompute(img2, noArray(), kp2, desc2);

    // Brute-force matcher（ハミング距離）
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(desc1, desc2, matches);

    // 対応点を抽出
    vector<Point2f> pts1, pts2;
    for (const auto& m : matches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }

    // ホモグラフィ行列を推定（RANSACで外れ値除去）
    Mat inlierMask;
    Mat H = findHomography(pts1, pts2, RANSAC, 3, inlierMask);
    if (H.empty()) {
        cerr << "ホモグラフィ行列が推定できませんでした" << endl;
        return -1;
    }

    // 平行移動量（X方向）
    double dx = H.at<double>(0, 2);
    double dy = H.at<double>(1, 2);
    cout << "X方向の平行移動量: dx = " << dx << ", dy = " << dy << endl;

    // オプション：補正結果を確認したい場合
    Mat aligned;
    warpPerspective(img2, aligned, H, img2.size());
    imshow("Aligned", aligned);
    waitKey();

    return 0;
}


【画像位置合わせ手法の比較】

項目                            | POC（位相限定相関）     | ECC（拡張相関係数法）       | 特徴点マッチング＋ホモグラフィ
-------------------------------|--------------------------|-----------------------------|-------------------------------
処理時間（速度）               | ◎ 非常に高速（FFT一発）   | △ 遅い（反復最適化）         | ○ 中速（特徴点数次第）
検出できる変形方向             | △ 平行移動のみ           | ◎ 平行移動～射影まで対応     | ◎ 射影変形まで自由に対応
ノイズ耐性                     | △ 普通（周波数に影響）     | ◎ 強い（正規化相関）         | ○ RANSAC併用で対処可
濃淡変化（明るさ差）への強さ   | △ 弱い（感度高い）        | ◎ 強い（平均差を吸収）       | △ 弱い（特徴記述子は影響受ける）
サブピクセル精度               | ◎ あり（理論的に可能）     | ◎ あり（勾配最適化）         | △ 通常はピクセル精度
前提条件（使える画像）         | 同サイズグレースケール     | 正規化済みグレースケール     | 特徴がある（角・模様など）
変形が小さい場合の安定性       | ◎ 高精度                  | ◎ 高精度                     | ◎ 安定
変形が大きい場合の耐性         | × 非対応                  | △ 初期値依存で失敗もある     | ◎ マッチングできれば対応可能
平坦画像（特徴がない）への対応 | ○ 対応可（濃淡差あれば）   | ◎ 対応可                     | × 不可（マッチできない）

※ ◎=非常に良い、○=良い、△=注意、×=不向き




