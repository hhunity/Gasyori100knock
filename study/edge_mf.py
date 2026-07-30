"""
slanted_edge_mtf.py

ISO 12233 準拠の Slanted-Edge 法による MTF 測定ツール
ESF (Edge Spread Function) → LSF (微分) → FFT → MTF

--------------------------------------------------------------------
使い方(実画像):
    python slanted_edge_mtf.py --image edge.png --pixel-pitch 2.0

使い方(合成エッジで動作確認・デモ):
    python slanted_edge_mtf.py --demo

主な関数:
    measure_mtf(roi, pixel_pitch_um, oversample=4)
        -> dict(freq_cyc_per_um, mtf, esf_x, esf, lsf_x, lsf, angle_deg, nyquist_mtf)
--------------------------------------------------------------------
"""

import argparse
import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. エッジ角度の検出
# ----------------------------------------------------------------------
def detect_edge_centers(roi):
    """
    各行(row)ごとにエッジの中心位置(sub-pixel)を検出する。
    エッジは横方向(x方向)に明暗が変化している前提。
    重心法(intensity-weighted centroid of the gradient)でサブピクセル位置を求める。

    Returns
    -------
    rows : ndarray  (使用した行インデックス)
    centers : ndarray (各行でのエッジ位置 x [pixel])
    """
    h, w = roi.shape
    rows = []
    centers = []

    for y in range(h):
        line = roi[y, :].astype(np.float64)
        grad = np.gradient(line)
        g_abs = np.abs(grad)

        if g_abs.max() < 1e-6:
            continue

        # 勾配の重心(サブピクセル位置)
        idx = np.arange(w)
        # ノイズ対策: ピーク付近だけを使う(ピークの半値以上の範囲)
        thresh = g_abs.max() * 0.3
        mask = g_abs >= thresh
        if mask.sum() < 2:
            continue

        center = np.sum(idx[mask] * g_abs[mask]) / np.sum(g_abs[mask])
        rows.append(y)
        centers.append(center)

    return np.array(rows), np.array(centers)


def fit_edge_line(rows, centers):
    """
    エッジ位置 x(y) = a*y + b を直線フィットし、傾き角度を返す。
    """
    a, b = np.polyfit(rows, centers, 1)
    angle_deg = np.degrees(np.arctan(a))
    return a, b, angle_deg


# ----------------------------------------------------------------------
# 2. ESF (Edge Spread Function) の構築(オーバーサンプリング投影)
# ----------------------------------------------------------------------
def build_esf(roi, a, b, oversample=4, half_width_px=8):
    """
    各画素を、フィット直線からの垂直距離(≒エッジ法線方向の距離)に投影し、
    ビニングしてオーバーサンプリングされたESFを作る。

    Parameters
    ----------
    roi : 2D array
    a, b : エッジ直線 x = a*y + b の係数
    oversample : 1画素あたりの分割数(オーバーサンプリング倍率)
    half_width_px : エッジ中心から前後何画素分を使うか

    Returns
    -------
    esf_x : ndarray  (画素単位のオフセット, オーバーサンプル済み)
    esf   : ndarray  (対応する輝度値, 平均化済み)
    """
    h, w = roi.shape
    bin_w = 1.0 / oversample
    bins_edges = np.arange(-half_width_px, half_width_px + bin_w, bin_w)
    sums = np.zeros(len(bins_edges) - 1)
    counts = np.zeros(len(bins_edges) - 1)

    for y in range(h):
        edge_x = a * y + b
        xs = np.arange(w) - edge_x  # エッジからの距離(画素)
        vals = roi[y, :].astype(np.float64)

        valid = (xs >= bins_edges[0]) & (xs < bins_edges[-1])
        idx = np.digitize(xs[valid], bins_edges) - 1

        np.add.at(sums, idx, vals[valid])
        np.add.at(counts, idx, 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        esf = sums / counts

    esf_x = (bins_edges[:-1] + bins_edges[1:]) / 2

    # 欠損ビンの補間
    valid_mask = ~np.isnan(esf)
    if valid_mask.sum() < len(esf):
        esf = np.interp(esf_x, esf_x[valid_mask], esf[valid_mask])

    return esf_x, esf


# ----------------------------------------------------------------------
# 3. LSF (微分) と 窓関数
# ----------------------------------------------------------------------
def esf_to_lsf(esf_x, esf, apply_window=True):
    lsf = np.gradient(esf, esf_x)

    if apply_window:
        window = np.hamming(len(lsf))
        lsf = lsf * window

    return lsf


# ----------------------------------------------------------------------
# 4. FFT -> MTF
# ----------------------------------------------------------------------
def lsf_to_mtf(esf_x, lsf, oversample, pixel_pitch_um):
    """
    LSFをFFTしてMTFを求める。
    横軸は 空間周波数 [cycle/um]。

    サンプリング間隔(オーバーサンプル後) = pixel_pitch_um / oversample
    """
    n = len(lsf)
    sample_spacing_um = pixel_pitch_um / oversample

    mtf_complex = np.fft.rfft(lsf)
    mtf = np.abs(mtf_complex)
    mtf = mtf / mtf[0]  # DC(周波数0)で正規化

    freq = np.fft.rfftfreq(n, d=sample_spacing_um)  # cycle/um

    return freq, mtf


# ----------------------------------------------------------------------
# 5. まとめ関数
# ----------------------------------------------------------------------
def measure_mtf(roi, pixel_pitch_um, oversample=4, half_width_px=8):
    """
    roi (2D numpy array, グレースケール) からMTFを測定する。

    Returns
    -------
    result : dict
    """
    rows, centers = detect_edge_centers(roi)
    if len(rows) < 10:
        raise ValueError("エッジを十分検出できませんでした。ROIやしきい値を見直してください。")

    a, b, angle_deg = fit_edge_line(rows, centers)

    if abs(angle_deg) < 1.0 or abs(angle_deg) > 15.0:
        print(f"[警告] エッジ角度 {angle_deg:.2f}度 は推奨範囲(概ね3〜10度)から外れています。"
              f" オーバーサンプリング精度が低下する可能性があります。")

    esf_x, esf = build_esf(roi, a, b, oversample=oversample, half_width_px=half_width_px)
    lsf = esf_to_lsf(esf_x, esf)
    freq, mtf = lsf_to_mtf(esf_x, lsf, oversample, pixel_pitch_um)

    nyquist = 1.0 / (2 * pixel_pitch_um)
    nyquist_mtf = float(np.interp(nyquist, freq, mtf))

    return {
        "angle_deg": angle_deg,
        "esf_x": esf_x,
        "esf": esf,
        "lsf_x": esf_x,
        "lsf": lsf,
        "freq_cyc_per_um": freq,
        "mtf": mtf,
        "nyquist_cyc_per_um": nyquist,
        "nyquist_mtf": nyquist_mtf,
    }


# ----------------------------------------------------------------------
# 6. プロット
# ----------------------------------------------------------------------
def plot_result(result, pixel_pitch_um, out_path="mtf_result.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(result["esf_x"], result["esf"], lw=1)
    axes[0].set_title(f"ESF (edge angle={result['angle_deg']:.2f} deg)")
    axes[0].set_xlabel("distance from edge [pixel]")
    axes[0].set_ylabel("intensity")

    axes[1].plot(result["lsf_x"], result["lsf"], lw=1)
    axes[1].set_title("LSF (dESF/dx, windowed)")
    axes[1].set_xlabel("distance from edge [pixel]")

    axes[2].plot(result["freq_cyc_per_um"], result["mtf"], lw=1.5, label="MTF")
    axes[2].axvline(result["nyquist_cyc_per_um"], color="r", ls="--",
                     label=f"Nyquist ({result['nyquist_cyc_per_um']:.3f} cyc/um)")
    axes[2].axhline(result["nyquist_mtf"], color="gray", ls=":", lw=0.8)
    axes[2].scatter([result["nyquist_cyc_per_um"]], [result["nyquist_mtf"]], color="r", zorder=5)
    axes[2].annotate(f"MTF@Nyquist={result['nyquist_mtf']*100:.1f}%",
                      xy=(result["nyquist_cyc_per_um"], result["nyquist_mtf"]),
                      xytext=(10, 10), textcoords="offset points", color="r")
    axes[2].set_xlim(0, result["nyquist_cyc_per_um"] * 2.5)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title(f"MTF (pixel pitch={pixel_pitch_um} um)")
    axes[2].set_xlabel("spatial frequency [cycle/um]")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[saved] {out_path}")
    return out_path


# ----------------------------------------------------------------------
# 7. デモ用: 合成エッジ画像の生成(理論MTFとの照合用)
# ----------------------------------------------------------------------
def generate_synthetic_edge(width=120, height=200, angle_deg=5.0,
                             blur_sigma_px=1.5, noise_std=2.0, bit_depth=8):
    """
    既知のガウシアンPSF(sigma指定)でぼかした傾斜エッジ画像を作る。
    理論MTF(ガウシアン) = exp(-2*(pi*sigma*f)^2)  ※fは cycle/pixel
    と比較することで、本コードの正しさを検証できる。
    """
    yy, xx = np.mgrid[0:height, 0:width]
    a = np.tan(np.radians(angle_deg))
    edge_x = width / 2 + a * (yy - height / 2)

    # シャープなエッジ(ステップ関数)
    sharp = np.where(xx > edge_x, 200.0, 50.0)

    # ガウシアンぼかし(x方向のみ、角度は近似的に無視して単純化)
    blurred = ndimage.gaussian_filter1d(sharp, sigma=blur_sigma_px, axis=1)

    noisy = blurred + np.random.normal(0, noise_std, blurred.shape)
    noisy = np.clip(noisy, 0, 2**bit_depth - 1)
    return noisy.astype(np.float64)


def theoretical_gaussian_mtf(freq_cyc_per_um, pixel_pitch_um, sigma_px):
    sigma_um = sigma_px * pixel_pitch_um
    return np.exp(-2 * (np.pi * sigma_um * freq_cyc_per_um) ** 2)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Slanted-edge MTF measurement")
    parser.add_argument("--image", type=str, help="エッジを含む画像ファイル(グレースケール推奨)")
    parser.add_argument("--pixel-pitch", type=float, default=2.0, help="撮像ピッチ [um/pixel]")
    parser.add_argument("--oversample", type=int, default=4, help="オーバーサンプリング倍率")
    parser.add_argument("--half-width", type=float, default=8, help="エッジ中心から使う片側幅[pixel]")
    parser.add_argument("--out", type=str, default="mtf_result.png")
    parser.add_argument("--demo", action="store_true", help="合成エッジで動作検証する")
    args = parser.parse_args()

    if args.demo:
        print("=== デモモード: 合成エッジ(既知のガウシアンボケ)で検証 ===")
        sigma_px = 1.2
        roi = generate_synthetic_edge(blur_sigma_px=sigma_px, angle_deg=5.0)
        result = measure_mtf(roi, args.pixel_pitch, oversample=args.oversample,
                              half_width_px=args.half_width)

        theo = theoretical_gaussian_mtf(result["freq_cyc_per_um"], args.pixel_pitch, sigma_px)

        print(f"検出エッジ角度: {result['angle_deg']:.2f} deg (真値 5.00 deg)")
        print(f"ナイキスト周波数: {result['nyquist_cyc_per_um']:.4f} cycle/um")
        print(f"MTF@Nyquist(実測): {result['nyquist_mtf']*100:.1f}%")
        theo_nyq = float(np.interp(result["nyquist_cyc_per_um"], result["freq_cyc_per_um"], theo))
        print(f"MTF@Nyquist(理論): {theo_nyq*100:.1f}%")

        out_path = plot_result(result, args.pixel_pitch, args.out)

        # 理論値との比較プロットも追加
        plt.figure(figsize=(6, 4))
        plt.plot(result["freq_cyc_per_um"], result["mtf"], label="measured")
        plt.plot(result["freq_cyc_per_um"], theo, "--", label="theoretical (gaussian)")
        plt.axvline(result["nyquist_cyc_per_um"], color="r", ls=":", label="Nyquist")
        plt.xlim(0, result["nyquist_cyc_per_um"] * 2.5)
        plt.ylim(0, 1.05)
        plt.xlabel("spatial frequency [cycle/um]")
        plt.ylabel("MTF")
        plt.legend()
        plt.title("Measured vs Theoretical MTF (validation)")
        plt.tight_layout()
        plt.savefig("mtf_validation.png", dpi=150)
        print("[saved] mtf_validation.png")
        return

    if not args.image:
        parser.error("--image を指定するか、--demo で動作確認してください。")

    from PIL import Image
    img = Image.open(args.image).convert("L")
    roi = np.array(img, dtype=np.float64)

    result = measure_mtf(roi, args.pixel_pitch, oversample=args.oversample,
                          half_width_px=args.half_width)

    print(f"検出エッジ角度: {result['angle_deg']:.2f} deg")
    print(f"ナイキスト周波数: {result['nyquist_cyc_per_um']:.4f} cycle/um "
          f"(撮像ピッチ {args.pixel_pitch} um)")
    print(f"MTF@Nyquist: {result['nyquist_mtf']*100:.1f}%")

    plot_result(result, args.pixel_pitch, args.out)


if __name__ == "__main__":
    main()
