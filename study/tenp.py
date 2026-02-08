import numpy as np
import cv2

def _hann2(h, w):
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)

def _fft_mag(img_f32: np.ndarray, use_hann=True, dc_remove=True):
    """Return fftshift(|FFT|) for a float32 grayscale ROI."""
    x = img_f32
    if dc_remove:
        x = x - float(np.mean(x))
    if use_hann:
        x = x * _hann2(*x.shape)
    F = np.fft.fft2(x)
    F = np.fft.fftshift(F)
    mag = np.abs(F)
    return mag

def _nonmax_suppression_peaks(mag, num_peaks=20, radius=6, dc_radius=16):
    """
    Simple peak picker on magnitude spectrum:
    - zeros out DC neighborhood
    - iteratively picks max, then suppresses a radius
    Returns list of (u, v, value) in fftshift coordinates, with origin at center.
    """
    H, W = mag.shape
    cy, cx = H // 2, W // 2
    m = mag.copy()

    # suppress DC
    yy, xx = np.ogrid[:H, :W]
    dc_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= dc_radius ** 2
    m[dc_mask] = 0.0

    peaks = []
    for _ in range(num_peaks):
        idx = np.argmax(m)
        val = float(m.flat[idx])
        if val <= 0:
            break
        y, x = np.unravel_index(idx, m.shape)
        # convert to frequency coordinates centered at (0,0)
        v = y - cy
        u = x - cx
        peaks.append((u, v, val))

        # suppress neighborhood
        y0, y1 = max(0, y - radius), min(H, y + radius + 1)
        x0, x1 = max(0, x - radius), min(W, x + radius + 1)
        m[y0:y1, x0:x1] = 0.0

    return peaks

def _canonical_angle(theta):
    """Map angle to [-pi/2, pi/2) because +/- directions are equivalent for peak orientation."""
    # Treat direction k and -k as the same orientation -> fold modulo pi
    theta = (theta + np.pi) % np.pi
    if theta >= np.pi / 2:
        theta -= np.pi
    return theta  # in [-pi/2, pi/2)

def estimate_roll_from_fft_peaks(
    img_gray: np.ndarray,
    roi=None,
    num_peaks=30,
    peak_radius=6,
    dc_radius=16,
    min_r=20,           # ignore very low frequency peaks
    noncollinear_deg=20 # ensure 2 picked peaks are sufficiently different in angle
):
    """
    Estimate roll angle (orientation) from FFT magnitude peaks of a 2D periodic pattern.

    Returns:
      roll_rad: estimated roll angle in radians (in [-pi/2, pi/2))
      info: dict with selected peaks & angles
    """
    # --- prepare ROI ---
    if roi is not None:
        x, y, w, h = roi
        img = img_gray[y:y+h, x:x+w]
    else:
        img = img_gray

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_f32 = img.astype(np.float32) / 255.0

    # --- FFT magnitude ---
    mag = _fft_mag(img_f32, use_hann=True, dc_remove=True)

    # --- peak picking ---
    peaks = _nonmax_suppression_peaks(
        mag, num_peaks=num_peaks, radius=peak_radius, dc_radius=dc_radius
    )

    # --- filter peaks by radius (avoid near-DC / very low frequency) ---
    cand = []
    for u, v, val in peaks:
        r = np.hypot(u, v)
        if r >= min_r:
            theta = np.arctan2(v, u)  # angle of frequency vector
            theta = _canonical_angle(theta)
            cand.append((u, v, val, r, theta))

    if len(cand) < 1:
        raise RuntimeError("No usable FFT peaks found. Increase ROI size or adjust thresholds.")

    # Sort by magnitude descending
    cand.sort(key=lambda t: t[2], reverse=True)

    # --- choose 2 non-collinear strong peaks (optional but recommended) ---
    # For a grid, you want two directions; for roll estimation alone, one direction can be enough,
    # but two helps stability / reject harmonics.
    chosen = [cand[0]]
    ang0 = cand[0][4]
    for c in cand[1:]:
        ang = c[4]
        d = abs(ang - ang0)
        d = min(d, np.pi - d)  # angular distance modulo pi
        if d >= np.deg2rad(noncollinear_deg):
            chosen.append(c)
            break

    # If we didn't find a second, just use the best one.
    # Roll is defined by the dominant frequency direction; grid gives multiple.
    # We'll compute a weighted average of chosen directions (1 or 2).
    thetas = np.array([c[4] for c in chosen], dtype=np.float64)
    weights = np.array([c[2] for c in chosen], dtype=np.float64)

    # Weighted circular mean on angles modulo pi (using doubled-angle trick)
    # because theta and theta+pi represent same direction.
    C = np.sum(weights * np.cos(2 * thetas))
    S = np.sum(weights * np.sin(2 * thetas))
    roll = 0.5 * np.arctan2(S, C)
    roll = _canonical_angle(roll)

    info = {
        "roi_used": roi,
        "candidates": cand,     # list of (u,v,val,r,theta)
        "chosen": chosen,       # chosen peaks (u,v,val,r,theta)
        "roll_rad": float(roll),
        "roll_deg": float(np.rad2deg(roll)),
    }
    return float(roll), info

def estimate_roll_diff(A_gray: np.ndarray, B_gray: np.ndarray, roi=None, **kwargs):
    """
    Return roll difference (B - A) in radians and degrees, using FFT peak orientations.
    """
    ra, ia = estimate_roll_from_fft_peaks(A_gray, roi=roi, **kwargs)
    rb, ib = estimate_roll_from_fft_peaks(B_gray, roi=roi, **kwargs)

    d = rb - ra
    # wrap to [-pi/2, pi/2)
    d = _canonical_angle(d)

    return {
        "rollA_rad": ra, "rollA_deg": ia["roll_deg"],
        "rollB_rad": rb, "rollB_deg": ib["roll_deg"],
        "droll_rad": float(d), "droll_deg": float(np.rad2deg(d)),
        "infoA": ia, "infoB": ib
    }

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    imgA = cv2.imread("A.png", cv2.IMREAD_GRAYSCALE)
    imgB = cv2.imread("B.png", cv2.IMREAD_GRAYSCALE)

    # If you know a stable periodic region, specify ROI = (x, y, w, h)
    # roi = (100, 200, 512, 512)
    roi = None

    res = estimate_roll_diff(
        imgA, imgB, roi=roi,
        num_peaks=40,
        peak_radius=7,
        dc_radius=20,
        min_r=25,
        noncollinear_deg=25
    )

    print("rollA(deg):", res["rollA_deg"])
    print("rollB(deg):", res["rollB_deg"])
    print("droll(deg):", res["droll_deg"])