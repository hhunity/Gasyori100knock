# 3D visualization & quick analysis of a 512x512 image
# (Demo with a synthetic image. Replace `img` creation with your own data as needed.)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pandas as pd
from datetime import datetime
from pathlib import Path

# --- 1) Create or load a 512x512 image ---
# Demo: synthetic image = smooth gradient + Gaussian bump + sinusoidal texture
h, w = 512, 512
y = np.linspace(-1, 1, h)
x = np.linspace(-1, 1, w)
X, Y = np.meshgrid(x, y)

# Components
gradient = (X + Y + 2) / 4.0
gauss = np.exp(-((X*2.2)**2 + (Y*1.6)**2) * 2.5)
sine = 0.15 * (np.sin(18*np.pi*X) * np.cos(18*np.pi*Y))

img = gradient * 0.5 + gauss * 0.5 + sine
img = (img - img.min()) / (img.max() - img.min())  # normalize to [0,1]

# --- 2) 3D surface plot (Z = intensity) ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# downsample for faster rendering (optional)
step = 2
Xs = X[::step, ::step]
Ys = Y[::step, ::step]
Zs = img[::step, ::step]

ax.plot_surface(Xs, Ys, Zs, rstride=1, cstride=1, linewidth=0, antialiased=True)
ax.set_title("3D Surface (Intensity as Height)")
ax.set_xlabel("X (normalized)")
ax.set_ylabel("Y (normalized)")
ax.set_zlabel("Intensity")
plt.show()

# --- 3) 2D view for reference ---
plt.figure(figsize=(6, 6))
plt.imshow(img, cmap="gray", origin="lower")
plt.title("2D Intensity Map (Reference)")
plt.colorbar(label="Intensity")
plt.tight_layout()
plt.show()

# --- 4) Quick analysis ---
# Basic stats
vmin = float(img.min())
vmax = float(img.max())
vmean = float(img.mean())
vstd = float(img.std())

# Peak location
peak_idx = np.unravel_index(np.argmax(img), img.shape)
peak_y, peak_x = int(peak_idx[0]), int(peak_idx[1])
peak_val = float(img[peak_y, peak_x])

# Gradient metrics
gy, gx = np.gradient(img.astype(np.float64))
grad_mag = np.hypot(gx, gy)
gmean = float(grad_mag.mean())
gstd = float(grad_mag.std())
gmax = float(grad_mag.max())
gmax_idx = np.unravel_index(np.argmax(grad_mag), grad_mag.shape)
gmax_y, gmax_x = int(gmax_idx[0]), int(gmax_idx[1])

# Frequency analysis (radially averaged power spectrum)
# Compute 2D FFT and power spectrum
fft = np.fft.fft2(img)
ps = np.abs(np.fft.fftshift(fft)) ** 2

# Radial bins
cy, cx = h//2, w//2
yy, xx = np.indices((h, w))
rr = np.hypot(yy - cy, xx - cx)
r = rr.astype(np.int32)

radial_power = np.bincount(r.ravel(), ps.ravel()) / np.maximum(1, np.bincount(r.ravel()))
freq_bins = np.arange(radial_power.size) / (max(h, w) / 2.0)  # normalized spatial frequency (0..~1)

# Save radial power to CSV for inspection
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_csv = Path(f"/mnt/data/radial_power_{ts}.csv")
pd.DataFrame({"normalized_frequency": freq_bins, "radial_power": radial_power}).to_csv(out_csv, index=False)

# Print a short summary
print({
    "min": vmin,
    "max": vmax,
    "mean": vmean,
    "std": vstd,
    "peak_y": peak_y,
    "peak_x": peak_x,
    "peak_value": peak_val,
    "grad_mean": gmean,
    "grad_std": gstd,
    "grad_max": gmax,
    "grad_max_y": gmax_y,
    "grad_max_x": gmax_x,
    "radial_power_csv": str(out_csv),
})