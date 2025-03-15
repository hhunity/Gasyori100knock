import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# template matching
def rgb2y(img: np.ndarray):
    imgy = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return imgy

def getGxGy(grray: np.ndarray):
    H, W = grray.shape

    gx = np.zeros([H,W],dtype=np.float32)
    gy = np.zeros([H,W],dtype=np.float32)

    for y in range(H):
        for x in range(W):
            gx[y,x] = grray[y,min(x+1,W-1)] - grray[y,max(x-1,0)]
            gy[y,x] = grray[min(y+1,H-1),x] - grray[max(y-1,0),x]

    return gx,gy

def getMagAng(gx: np.ndarray,gy :np.ndarray):
    H, W = gx.shape

    mag = ang = np.zeros([H,W],dtype=np.float32)

    mag = np.sqrt( np.where(gx**2 + gy**2 >= 0, gx**2 + gy**2 , 0))
    # ang = np.arctan(np.where( gx != 0, gy/gx,0 ))
    ang = np.arctan(gy/gx)
    # ang = np.arctan2(gx,gy)
    ang = np.degrees(ang)

    ang[ang<0] += 180
    
    return mag,ang

def convang(mag:np.ndarray):
    
    H, W = mag.shape

    out = np.zeros([H,W],dtype=np.uint8)

    for i in range(20,200,20):
        out[(mag[...]< i) & (mag[...] > i-20)] = (i-20)/20

    return out

# get gradient histogram
def gradient_histogram(gradient_quantized, magnitude, N=8):
    # get shape
    H, W = magnitude.shape

    # get cell num
    cell_N_H = H // N
    cell_N_W = W // N
    histogram = np.zeros((cell_N_H, cell_N_W, 9), dtype=np.float32)

    # each pixel
    for y in range(cell_N_H):
        for x in range(cell_N_W):
            for j in range(N):
                for i in range(N):
                    histogram[y, x, gradient_quantized[y * 4 + j, x * 4 + i]] += magnitude[y * 4 + j, x * 4 + i]

    return histogram
    
def HOG_step1(img):
    # get original image shape
    grray = rgb2y(img)

    H, W = grray.shape
    
    gx,gy = getGxGy(grray)
    mag,ang = getMagAng(gx,gy)
    qang  = convang(ang)

    return mag,qang

def HOG_step2(magnitude,gradient_quantized):
 
    hist = gradient_histogram(gradient_quantized,magnitude)

    return hist

# Read image
start_time = time.time()

img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_61_70/imori.jpg").astype(np.float32)

# get HOG step1
magnitude, gradient_quantized = HOG_step1(img)


# Write gradient magnitude to file
_magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)

cv2.imwrite("out_mag.jpg", _magnitude)

# Write gradient angle to file
H, W, C = img.shape
out = np.zeros((H, W, 3), dtype=np.uint8)

# define color
C = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
     [127, 127, 0], [127, 0, 127], [0, 127, 127]]

# draw color
for i in range(9):
     out[gradient_quantized == i] = C[i]


cv2.imwrite("out_gra.jpg", out)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"処理時間: {elapsed_time} 秒")
                
# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
