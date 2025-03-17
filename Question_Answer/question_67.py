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

def normalize(hist,epsilon=1):

    H, W, _ = hist.shape

    for y in range(H):
        for x in range(W):
            hist[y,x] = hist[y,x]  / np.sqrt( np.sum(hist[max(y - 1, 0) : min(y + 2, H),max(x - 1, 0) : min(x + 2, W)]** 2) + epsilon) 

    return hist
                    
def HOG_step3(hist):
 
    normalize(hist)

    return hist

# draw HOG
def draw_HOG(img, histogram):
    # Grayscale
    def BGR2GRAY(img):
        gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
        return gray

    def draw(gray, histogram, N=8):
        # get shape
        H, W = gray.shape
        cell_N_H, cell_N_W, _ = histogram.shape
        
        ## Draw
        out = gray[1 : H + 1, 1 : W + 1].copy().astype(np.uint8)

        for y in range(cell_N_H):
            for x in range(cell_N_W):
                cx = x * N + N // 2
                cy = y * N + N // 2
                x1 = cx + N // 2 - 1
                y1 = cy
                x2 = cx - N // 2 + 1
                y2 = cy
                
                h = histogram[y, x] / np.sum(histogram[y, x])
                h /= h.max()
        
                for c in range(9):
                    #angle = (20 * c + 10 - 90) / 180. * np.pi
                    # get angle
                    angle = (20 * c + 10) / 180. * np.pi
                    rx = int(np.sin(angle) * (x1 - cx) + np.cos(angle) * (y1 - cy) + cx)
                    ry = int(np.cos(angle) * (x1 - cx) - np.cos(angle) * (y1 - cy) + cy)
                    lx = int(np.sin(angle) * (x2 - cx) + np.cos(angle) * (y2 - cy) + cx)
                    ly = int(np.cos(angle) * (x2 - cx) - np.cos(angle) * (y2 - cy) + cy)

                    # color is HOG value
                    c = int(255. * h[c])

                    # draw line
                    cv2.line(out, (lx, ly), (rx, ry), (c, c, c), thickness=1)

        return out
    
    # get gray
    gray = BGR2GRAY(img)

    # draw HOG
    out = draw(gray, histogram)

    return out

# Read image
start_time = time.time()

img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_61_70/imori.jpg").astype(np.float32)

# get HOG step1
magnitude, gradient_quantized = HOG_step1(img)

hist = HOG_step2(magnitude,gradient_quantized)

hist = HOG_step3(hist)

# out  = draw_HOG(img,hist)

# cv2.imwrite("out.jpg", out)

# write histogram to file
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(hist[..., i])
    plt.axis('off')
    plt.xticks(color="None")
    plt.yticks(color="None")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"処理時間: {elapsed_time} 秒")
plt.show()
# Save result
# cv2.imwrite("out.jpg", out)
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
