import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import pickle
import os
from question_96 import NN

from glob import glob

np.random.seed(0)

def intersection_over_union(a,b):

	area_a = (a[2] - a[0]) *  (a[3] - a[1]) 
	area_b = (b[2] - b[0]) *  (b[3] - b[1]) 

	left_x = max(a[0],b[0])
	righ_x = min(a[2],b[2])
	up_y   = max(a[1],b[1])
	bott_y = min(a[3],b[3])

	Rol    = (righ_x-left_x)*(bott_y-up_y)

	IoU    = (Rol)/(area_a + area_b - Rol)

	return IoU

def intersection_over_union(a,b):

	area_a = (a[2] - a[0]) *  (a[3] - a[1]) 
	area_b = (b[2] - b[0]) *  (b[3] - b[1]) 

	left_x = max(a[0],b[0])
	righ_x = min(a[2],b[2])
	up_y   = max(a[1],b[1])
	bott_y = min(a[3],b[3])

	Rol    = (righ_x-left_x)*(bott_y-up_y)

	IoU    = (Rol)/(area_a + area_b - Rol)

	return IoU

def resize(img, h, w):
    _h, _w  = img.shape
    ah = 1. * h / _h
    aw = 1. * w / _w
    y = np.arange(h).repeat(w).reshape(w, -1)
    x = np.tile(np.arange(w), (h, 1))
    y = (y / ah)
    x = (x / aw)

    ix = np.floor(x).astype(np.int32)
    iy = np.floor(y).astype(np.int32)
    ix = np.minimum(ix, _w-2)
    iy = np.minimum(iy, _h-2)

    dx = x - ix
    dy = y - iy
    
    out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]
    out[out>255] = 255

    return out

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
    
def HOG_step1(grray):
    # get original image shape
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

# Read image
start_time = time.time()

# read image
img = cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_91_100/imori_many.jpg").astype(np.float32)
H, W, C = img.shape

try:
    with open("/Users/hiroyukih/vscode/Gasyori100knock/Question_Answer/NN.pkl","rb") as file:
        hh:NN = pickle.load(file)
except Exception as e:
    print(f"NNファイルが見つかりません:{e}")
    exit(1)

# Grayscale
gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]

# イモリの正解位置
# gt = np.array((47, 41, 129, 103), dtype=np.float32)
H_size   = 32
Sride    = 4
F_n      =  ((H_size // 8) ** 2) * 9
recs     = np.array(((42,42),(56,56),(70,70)),dtype=np.float32)
img_num  = int( (H*W)/(4*4)*recs.shape[0] )

# db       = np.zeros((img_num, F_n))
db       = np.zeros((img_num, F_n))
index    = 0
correct_list = list()

for y in range(0,H,4):
    for x in range(0,W,4):
        for rec in recs:
            crop_w = rec[0]
            crop_h = rec[1]

            #クロップ開始位置と終了位置
            x1=  int(max(x - crop_w/2,0))
            y1=  int(max(y - crop_h/2,0))
            x2=  int(min(x + crop_w/2,W-1))
            y2=  int(min(y + crop_h/2,H-1))            

            #リサイズ
            crop      = gray[y1:y2,x1:x2]
            res       = resize(crop,H_size,H_size)
            mag,qang  = HOG_step1(res)
            step2     = HOG_step2(mag,qang)
            step3     = HOG_step3(step2)
            #NN
            score = hh.forward(step3.ravel())

            if score[0] > 0.7:
                cv2.rectangle(img,(x1,y1),(x2,y2),[0,0,255],1)
                correct_list.append([x1,y1,x2,y2,score[0]])

            #1次元データにしてラベルを最後に格納
            # db[index,:F_n]= step3.ravel()
            # index+=1

for a in correct_list:
    print(a)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"処理時間: {elapsed_time} 秒")

# plt.plot(np.arange(epch),corect_answer/Crop_num)
# plt.xlabel("epch")
# plt.ylabel("Accuracy")
# plt.show()

# Save result
# cv2.imwrite("out.jpg", out)
cv2.imshow("result", np.clip(img,0,255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()