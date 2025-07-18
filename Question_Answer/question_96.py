import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import pickle
from glob import glob

np.random.seed(0)

class NN:
    def __init__(self, ind=2, w=64, w2=64, outd=1, lr=0.1):
        self.w2 = np.random.randn(ind, w)
        self.b2 = np.random.randn(w)
        self.w3 = np.random.randn(w, w2)
        self.b3 = np.random.randn(w2)
        self.wout = np.random.randn(w2, outd)
        self.bout = np.random.randn(outd)
        self.lr = lr

    def forward(self, x):
        self.z1 = x
        self.z2 = self.sigmoid(np.dot(self.z1, self.w2) + self.b2)
        self.z3 = self.sigmoid(np.dot(self.z2, self.w3) + self.b3)
        self.out = self.sigmoid(np.dot(self.z3, self.wout) + self.bout)
        return self.out

    def train(self, x, t):
        # backpropagation output layer
        # self.out * (1 - self.out)はシグモンドレイヤの逆伝搬
        # 2*(self.out - t) は２乗誤差の逆伝搬
        out_d = 2*(self.out - t) * self.out * (1 - self.out)
        #affineの逆伝搬
        out_dW = np.dot(self.z3.T, out_d)
        #バイアスの逆伝搬
        out_dB = np.dot(np.ones([1, out_d.shape[0]]), out_d)
        # out_dB = np.sum(out_d,axis=0)
        self.wout -= self.lr * out_dW
        self.bout -= self.lr * out_dB[0]

        #np.dot(out_d, self.wout.T) は出力層の誤差を１つ前の層に伝搬してる
        w3_d = np.dot(out_d, self.wout.T) * self.z3 * (1 - self.z3)
        w3_dW = np.dot(self.z2.T, w3_d)
        w3_dB = np.dot(np.ones([1, w3_d.shape[0]]), w3_d)
        self.w3 -= self.lr * w3_dW
        self.b3 -= self.lr * w3_dB[0]
        
        # backpropagation inter layer
        w2_d = np.dot(w3_d, self.w3.T) * self.z2 * (1 - self.z2)
        w2_dW = np.dot(self.z1.T, w2_d)
        w2_dB = np.dot(np.ones([1, w2_d.shape[0]]), w2_d)
        self.w2 -= self.lr * w2_dW
        self.b2 -= self.lr * w2_dB[0]
        
    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

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

if  __name__ == "__main__":

    # Read image
    start_time = time.time()

    # read image
    img = cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_91_100/imori_1.jpg").astype(np.float32)
    H, W, C = img.shape

    # Grayscale
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]

    # イモリの正解位置
    gt = np.array((47, 41, 129, 103), dtype=np.float32)

    Crop_num = 200
    L        = 60
    H_size   = 32
    crop_wh  = 60
    F_n      =  ((H_size // 8) ** 2) * 9

    db  = np.zeros((Crop_num, F_n+1))

    for i in range(Crop_num):
        #ランダムクロップ
        x1= np.random.randint(W-crop_wh)
        y1= np.random.randint(H-crop_wh)

        b = np.array((x1,y1,x1+crop_wh,y1+crop_wh),dtype=np.int32)
        lou = intersection_over_union(gt,b)
        label = 0
        #GTと比較してラベルを張る
        if(lou >= 0.5):
            cv2.rectangle(img,tuple(b[0:2]),tuple(b[2:4]),[0,0,255],1)
            label = 1
        else:
            cv2.rectangle(img,tuple(b[0:2]),tuple(b[2:4]),[255,0,0],1)
            label = 0

        #リサイズ
        crop      = gray[b[1]:b[3],b[0]:b[2]]
        res       = resize(crop,H_size,H_size)
        mag,qang  = HOG_step1(res)
        step2     = HOG_step2(mag,qang)
        step3     = HOG_step3(step2)
        #1次元データにしてラベルを最後に格納
        db[i,:F_n]= step3.ravel()
        db[i,-1] = label


    hh = NN(F_n,64,64,1,0.01)

    epch = 10000

    corect_answer = np.zeros([epch],dtype=np.float32)

    for j in range(epch):
        temp = 0
        answer = hh.forward(db[...,:F_n])
        hh.train(db[i,F_n],db[...,-1].reshape(-1,1))
        
        answer = answer.ravel()
        answer[answer[...]>=0.5] = 1
        answer[answer[...]< 0.5] = 0

        # print(answer)

        corect_answer[j] = np.sum(answer[...]==db[...,-1])

        # print(f"Accuracy:{corect_answer[j]}")

    print(f"Accuracy >> {corect_answer[-1]/Crop_num} ({corect_answer[-1]}/{Crop_num})")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"処理時間: {elapsed_time} 秒")

    try:
        with open("/Users/hiroyukih/vscode/Gasyori100knock/Question_Answer/NN.pkl","wb") as file:
            pickle.dump(hh,file)
    except Exception as e:
        print(f"pklファイルの書き込みエラー{e}")

    plt.plot(np.arange(epch),corect_answer/Crop_num)
    plt.xlabel("epch")
    plt.ylabel("Accuracy")
    plt.show()

    # Save result
    # cv2.imwrite("out.jpg", out)
    # cv2.imshow("result", np.clip(img,0,255).astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()