import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# template matching

def reketu8(org):
    c = 0

    for i in range(1,9,1):
        xn = org
        xn[i] = 1
        S = (xn[1]-np.prod(xn[1:4])) + (xn[3]-np.prod(xn[3:6])) + (xn[5] - np.prod(xn[5:8])) + (xn[7] -xn[7]*xn[8]*xn[1])
        if S == 1:
            c+=1

    if c == 8:
        return True
    else:
        return False

def saisenka(img):
    # get original image shape
    H, W, C = img.shape
    
    out = img.copy()

    # prepare temporary image
    tmp = np.zeros((H, W), dtype=np.int8)

    # binarize
    tmp[img[..., 0] > 0] = 1

    xn = np.zeros(9,dtype=np.int8)

    # out = np.zeros((H, W, 3), dtype=np.uint8)

    count = 1
    while(count >= 1 ):
        count = 0
        for y in range(H):
            for x in range(W):

                xn[0] = 1- tmp[y,x]
            
                if xn[0] == 1 or xn[0] == 2:
                    continue

                xn[1] = 1 - tmp[y,min(x+1,W-1)]
                xn[2] = 1 - tmp[max(y-1,0),min(x+1,W-1)]
                xn[3] = 1 - tmp[max(y-1,0),x]
                xn[4] = 1 - tmp[max(y-1,0),max(x-1,0)]
                xn[5] = 1 - tmp[y,max(x-1,0)]
                xn[6] = 1 - tmp[min(y+1,H-1),max(x-1,0)]
                xn[7] = 1 - tmp[min(y+1,H-1),x]
                xn[8] = 1 - tmp[min(y+1,H-1),min(x+1,W-1)]

                # S = (x1-x1*x2*x3) + (x3-x3*x4*x5) + (x5 - x5*x6*x7) + (x7 -x7*x8*x1)
                S = (xn[1]-np.prod(xn[1:4])) + (xn[3]-np.prod(xn[3:6])) + (xn[5] - np.prod(xn[5:8])) + (xn[7] -xn[7]*xn[8]*xn[1])

                if xn[1] == 1 or xn[3] == 1 or xn[4] == 1 or xn[7] == 1:
                    if S == 1:
                        if np.sum( abs(1- xn[1:9]) )>= 2:
                            if np.any((1- xn[1:9]) == 1):
                                 if np.all((1-xn[1:9]) != -1) or reketu8(xn):
                                     tmp[y,x]=-1
                                     count+=1
        tmp[tmp[...]==-1] = 0
    
    # out = np.clip(out,0,255).astype(np.uint8)

    return tmp.astype(np.uint8) * 255

# Read image
start_time = time.time()

img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_61_70/gazo.png").astype(np.float32)
# img2 = cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_61_70/thorino.jpg").astype(np.float32)

# Read templete image

# Template matching
out = saisenka(img)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"処理時間: {elapsed_time} 秒")
                
# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
