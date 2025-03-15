import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# template matching
def saisenka(img):
    # get original image shape
    H, W, C = img.shape
    
    out = img.copy()

    # prepare temporary image
    tmp = np.zeros((H, W), dtype=np.uint8)

    # binarize
    tmp[img[..., 0] > 0] = 1

    # out = np.zeros((H, W, 3), dtype=np.uint8)

    count = 1
    while(count >= 1 ):
        count = 0
        for y in range(H):
            for x in range(W):

                x0 = tmp[y,x]
            
                if x0 == 0:
                    continue

                x1 = tmp[y,min(x+1,W-1)]
                x2 = tmp[max(y-1,0),min(x+1,W-1)]
                x3 = tmp[max(y-1,0),x]
                x4 = tmp[max(y-1,0),max(x-1,0)]
                x5 = tmp[y,max(x-1,0)]
                x6 = tmp[min(y+1,H-1),max(x-1,0)]
                x7 = tmp[min(y+1,H-1),x]
                x8 = tmp[min(y+1,H-1),min(x+1,W-1)]

                # S = (x1-x1*x2*x3) + (x3-x3*x4*x5) + (x5 - x5*x6*x7) + (x7 -x7*x8*x1)
                S = (x1-x1*x2*x3) + (x3-x3*x4*x5) + (x5 - x5*x6*x7) + (x7 -x7*x8*x1)

                if x1 + x3 + x4 +x7 >= 0:
                    if S == 1:
                        if x1 + x2 + x3 + x4 +x5 + x6 + x7 + x8 >= 3:
                            out[y,x,0] = 0
                            out[y,x,1] = 0
                            out[y,x,2] = 0
                            count+=1
        # tmp = np.zeros((H, W), dtype=np.uint8)
        tmp[out[...,0]==0] = 0

    
    # out = np.clip(out,0,255).astype(np.uint8)

    return out

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
