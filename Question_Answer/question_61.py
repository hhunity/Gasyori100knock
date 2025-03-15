import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# template matching
def renketu(img):
    # get original image shape
    H, W, C = img.shape
    
    out = img.copy()

    # prepare temporary image
    tmp = np.zeros((H, W), dtype=np.uint8)

    # binarize
    tmp[img[..., 0] > 0] = 1

    out = np.zeros((H, W, 3), dtype=np.uint8)

    for y in range(H):
        for x in range(W):

            x0 = tmp[y,x]
           
            if x0 == 0:
                continue

            x1 = 1 - tmp[y,min(x+1,W-1)]
            x2 = 1 - tmp[max(y-1,0),min(x+1,W-1)]
            x3 = 1 - tmp[max(y-1,0),x]
            x4 = 1 - tmp[max(y-1,0),max(x-1,0)]
            x5 = 1 - tmp[y,max(x-1,0)]
            x6 = 1 - tmp[min(y+1,H-1),max(x-1,0)]
            x7 = 1 - tmp[min(y+1,H-1),x]
            x8 = 1 - tmp[min(y+1,H-1),min(x+1,W-1)]

            # S = (x1-x1*x2*x3) + (x3-x3*x4*x5) + (x5 - x5*x6*x7) + (x7 -x7*x8*x1)
            S = (x1-x1*x2*x3) + (x3-x3*x4*x5) + (x5 - x5*x6*x7) + (x7 -x7*x8*x1)

            if S == 0:
                out[y,x] = [0, 0, 255]
            elif S == 1:
                out[y,x] = [0, 255, 0]
            elif S == 2:
                out[y,x] = [255, 0, 0]
            elif S == 3:
                out[y,x] = [255, 255, 0]
            elif S == 4:
                out[y,x] = [255, 0, 255]
            
    
    # out = np.clip(out,0,255).astype(np.uint8)

    return out

# Read image
start_time = time.time()

img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_61_70/renketsu.png").astype(np.float32)
# img2 = cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_61_70/thorino.jpg").astype(np.float32)

# Read templete image

# Template matching
out = renketu(img)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"処理時間: {elapsed_time} 秒")
                
# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
