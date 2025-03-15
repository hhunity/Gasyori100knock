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
    tmp = np.zeros((H, W), dtype=np.int8)

    # binarize
    tmp[img[..., 0] == 0] = 1

    xn = np.zeros(10,dtype=np.int8)

    s = list()
    # out = np.zeros((H, W, 3), dtype=np.uint8)

    count = 1
    while(count >= 1 ):
        count = 0
        s = list()

        for y in range(H):
            for x in range(W):

                xn[1] = tmp[y,x]
                xn[2] = tmp[max(y-1,0),x]
                xn[3] = tmp[max(y-1,0),min(x+1,W-1)]
                xn[4] = tmp[y,min(x+1,W-1)]
                xn[5] = tmp[min(y+1,H-1),min(x+1,W-1)]
                xn[6] = tmp[min(y+1,H-1),x]
                xn[7] = tmp[min(y+1,H-1),max(x-1,0)]
                xn[8] = tmp[y,max(x-1,0)]
                xn[9] = tmp[max(y-1,0),max(x-1,0)]

                judge = 0

                # 1-1
                if xn[1] == 0:
                    judge+=1
                
                # 1-2
                t = 0
                for i in range(2,9,1):
                    if xn[i+1]==1 and xn[i] ==0 :
                        t+=1
                
                if xn[2]==1 and xn[9] ==0 :
                    t+=1

                if t == 1:
                    judge+=1

                t = np.sum(xn[2:10]==1)
                if t >=2 and t <= 6:
                    judge+=1

                if np.any( xn[[2,4,6]] == 1):
                    judge+=1

                if np.any( xn[[4,6,8]] == 1):
                    judge+=1

                if judge == 5:
                    s.append([y,x])
                    count+=1

        for xx in s:
            tmp[xx[0],xx[1]] = 1

        s = list()

        for y in range(H):
            for x in range(W):

                xn[1] = tmp[y,x]
                xn[2] = tmp[max(y-1,0),x]
                xn[3] = tmp[max(y-1,0),min(x+1,W-1)]
                xn[4] = tmp[y,min(x+1,W-1)]
                xn[5] = tmp[min(y+1,H-1),min(x+1,W-1)]
                xn[6] = tmp[min(y+1,H-1),x]
                xn[7] = tmp[min(y+1,H-1),max(x-1,0)]
                xn[8] = tmp[y,max(x-1,0)]
                xn[9] = tmp[max(y-1,0),max(x-1,0)]

                judge = 0

                # 1-1
                if xn[1] == 0:
                    judge+=1
                
                # 1-2
                t = 0
                for i in range(2,9,1):
                    if xn[i+1]==1 and xn[i] ==0 :
                        t+=1

                if xn[2]==1 and xn[9] ==0 :
                    t+=1
                    
                if t == 1:
                    judge+=1

                t = np.sum(xn[2:10]==1)
                if t >=2 and t <= 6:
                    judge+=1

                if np.any( xn[[2,4,8]] == 1):
                    judge+=1

                if np.any( xn[[2,6,8]] == 1):
                    judge+=1

                if judge == 5:
                    s.append([y,x])
                    count+=1

        for xx in s:
            tmp[xx[0],xx[1]] = 1

    out = 1 - tmp 
    out = out.astype(np.uint8) * 255

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
