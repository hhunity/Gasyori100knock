import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# template matching
def classtaring(img):
    # get original image shape
    H, W, C = img.shape

    # prepare label tempolary image
    label = np.zeros((H, W), dtype=np.int32)
    label[img[..., 0]>0] = 1

    # look up table
    LUT = [0 for _ in range(H*W)]
    
    n = 1

    for y in range(H):
        for x in range(W):

            if label[y, x] == 0:
                continue

            # get above pixel
            c2 = label[max(y-1,0),max(x-1,0)]
            c3 = label[max(y-1,0), x]
            c4 = label[max(y-1,0),min(x+1,W-1)]
            c5 = label[y, max(x-1,0)]

            # if not labeled
            if c3 < 2 and c5 < 2 and c2 < 2 and c4 < 2:
                # labeling
                n += 1
                label[y, x] = n
            else:
                # replace min label index
                _vs = [c2 ,c3,c4, c5]
                vs = [a for a in _vs if a > 1]
                v = min(vs)
                label[y, x] = v
                
                minv = v
                for _v in vs:
                    if LUT[_v] != 0:
                        minv = min(minv, LUT[_v])
                for _v in vs:
                    LUT[_v] = minv

    count = 1

    # for i, x in  enumerate(LUT):
    #     if x != 0:
    #         print(f"Lut1:{i},{x}")
    
    # integrate index of look up table
    for l in range(2, n+1):
        flag = True
        for i in range(n+1):
            if LUT[i] == l:
                if flag:
                    count += 1
                    flag = False
                LUT[i] = count

    # for i, x in  enumerate(LUT):
    #     if x != 0:
    #         print(f"Lut2:{i},{x}")

    # draw color
    COLORS = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]]
    out = np.zeros((H, W, C), dtype=np.uint8)

    for i, lut in enumerate(LUT[2:]):
        out[label == (i+2)] = COLORS[lut-2]

    return out

# Read image
start_time = time.time()

img = cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_51_60/seg.png").astype(np.float32)

# Read templete image

# Template matching
out = classtaring(img)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"処理時間: {elapsed_time} 秒")
                
# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
