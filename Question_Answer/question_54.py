import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# template matching
def Template_matching(img, template):
    # get original image shape
    H, W, C = img.shape

    # get template image shape
    Ht, Wt, Ct = template.shape

    # Templete matching
    # prepare x, y index
    i, j = -1, -1
    # prepare evaluate value
    v = 0
    

    for y in range(H - Ht):
        for x in range(W - Wt):
            # get SSD value
            _v = np.sum(img[y : y + Ht, x : x + Wt] * template)
            # devided numerator
            _v /= (np.sqrt(np.sum(img[y : y + Ht, x : x + Wt] ** 2)) * np.sqrt(np.sum(template ** 2)))

            # if SSD is min
            if _v > v:
                v = _v
                i, j = x, y
    
    out = img.copy()
    # draw rectangle
    cv2.rectangle(out, pt1=(i, j), pt2=(i+Wt, j+Ht), color=(0,0,255), thickness=2)
    out = out.astype(np.uint8)

    return out

# Read image
start_time = time.time()

img = cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_Answer/imori.jpg").astype(np.float32)

# Read templete image
template = cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_Answer/imori_part.jpg").astype(np.float32)

# Template matching
out = Template_matching(img, template)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"処理時間: {elapsed_time} 秒")
                
# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
