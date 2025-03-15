import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# template matching
def blend(img,img2):
    # get original image shape
    H, W, C = img.shape
    
    out = img.copy()

    out = img * 0.5 + img2 *(1.0-0.5)
    
    out = np.clip(out,0,255).astype(np.uint8)

    return out

# Read image
start_time = time.time()

img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_51_60/imori.jpg").astype(np.float32)
img2 = cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_51_60/thorino.jpg").astype(np.float32)

# Read templete image

# Template matching
out = blend(img,img2)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"処理時間: {elapsed_time} 秒")
                
# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
