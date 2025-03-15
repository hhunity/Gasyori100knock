import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# template matching

# BGR -> HSV
def BGR2HSV(_img):
	img = _img.copy() / 255.

	hsv = np.zeros_like(img, dtype=np.float32)

	# get max and min
	max_v = np.max(img, axis=2).copy()
	min_v = np.min(img, axis=2).copy()
	min_arg = np.argmin(img, axis=2)

	# H
	hsv[..., 0][np.where(max_v == min_v)]= 0
	## if min == B
	ind = np.where(min_arg == 0)
	hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
	## if min == R
	ind = np.where(min_arg == 2)
	hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180
	## if min == G
	ind = np.where(min_arg == 1)
	hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300
		
	# S
	hsv[..., 1] = max_v.copy() - min_v.copy()

	# V
	hsv[..., 2] = max_v.copy()
	
	return hsv

def color_tracking(img : np.ndarray):

		H,W,_ = img.shape

		hsv = BGR2HSV(img)
    
		mask = np.zeros([H,W,3],dtype=np.uint8)

		mask[...] = 1

		mask[ (hsv[...,0] >= 180) & (hsv[...,0] <= 260) ] = 0
		
		out = np.zeros_like(img,dtype=np.uint8)

		out = img.astype(np.uint8) * mask
		out = np.multiply(img.astype(np.uint8),mask)

		return out


# Read image
start_time = time.time()

img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_61_70/imori.jpg").astype(np.float32)

out = color_tracking(img)

# cv2.imwrite("out.jpg", out)

# write histogram to file
# for i in range(9):
#     plt.subplot(3,3,i+1)
#     plt.imshow(hist[..., i])
#     plt.axis('off')
#     plt.xticks(color="None")
#     plt.yticks(color="None")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"処理時間: {elapsed_time} 秒")
# plt.show()
# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
