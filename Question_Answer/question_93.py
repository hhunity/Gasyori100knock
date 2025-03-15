import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

from glob import glob

# template matching
def RGB2Y(img : np.ndarray):
	H,W,_ = img.shape

	imgY = np.zeros([H,W],dtype=np.float32)

	imgY = 0.2126 * img[...,0] + 0.7152 * img[...,1] + 0.0722 * img[...,2]

	return imgY.clip(0,255).astype(np.uint8)

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

def cropping(img:np.ndarray,N=200):

	H,W,_  = img.shape
	crop_wh= 60

	np.random.seed(0)

	gt    = np.array((47, 41, 129, 103), dtype=np.float32)

	for i in range(200):
		x1= np.random.randint(W-crop_wh)
		y1= np.random.randint(H-crop_wh)

		b = np.array((x1,y1,x1+crop_wh,y1+crop_wh),dtype=np.float32)
		lou = intersection_over_union(gt,b)

		if(lou > 0.5):
			color = [0,0,255]
		else:
			color = [255,0,0]

		b = b.astype(np.int32)	
		cv2.rectangle(img,b[0:2],b[2:4],color,1)

	gt = gt.astype(np.int32)
	cv2.rectangle(img,gt[0:2],gt[2:4],[0,255,0],1)

	return np.clip(img,0,255).astype(np.uint8)

# Read image
start_time = time.time()

img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_91_100/imori_1.jpg").astype(np.float32)
# img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_Answer/IMG_0217.JPG").astype(np.float32)
# img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_Answer/IMG_0217.JPG").astype(np.float32)

out = cropping(img)

# cv2.imwrite("out.jpg", out)

# write histogram to file
# plt.rcParams.update({'font.size': 4})

# for i in range(database.shape[0]):
# 		plt.subplot(2,5,i+1)
# 		# plt.imshow(out[i], cmap='gray')
# 		plt.bar(np.arange(12),database[i][0:12])
# 		# plt.axis('off')
# 		plt.title(filename_list[i])
# 		# plt.xticks(color="None")
# 		# plt.yticks(color="None")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"処理時間: {elapsed_time} 秒")
# plt.show()
# Save result
# cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
