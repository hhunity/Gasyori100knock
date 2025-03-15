import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# template matching
def RGB2Y(img : np.ndarray):
	H,W,_ = img.shape

	imgY = np.zeros([H,W],dtype=np.float32)

	imgY = 0.2126 * img[...,0] + 0.7152 * img[...,1] + 0.0722 * img[...,2]

	return imgY.clip(0,255).astype(np.uint8)

# def make_carbo_filter(img: np.ndarray,K=111,s=10,g=1.2,l=10,p=0,A=0):
def make_carbo_filter( K=111, s=10, g=1.2, l=10, p=0, A=0):

		d = K // 2

		out = np.zeros([K,K],dtype=np.float32)

		for y in range(K):
			for x in range(K):
				py = y - d
				px = x - d

				theta = A / 180. * np.pi

				xx =  np.cos(theta) * px + np.sin(theta)*py
				yy = -np.sin(theta) * px + np.cos(theta)*py

				out[y,x] =   np.exp(-((xx**2)+(g**2)*(yy**2))/(2 * (s**2))) * np.cos(2*np.pi*xx/l+p)

		out /= np.sum(np.abs(out))
		# out = out - np.min(out)
		# out /= np.max(out)
		# out *= 255

		return out

def make_carbo_filter_set(K=111, s=10, g=1.2, l=10, p=0):

		filter_set = list()

		# out = make_carbo_filter(A=0)
		filter_set.append(make_carbo_filter(K,s,g,l,p,A=0))
		filter_set.append(make_carbo_filter(K,s,g,l,p,A=45))
		filter_set.append(make_carbo_filter(K,s,g,l,p,A=90))
		filter_set.append(make_carbo_filter(K,s,g,l,p,A=135))

		return filter_set

def sobalFilter(imgY :np.ndarray,fh:bool=True):

		H,W, = imgY.shape

		tmp = np.pad(imgY,(1,1),mode="edge")
		out = np.zeros_like(imgY,dtype=np.float32)

		fil =  np.array([[1,2,1],[0,0,0],[-1,-2,-1]]) if fh else np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

		for y in range(1,H+1):
			for x in range(1,W+1):
				out[y-1,x-1] = np.sum(tmp[y-1:y+2,x-1:x+2]*fil[...])


		return out

# def exec_hesian(imgY:np.ndarray,Ix:np.ndarray,Iy:np.ndarray):

	

def hesian(img:np.ndarray):

	H,W,_ = img.shape

	imgY = RGB2Y(img)

	iy  = sobalFilter(imgY,True)
	ix  = sobalFilter(imgY,False)

	detH = np.zeros([H,W],dtype=np.float32)

	out = np.array((imgY, imgY, imgY))
	out = np.transpose(out, (1,2,0))

	# out  = np.zeros([H,W,3],dtype=np.uint8)
	# out[...,0] = imgY
	# out[...,1] = imgY
	# out[...,2] = imgY

	ixx = ix ** 2
	iyy = iy ** 2
	ixy = ix * iy

	for y in range(H):
		for x in range(W):
			detH[y,x] = (ixx[y,x]) * (iyy[y,x]) - (ixy[y,x]**2)

	#256*256かけないと上手くいかない。一画素づつやれば上手く行く
	ixx = ixx*256*256
	iyy = iyy*256*256
	ixy = ixy*256*256
	ixx = ixx.astype(np.int32)
	iyy = iyy.astype(np.int32)
	ixy = ixy.astype(np.int32)

	detH= (ixx * iyy) - (ixy*ixy)

	detH=detH/256/256

	for y in range(1,H-1):
		for x in range(1,W-1):
			if np.max(detH[y-1:y+2,x-1:x+2]) == detH[y,x] and detH[y, x] > np.max(detH) * 0.1:
				out[y,x] = [0,0,255]

	return np.clip(out,0,255).astype(np.uint8)

# Read image
start_time = time.time()

img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_81_90/thorino.jpg").astype(np.float32)
# img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_Answer/IMG_0217.JPG").astype(np.float32)

out = hesian(img)

# cv2.imwrite("out.jpg", out)

# write histogram to file
# for i in range(len(fil_list)):
# 		plt.subplot(1,4,i+1)
# 		plt.imshow(fil_list[i], cmap='gray')
# 		plt.axis('off')
# 		plt.title("Angle "+str(i*45))
# 		plt.xticks(color="None")
# 		plt.yticks(color="None")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"処理時間: {elapsed_time} 秒")
# plt.show()
# Save result
# cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
