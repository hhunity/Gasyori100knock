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
# gaussian filtering
def gaussian_filtering(I, K_size=3, sigma=3):
	# get shape
	H, W = I.shape

	## gaussian
	I_t = np.pad(I, (K_size // 2, K_size // 2), 'edge')

	# gaussian kernel
	K = np.zeros((K_size, K_size), dtype=np.float32)
	for x in range(K_size):
		for y in range(K_size):
			_x = x - K_size // 2
			_y = y - K_size // 2
			K[y, x] = np.exp( -(_x ** 2 + _y ** 2) / (2 * (sigma ** 2)))
	K /= (sigma * np.sqrt(2 * np.pi))
	K /= K.sum()

	# filtering
	for y in range(H):
		for x in range(W):
			I[y,x] = np.sum(I_t[y : y + K_size, x : x + K_size] * K)
			
	return I

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

	gaussian_filtering(ixx,3,3)
	gaussian_filtering(iyy,3,3)
	gaussian_filtering(ixy,3,3)

	ixx=(ixx-ixx.min())*255/(ixx.max()-ixx.min())
	iyy=(iyy-iyy.min())*255/(iyy.max()-iyy.min())
	ixy=(ixy-ixy.min())*255/(ixy.max()-ixy.min())

	ixx=np.clip(ixx,0,255).astype(np.uint8)
	iyy=np.clip(iyy,0,255).astype(np.uint8)
	ixy=np.clip(ixy,0,255).astype(np.uint8)

	out = [ixx,iyy,ixy]
	title = ["iX^2","iy^2","ixy"]
	return out,title

# Read image
start_time = time.time()

img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_81_90/thorino.jpg").astype(np.float32)
# img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_Answer/IMG_0217.JPG").astype(np.float32)

out,title = hesian(img)

# cv2.imwrite("out.jpg", out)

# write histogram to file
for i in range(len(out)):
		plt.subplot(1,3,i+1)
		plt.imshow(out[i], cmap='gray')
		plt.axis('off')
		plt.title(title[i])
		plt.xticks(color="None")
		plt.yticks(color="None")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"処理時間: {elapsed_time} 秒")
plt.show()
# Save result
# cv2.imwrite("out.jpg", out)
# cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
