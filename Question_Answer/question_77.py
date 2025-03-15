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
		out = out - np.min(out)
		out /= np.max(out)
		out *= 255

		return np.clip(out,0,255).astype(np.uint8)

def make_carbo_filter_set(img):

		filter_set = list()

		# out = make_carbo_filter(A=0)
		filter_set.append(make_carbo_filter(A=0))
		filter_set.append(make_carbo_filter(A=45))
		filter_set.append(make_carbo_filter(A=90))
		filter_set.append(make_carbo_filter(A=135))

		return filter_set

# Read image
start_time = time.time()

img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_61_70/imori.jpg").astype(np.float32)

out = make_carbo_filter_set(img)

# cv2.imwrite("out.jpg", out)

# write histogram to file
for i in range(len(out)):
		plt.subplot(1,len(out),i+1)
		plt.imshow(out[i], cmap='gray')
		plt.axis('off')
		plt.title("Angle "+str(i*45))
		plt.xticks(color="None")
		plt.yticks(color="None")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"処理時間: {elapsed_time} 秒")
plt.show()
# Save result
# cv2.imwrite("out.jpg", out)
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
