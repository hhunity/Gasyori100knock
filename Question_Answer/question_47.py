import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# template matching

# BGR -> HSV
def RGB2Y(img : np.ndarray):
	H,W,_ = img.shape

	imgY = np.zeros([H,W],dtype=np.float32)

	imgY = 0.2126 * img[...,0] + 0.7152 * img[...,1] + 0.0722 * img[...,2]

	return imgY.clip(0,255).astype(np.uint8)

def ootsu2th(imgY:np.ndarray):

	H,V = imgY.shape

	SbMax = 0
	tMax = 0

	for t in range(1,255):
		v0 = imgY[imgY[...]>= t]
		v1 = imgY[imgY[...]<  t]
		m0 = np.mean(v0) if len(v0) > 0 else 0.
		m1 = np.mean(v1) if len(v1) > 0 else 0.
		w0 = len(v0)/(H*V)
		w1 = len(v1)/(H*V)

		Sb = w0*w1*((m0-m1)**2)

		if Sb > SbMax:
			SbMax = Sb
			tMax  = t


	imgY[imgY[...]>=tMax] = 255
	imgY[imgY[...]< tMax] = 0

	return imgY

def morophology_Dilation(imgY : np.ndarray,n=1):
	
	H,W = imgY.shape
	fil = np.array([[0,1,0], [1,0,1], [0,1,0]])

	for _ in range(n):
		temp = np.pad(imgY,(1,1),mode="edge")
		for y in range(1,H+1):
			for x in range(1,W+1):

				if temp[y,x]!= 0:
					continue

				filout = np.sum(temp[y-1:y+2,x-1:x+2]*fil[...])

				if filout >= 255:
					imgY[y-1,x-1] = 255
	
	return imgY

def morophology_Erosion(imgY : np.ndarray,n=1):
	
	H,W = imgY.shape
	fil = np.array([[0,1,0], [1,0,1], [0,1,0]])

	for _ in range(n):
		temp = np.pad(imgY,(1,1),mode="edge")
		for y in range(1,H+1):
			for x in range(1,W+1):

				if temp[y,x]!= 255:
					continue

				filout = np.sum(temp[y-1:y+2,x-1:x+2]*fil[...])

				if filout < 255*4:
					imgY[y-1,x-1] = 0
	
	return imgY

def closing(imgY,N=1):
		
		out = morophology_Dilation(imgY,N)
		out = morophology_Erosion(out,N)	
		return out

def opening(imgY,N=1):
		
		out = morophology_Erosion(imgY,N)
		out = morophology_Dilation(out,N)	
		return out

def morophology(img : np.ndarray):

		H,W,_ = img.shape

		imgY = RGB2Y(img)

		imgY = ootsu2th(imgY)

		imgY = closing(imgY,1)

		return imgY


# Read image
start_time = time.time()

img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_61_70/imori.jpg").astype(np.float32)

out = morophology(img)

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
