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

def bilinear(imgY :np.ndarray,mag):

	H,W = imgY.shape

	oH = int(H*mag)
	oW = int(W*mag)

	out = np.zeros([oH,oW],dtype=np.float16)
	tmp = imgY.copy()

	for y in range(oH):
		for x in range(oW):
			iy = int(y/mag)
			ix = int(x/mag)

			a = (x-ix*mag)/mag
			b = (y-iy*mag)/mag

			tmp1 = (1-a)*tmp[iy,ix] + a*tmp[iy,min(ix+1,W-1)]
			tmp2 = (1-a)*tmp[min(iy+1,H-1),ix] + a*tmp[min(iy+1,H-1),min(ix+1,W-1)]

			out[y,x] = (1-b)*tmp1 + b*tmp2

	return out

# make image pyramid
def make_pyramid(gray,borg_size=False):
	# first element
	pyramid = [gray]
	# each scale
	for i in range(1, 6):
		# define scale
		a = 2. ** i

		# down scale
		p = bilinear(gray, 1./a)
		
		if borg_size:
			p = bilinear(p,a)
			# print(f"{gray.shape[1]}x{gray.shape[0]}->{p.shape[1]}x{p.shape[0]}")

		# add pyramid list
		pyramid.append(p)
		
	return pyramid

def mag_small(img : np.ndarray):

		H,W,_ = img.shape

		imgY = RGB2Y(img)
    
		outY = bilinear(imgY,0.5)
		outY = bilinear(outY,2.0)

		out  = np.abs(imgY - outY)
		max  = np.max(out)
		out  = out*255/max

		return out.clip(0,255).astype(np.uint8)

def make_gucian_piramid(img : np.ndarray):

		H,W,_ = img.shape

		imgY = RGB2Y(img)
    
		out  = make_pyramid(imgY,False)

		return out

def kentyoka_map(img: np.ndarray):

		H,W,_ = img.shape

		imgY = RGB2Y(img)
    
		out  = make_pyramid(imgY)



# Read image
start_time = time.time()

img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_61_70/imori.jpg").astype(np.float32)

out = make_gucian_piramid(img)

# cv2.imwrite("out.jpg", out)

# write histogram to file
for i in range(len(out)):
    plt.subplot(3,2,i+1)
    plt.imshow(out[i], cmap='gray')
    plt.axis('off')
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
