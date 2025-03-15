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

def k_mean_color_quantitie(img:np.ndarray,K=5):

	H,W,ch=img.shape

	img = img.reshape((H*W,ch))
	out = np.zeros([H*W],dtype=np.uint8)

	#step 1
	np.random.seed(0)

	choice = np.array(np.random.choice(np.arange(H*W),K,replace=False))

	choice_color = img[choice,:]
	choice_color = np.sort(choice_color)
	
	b_comp = False

	si=0
	while(True):
		#step 2
		for i in range(H*W):
			dis = np.sqrt(np.sum(np.square(np.tile(img[i,],(K,1))-choice_color),axis=1))

			index = np.argsort(dis)

			out[i] = index[0]

		if b_comp:
			break

		#step 3
		new_choice = np.zeros_like(choice_color)

		for i in range(K):
			new_choice[i] = np.average(img[np.where(out==i)[0]],axis=0)

		# new_choice = np.sort(new_choice)

		#step 4
		if np.all(choice_color == new_choice):
			b_comp = True
		spinner = "|/-\\"
		print(f"\rProcessing {spinner[si % len(spinner)]}", end="", flush=True)
		si+=1
		choice_color = new_choice
	print("")
	print(choice_color)
	# out = out * 50
	# out = out.reshape((H,W))

	for i in range(K):
		img[np.where(out==i)] = choice_color[i]
	
	img = img.reshape([H,W,3])

	return np.clip(img,0,255).astype(np.uint8)

# Read image
start_time = time.time()

img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_91_100/madara.jpg").astype(np.float32)
# img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_Answer/IMG_0217.JPG").astype(np.float32)
# img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_Answer/IMG_0217.JPG").astype(np.float32)

out = k_mean_color_quantitie(img,K=10)

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
