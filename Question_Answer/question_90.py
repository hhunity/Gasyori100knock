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

def color_quantization(img:np.ndarray,d):

	img = np.floor(img/d)
	img = img*d

	return img

def make_histgram(img:np.ndarray,d):

	hist = np.zeros([12],dtype=np.int32)
	tmp  = np.clip(img,0,255).astype(np.uint8)

	for i in range( int(len(hist)/3) ):
		hist[i]   = np.sum(tmp[...,0]==(i*d))
		hist[i+4] = np.sum(tmp[...,1]==(i*d))
		hist[i+8] = np.sum(tmp[...,2]==(i*d))

	return hist
	
def make_database():

	d = 256/4

	# filename_list  = [f"train_akahara_{i}.jpg" for i in range(1,6)]
	# filename_list += [f"train_madara_{i}.jpg" for i in range(1,6)]

	absolute_pass = "/Users/hiroyukih/vscode/Gasyori100knock/Question_81_90/dataset/"

	train = glob(absolute_pass+"train_*")
	train.sort()

	database = np.zeros([len(train),13],np.int32)
	file_list= list()

	for i, path in enumerate(train):
		img  =  cv2.imread(path).astype(np.float32)
		
		img = color_quantization(img,d)

		hist = make_histgram(img,d)

		# get class
		if 'akahara' in path:
			cls = 0
		elif 'madara' in path:
			cls = 1

		# store class label
		database[i, -1]   = cls
		database[i][0:12] = hist[...]

		file_list.append(path[len(absolute_pass):])

	return database,file_list

def k_mean_step1(database,flist, Class=2,seed=1,th=0.5):

	d = 256/4

	database = db.copy()
	gs			 = np.zeros([Class,12],dtype=np.float32)

	np.random.seed(seed)

	for i in range(len(database)):
		database[i,-1]  = 0 if np.random.random() < th else 1

	# tmp = database[:,:12]
	# gs[0,:] = np.mean(tmp[database[:,-1]==0].T, axis=1)
	# gs[1,:] = np.mean(tmp[database[:,-1]==1].T, axis=1)

	for i in range(Class):
		gs[i] = np.mean(database[np.where(database[..., -1] == i)[0], :12], axis=0)

	# print("assied label")
	
	for a in database:
		print(f"{a}")

	print("Grabity")
	print(f"{gs}")

	return database,gs


def k_mean_step2(database,flist,Class=2):

	bchange = True

	gs			 = np.zeros([Class,12],dtype=np.float32)

	print("###k_mean_step2###")

	while(bchange):

		bchange = False
		
		# compute gravity
		for i in range(Class):
			gs[i] = np.mean(database[np.where(database[..., -1] == i)[0], :12], axis=0)
		
		# re labering
		for i in range(len(database)):
			
			org_class = database[i,-1]
			r 	= np.sqrt(np.sum( np.square(np.abs(database[i,:12] - gs)), axis=1))
			# dis = np.sqrt(np.sum(np.square(np.abs(gs - feats[i, :12])), axis=1))

			# get new label
			pred = np.argmin(r, axis=0)
			
			if int(org_class) != pred:
				database[i,-1] = pred
				bchange = True
				print(f"changeClass r={r} {org_class}->{pred} ")

	for i,a in enumerate(flist):
		print(f"{a}:{database[i,-1]}")

	return

db,flist = make_database()

db,gs = k_mean_step1(db,flist,seed=4,th=0.3)
k_mean_step2(db,flist)

# Read image
start_time = time.time()

# img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_81_90/thorino.jpg").astype(np.float32)
# img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_Answer/IMG_0217.JPG").astype(np.float32)

# database,filename_list = make_database()

# judge_class(database,filename_list)

# cv2.imwrite("out.jpg", out)

# write histogram to file
plt.rcParams.update({'font.size': 4})

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
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
