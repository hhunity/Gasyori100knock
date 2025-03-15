import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
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

	database = np.zeros([10,13],np.int32)
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

def judge_class(database:np.ndarray,train_list:np.ndarray):
	
	d = 256/4

	class_name = ["akahara","madara"]

	absolute_pass = "/Users/hiroyukih/vscode/Gasyori100knock/Question_81_90/dataset/"

	test = glob(absolute_pass+"test*")
	test.sort()

	accuracy_N = len(test)
	accuracy = 0

	for i, path in enumerate(test):
		img  =  cv2.imread(path).astype(np.float32)
		
		img = color_quantization(img,d)

		hist = make_histgram(img,d)

		min_feature = img.shape[0]*img.shape[1]*3
		pred_i   = 0

		# for j,p in enumerate(train_list):
		# 	feature = np.sum( np.abs(database[j][:12] - hist) )
		# 	# print(f"{p}:{feature},{min_feature}")
		# 	if min_feature >= feature:
		# 		min_feature = feature
		# 		pred_i      = j

		# get histogram difference
		difs = np.abs(database[:, :12] - hist)
		# print(f"{difs}")
		difs = np.sum(difs, axis=1)

		# get argmin of difference
		sort_index = np.argsort(difs)

		pred_i  = sort_index[:3]

		class0n =  np.sum(database[pred_i][-1] == 0)
		class1n =  np.sum(database[pred_i][-1] == 1)

		if class0n > class1n:
			similar_class = class_name[0]
		else:
			similar_class = class_name[1]

		name = path[len(absolute_pass):]
		similar_name = [train_list[i] for i in pred_i]
		# similar_pred = database[pred_i][-1]
		
		print(f"{name} is similar >> {similar_name} Pred >> {similar_class}")

		gt = "akahara" if name in "akahara" else "madara"

		if gt == similar_class:
			accuracy+=1


	accuracy = accuracy_N / len(test)
	print("Accuracy >>", accuracy, "({}/{})".format(int(accuracy_N), len(test)))

# Read image
start_time = time.time()

# img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_81_90/thorino.jpg").astype(np.float32)
# img  =  cv2.imread("/Users/hiroyukih/vscode/Gasyori100knock/Question_Answer/IMG_0217.JPG").astype(np.float32)

database,filename_list = make_database()

judge_class(database,filename_list)

# cv2.imwrite("out.jpg", out)

# write histogram to file
plt.rcParams.update({'font.size': 4})

for i in range(database.shape[0]):
		plt.subplot(2,5,i+1)
		# plt.imshow(out[i], cmap='gray')
		plt.bar(np.arange(12),database[i][0:12])
		# plt.axis('off')
		plt.title(filename_list[i])
		# plt.xticks(color="None")
		# plt.yticks(color="None")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"処理時間: {elapsed_time} 秒")
plt.show()
# Save result
# cv2.imwrite("out.jpg", out)
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
