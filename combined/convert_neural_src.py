# usage python3 convert_neural_src.py train.txt train_preds.txt names.txt out_name


import sys
import numpy as np

names = {}
for line in open(sys.argv[3]):
	a,b = line.rstrip().split()
	names[b]=a

thresh_file1=open("type_complex/best_threshold.txt")
best_thresh=0
for line in thresh_file1:
	best_thresh=float(line.rstrip())
train_preds = [float(i) for i in open(sys.argv[2])]
poses = [train_preds[i] for i in range(len(train_preds)) if train_preds[i]>best_thresh]
negs = [train_preds[i] for i in range(len(train_preds)) if train_preds[i]<best_thresh]
p1=np.mean(poses)+(0.5*np.std(poses))
n1=np.mean(negs)-(0.75*np.std(negs))
# import random
# random.shuffle(poses)
# random.shuffle(negs)
print(p1)
print(n1)
to_add = set() #set(poses[:20*1000]+negs[:20*1000])
cnt = 0 
added = set()
with open(sys.argv[4],'w') as f:
	for line in open(sys.argv[1]):
		a,b,c= line.rstrip().split()
		# print(names[a])
		# print(names[b])
		# print(names[c]) 
		if  True : #or (cnt in to_add and (a,b,c) not in added): 
			# print(cnt)
			# print(train_preds[cnt])
			if ((train_preds[cnt]<p1) and (train_preds[cnt]>n1)):
				cnt+=1
				continue
			# print(train_preds[cnt])
			# print(cnt)
			f.write('%s\t%s\t%s\t%f\n'%(names[a],names[c],names[b],train_preds[cnt]))

			#added.add((a,b,c))
		cnt+=1
