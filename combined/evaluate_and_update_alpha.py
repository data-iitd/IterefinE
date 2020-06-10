# format - python3 evaluator.py rel_predicts data/label-test-uniq-raw-rel.db.TRAIN cat_predicts data/label-test-uniq-raw-cat.db.TRAIN

import sys

import numpy as np


rel_predicts = {}

cat_predicts = {}

from sklearn.metrics import f1_score,precision_score,recall_score,classification_report


names = dict([i.rstrip().split() for i in open(sys.argv[9])])

for line in open(sys.argv[1]):
	a,b,c,d = line.rstrip().split()
	a,b,c,d = int(a),int(b),int(c),float(d)
	rel_predicts[(a,b,c)]=d
	
for line in open(sys.argv[3]):
	a,b,d = line.rstrip().split()
	a,b,d = int(a),int(b),float(d)
	cat_predicts[(a,b)]=d

# print(rel_predicts)
predictions = {}
labels = {}
rel_predictions = []
cat_predictions = []
cat_labels = []
rel_labels = []

for line in open(sys.argv[5]):
	a,b,c,d = line.rstrip().split()
	a,b,c,d = int(a),int(b),int(c),int(d)
	# print(rel_predicts[(a,b,c)])
	if (names[str(a)],names[str(b)],names[str(c)]) not in predictions:
		predictions[(names[str(a)],names[str(b)],names[str(c)])]=[]
	predictions[(names[str(a)],names[str(b)],names[str(c)])].append(rel_predicts[(a,b,c)])
	if (names[str(a)],names[str(b)],names[str(c)]) not in labels:
		labels[(names[str(a)],names[str(b)],names[str(c)])]=[]
	labels[(names[str(a)],names[str(b)],names[str(c)])].append(d)
	rel_labels.append(d)
	rel_predictions.append(rel_predicts[(a,b,c)])
for line in open(sys.argv[6]):
	a,b,d = line.rstrip().split()
	a,b,d = int(a),int(b),int(d)
	if (names[str(a)],names[str(b)]) not in predictions:
		predictions[(names[str(a)],names[str(b)])]=[]
	predictions[(names[str(a)],names[str(b)])].append(cat_predicts[(a,b)])
	if (names[str(a)],names[str(b)]) not in labels:
		labels[(names[str(a)],names[str(b)])]=[]
	labels[(names[str(a)],names[str(b)])].append(d)
	# predictions.append(cat_predicts[(a,b)])
	# labels.append(d)
	cat_labels.append(d)
	cat_predictions.append(cat_predicts[(a,b)])

complex_predictions={}
neural_valid_file_name=sys.argv[7]
neural_valid_pred_name=sys.argv[8]
neural_valid_prediction=[float(line.rstrip()) for line in open(neural_valid_pred_name)]
neural_valid_names={}
count=0
for line in open(neural_valid_file_name):
	line1=line.rstrip().split("\t")
	print(line1)
	if line1[1]=="has_label":
		if (line1[0],line1[2]) not in neural_valid_names:
			neural_valid_names[(line1[0],line1[2])]=[]
		neural_valid_names[(line1[0],line1[2])].append(neural_valid_prediction[count])
	else:
		if (line1[0],line1[2],line1[1]) not in neural_valid_names:
			neural_valid_names[(line1[0],line1[2],line1[1])]=[]
		neural_valid_names[(line1[0],line1[2],line1[1])].append(neural_valid_prediction[count])
	count=count+1
best_threshold = 0.0
best_alpha=0.0
best_score = 0.0
thresholds = [i/100 for i in range(5,100,5)]
alphas = [i/100 for i in range(0,110,10)]

final_labels=[]
final_predictions=[]
final_neural_predictions=[]
for k in labels: 
	for t in range(len(labels[k])):
		final_labels.append(labels[k][t])
		final_predictions.append(predictions[k][t])
		final_neural_predictions.append(neural_valid_names[k][t])
print(len(final_predictions))
print(len(final_labels))
print(len(final_neural_predictions))

final_predictions=np.array(final_predictions)
final_neural_predictions=np.array(final_neural_predictions)
print(final_predictions[0])
print(final_neural_predictions[0])
for threshold in thresholds:
	for alpha in alphas:
		predictions= alpha*final_predictions+((1-alpha)*final_neural_predictions)
		sc = f1_score(final_labels,[(1 if i>=threshold else 0) for i in predictions],average='weighted')	
		if sc>best_score:
			best_score = sc
			best_threshold = threshold
			best_alpha = alpha
			print(sc)
			print(threshold)
			print(alpha)


predictions = {}
labels = {}
rel_predictions = []
cat_predictions = []
cat_labels = []
rel_labels = []

for line in open(sys.argv[2]):
	a,b,c,d = line.rstrip().split()
	a,b,c,d = int(a),int(b),int(c),int(d)
	# print(rel_predicts[(a,b,c)])
	if (names[str(a)],names[str(b)],names[str(c)]) not in predictions:
		predictions[(names[str(a)],names[str(b)],names[str(c)])]=[]
	predictions[(names[str(a)],names[str(b)],names[str(c)])].append(rel_predicts[(a,b,c)])
	if (names[str(a)],names[str(b)],names[str(c)]) not in labels:
		labels[(names[str(a)],names[str(b)],names[str(c)])]=[]
	labels[(names[str(a)],names[str(b)],names[str(c)])].append(d)
	rel_labels.append(d)
	rel_predictions.append(rel_predicts[(a,b,c)])
for line in open(sys.argv[4]):
	a,b,d = line.rstrip().split()
	a,b,d = int(a),int(b),int(d)
	if (names[str(a)],names[str(b)]) not in predictions:
		predictions[(names[str(a)],names[str(b)])]=[]
	predictions[(names[str(a)],names[str(b)])].append(cat_predicts[(a,b)])
	if (names[str(a)],names[str(b)]) not in labels:
		labels[(names[str(a)],names[str(b)])]=[]
	labels[(names[str(a)],names[str(b)])].append(d)
	# predictions.append(cat_predicts[(a,b)])
	# labels.append(d)
	cat_labels.append(d)
	cat_predictions.append(cat_predicts[(a,b)])

complex_predictions={}
neural_valid_file_name=sys.argv[10]
neural_valid_pred_name=sys.argv[11]
neural_valid_prediction=[float(line.rstrip()) for line in open(neural_valid_pred_name)]
neural_valid_names={}
count=0
for line in open(neural_valid_file_name):
	line1=line.rstrip().split("\t")
	print(line1)
	if line1[1]=="has_label":
		if (line1[0],line1[2]) not in neural_valid_names:
			neural_valid_names[(line1[0],line1[2])]=[]
		neural_valid_names[(line1[0],line1[2])].append(neural_valid_prediction[count])
	else:
		if (line1[0],line1[2],line1[1]) not in neural_valid_names:
			neural_valid_names[(line1[0],line1[2],line1[1])]=[]
		neural_valid_names[(line1[0],line1[2],line1[1])].append(neural_valid_prediction[count])
	count=count+1

final_labels=[]
final_predictions=[]
final_neural_predictions=[]
for k in labels: 
	for t in range(len(labels[k])):
		final_labels.append(labels[k][t])
		final_predictions.append(predictions[k][t])
		final_neural_predictions.append(neural_valid_names[k][t])
final_predictions=np.array(final_predictions)
final_neural_predictions=np.array(final_neural_predictions)
threshold = best_threshold
alpha = best_alpha
# print(alpha*final_predictions)
# print((1-alpha)*final_neural_predictions)
predictions= alpha*final_predictions+((1-alpha)*final_neural_predictions)
sc = f1_score(final_labels,[(1 if i>=threshold else 0) for i in predictions],average='weighted')	
print(sc)
print(classification_report(final_labels,[(1 if i>=threshold else 0) for i in predictions]))