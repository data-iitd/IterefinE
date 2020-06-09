# format - python3 evaluator.py rel_predicts data/label-test-uniq-raw-rel.db.TRAIN cat_predicts data/label-test-uniq-raw-cat.db.TRAIN

import sys


rel_predicts = {}

cat_predicts = {}

from sklearn.metrics import f1_score,precision_score,recall_score

for line in open(sys.argv[1]):
	a,b,c,d = line.rstrip().split()
	a,b,c,d = int(a),int(b),int(c),float(d)
	rel_predicts[(a,b,c)]=d
	
for line in open(sys.argv[3]):
	a,b,d = line.rstrip().split()
	a,b,d = int(a),int(b),float(d)
	cat_predicts[(a,b)]=d


predictions = []
labels = []
rel_predictions = []
cat_predictions = []
cat_labels = []
rel_labels = []

for line in open(sys.argv[5]):
	a,b,c,d = line.rstrip().split()
	a,b,c,d = int(a),int(b),int(c),int(d)
	predictions.append(rel_predicts[(a,b,c)])
	labels.append(d)
	rel_labels.append(d)
	rel_predictions.append(predictions[-1])
for line in open(sys.argv[6]):
	a,b,d = line.rstrip().split()
	a,b,d = int(a),int(b),int(d)
	predictions.append(cat_predicts[(a,b)])
	labels.append(d)
	cat_labels.append(d)
	cat_predictions.append(predictions[-1])

best_threshold = 0.0
best_score = 0.0
thresholds = [i/100 for i in range(5,100,5)]

for threshold in thresholds:
	sc = f1_score(labels,[(1 if i>=threshold else 0) for i in predictions],average='weighted')	
	if sc>best_score:
		best_score = sc
		best_threshold = threshold

threshold = best_threshold

# dump valid scores 
with open('predictions/psl-kgi_valid_cat_preds.txt','w') as f:
	for i in cat_predictions: f.write('%d\n'%(1 if i>=threshold else 0))	
with open('predictions/psl-kgi_valid_rel_preds.txt','w') as f:
	for i in rel_predictions: f.write('%d\n'%(1 if i>=threshold else 0))	
with open('predictions/valid_cat_labels.txt','w') as f:
	for i in cat_labels: f.write('%d\n'%(i))	
with open('predictions/valid_rel_labels.txt','w') as f:
	for i in rel_labels: f.write('%d\n'%(i))	

predictions = []
labels = []
rel_predictions = []
cat_predictions = []
cat_labels = []
rel_labels = []

for line in open(sys.argv[2]):
	a,b,c,d = line.rstrip().split()
	a,b,c,d = int(a),int(b),int(c),int(d)
	predictions.append(rel_predicts[(a,b,c)])
	labels.append(d)
	rel_labels.append(d)
	rel_predictions.append(predictions[-1])
for line in open(sys.argv[4]):
	a,b,d = line.rstrip().split()
	a,b,d = int(a),int(b),int(d)
	predictions.append(cat_predicts[(a,b)])
	labels.append(d)
	cat_labels.append(d)
	cat_predictions.append(predictions[-1])

# dump test scores
with open('predictions/psl-kgi_test_cat_preds.txt','w') as f:
	for i in cat_predictions: f.write('%d\n'%(1 if i>=threshold else 0))	
with open('predictions/psl-kgi_test_rel_preds.txt','w') as f:
	for i in rel_predictions: f.write('%d\n'%(1 if i>=threshold else 0))	

with open('predictions/test_cat_labels.txt','w') as f:
	for i in cat_labels: f.write('%d\n'%(i))	
with open('predictions/test_rel_labels.txt','w') as f:
	for i in rel_labels: f.write('%d\n'%(i))	


names = dict([i.rstrip().split() for i in open('data/names.txt')])
# update NN_data/train.txt with true label predicted relation facts
f2 = open('NN_data/targets.txt','w')
with open('NN_data/train.txt','w') as f:
	for a,b,c in rel_predicts:
		if rel_predicts[(a,b,c)]>=threshold: f.write('%s\t%s\t%s\n'%(names[str(a)],names[str(c)],names[str(b)]))
		f2.write('%s\t%s\t%s\n'%(names[str(a)],names[str(c)],names[str(b)]))
f2.close()
f1 = open('NN_data/test.txt','w')
f2 = open('NN_data/test_labels.txt','w')
for line in open(sys.argv[2]):
	a,b,c,d = line.rstrip().split()	
	f1.write('%s\t%s\t%s\n'%(names[a],names[c],names[b]))
	f2.write(d+'\n')
f1.close()
f2.close()

f1 = open('NN_data/valid.txt','w')
f2 = open('NN_data/valid_labels.txt','w')
for line in open(sys.argv[5]):
	a,b,c,d = line.rstrip().split()	
	f1.write('%s\t%s\t%s\n'%(names[a],names[c],names[b]))
	f2.write(d+'\n')
f1.close()
f2.close()