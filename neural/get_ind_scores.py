from sklearn.metrics import classification_report
import sys
line_array=[]
for line in open(sys.argv[1]):
	line1=line.rstrip().split("\t")
	line_array.append(line1[-1])
line1_array=[]
for line1 in open(sys.argv[2]):
	line1_array.append(line1.rstrip())
print(classification_report(line_array,line1_array))
# predictions = [(1 if i>=best_threshold else 0) for i in predictions]
# 	if not dname=='':
# 		with open(dname,'w') as f:
# 			for sc in predictions: f.write('%d\n'%(sc))
# print('f1 score %0.2f | wf1 score %0.2f | precision score %0.2f | recall score %0.2f | label wf1 %0.2f | relation wf1 %0.2f '%(f1_score(labels,predictions),f1_score(labels,predictions,average='weighted'),precision_score(labels,predictions,average='weighted'),recall_score(labels,predictions,average='weighted'),f1_score(cat_labels,cat_preds,average='weighted'),f1_score(rel_labels,rel_preds,average='weighted')))
# if len(org_test_indices)!=0:
# 	print('f1 score on original test set %0.2f',f1_score([labels[i] for i in range(len(labels)) if i not in org_test_indices],[predictions[i] for i in range(len(predictions)) if i not in org_test_indices]))
# return best_threshold,best_wf1