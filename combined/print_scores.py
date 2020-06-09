import sys
from sklearn.metrics import f1_score
cat_labels = [int(float(i.rstrip())) for i in open('predictions/valid_cat_labels.txt')]
rel_labels = [int(float(i.rstrip())) for i in open('predictions/valid_rel_labels.txt')]
psl_kgi_cats = [int(float(i.rstrip())) for i in open('predictions/psl-kgi_valid_cat_preds.txt')]
psl_kgi_rels = [int(float(i.rstrip())) for i in open('predictions/psl-kgi_valid_rel_preds.txt')]
typed_rels = [int(float(i.rstrip())) for i in open('predictions/typed_valid_rel_preds.txt')]

print('Valid scores :\nF1 %0.2f wF1 %0.2f label wF1 %0.2f relation wF1 %0.2f PSL-KGI rel wF1 %0.2f '%(f1_score(cat_labels+rel_labels,psl_kgi_cats+typed_rels),f1_score(cat_labels+rel_labels,psl_kgi_cats+typed_rels,average='weighted'),f1_score(cat_labels,psl_kgi_cats,average='weighted'),f1_score(rel_labels,typed_rels,average='weighted'),f1_score(rel_labels,psl_kgi_rels,average='weighted')))

cat_labels = [int(float(i.rstrip())) for i in open('predictions/test_cat_labels.txt')]
rel_labels = [int(float(i.rstrip())) for i in open('predictions/test_rel_labels.txt')]
psl_kgi_cats = [int(float(i.rstrip())) for i in open('predictions/psl-kgi_test_cat_preds.txt')]
psl_kgi_rels = [int(float(i.rstrip())) for i in open('predictions/psl-kgi_test_rel_preds.txt')]
typed_rels = [int(float(i.rstrip())) for i in open('predictions/typed_test_rel_preds.txt')]

try:
	org_test = set([tuple(i.rstrip().split()) for i in open(sys.argv[2])])
	org_test_indices = set([])
	for i,line in enumerate(open(sys.argv[1])):
		if tuple(line.rstrip().split()[:-1]) in org_test:
			org_test_indices.add(i)
	org_rel_wf1 = f1_score([rel_labels[i] for i in range(len(rel_labels)) if i in org_test_indices],[typed_rels[i] for i in range(len(rel_labels)) if i in org_test_indices],average='weighted')
	org_psl_rel_wf1 = f1_score([rel_labels[i] for i in range(len(rel_labels)) if i in org_test_indices],[psl_kgi_rels[i] for i in range(len(rel_labels)) if i in org_test_indices],average='weighted')
except :
	org_rel_wf1 = 0
	org_psl_rel_wf1 = 0
print('Test scores :\nF1 %0.2f F1(PSL) %0.2f wF1 %0.2f wF1(PSL) %0.2f label wF1 %0.2f relation wF1 %0.2f(%0.2f) PSL-KGI rel wF1 %0.2f(%0.2f) '%(f1_score(cat_labels+rel_labels,psl_kgi_cats+typed_rels),f1_score(cat_labels+rel_labels,psl_kgi_cats+psl_kgi_rels),f1_score(cat_labels+rel_labels,psl_kgi_cats+typed_rels,average='weighted'),f1_score(cat_labels+rel_labels,psl_kgi_cats+psl_kgi_rels,average='weighted'),f1_score(cat_labels,psl_kgi_cats,average='weighted'),f1_score(rel_labels,typed_rels,average='weighted'),org_rel_wf1,f1_score(rel_labels,psl_kgi_rels,average='weighted'),org_psl_rel_wf1))
