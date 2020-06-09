import re

import sys

if len(sys.argv)<4:
	print('usage : python3 out cat_predicts.txt rel_predicts.txt')

file = open(sys.argv[1])
cat_file = open(sys.argv[2],'w')
rel_file = open(sys.argv[3],'w')
data = ''.join(file.readlines())

cat_labels = re.findall('CAT\(\\d+, \\d+\) Truth=\[.*\]',data)
rel_labels = re.findall('REL\(\\d+, \\d+, \\d+\) Truth=\[.*\]',data)

def get_tuple(i):
	return re.findall("\d*\.\d+|\d+",i)

cat_labels = [get_tuple(i) for i in cat_labels]
rel_labels = [get_tuple(i) for i in rel_labels]

for i in cat_labels:
	cat_file.write('\t'.join(i)+'\n')

for i in rel_labels:
	rel_file.write('\t'.join(i)+'\n')