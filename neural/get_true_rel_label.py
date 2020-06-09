import sys
valid_labels=[float(line.rstrip()) for line in open(sys.argv[2])]
test_labels=[float(line.rstrip()) for line in open(sys.argv[3])]
count=0
file=open(sys.argv[4],"w")
for line in open(sys.argv[1]+"valid.txt"):
	if valid_labels[count]==1.0:
		line1=line.rstrip().split("\t")
		file.write("\t".join(line1[:3])+"\n")
	count=count+1
count=0
for line in open(sys.argv[1]+"test.txt"):
	if test_labels[count]==1.0:
		line1=line.rstrip().split("\t")
		file.write("\t".join(line1[:3])+"\n")
	count=count+1
