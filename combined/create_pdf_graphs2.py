count=0
import matplotlib.pyplot as plt
# thresh_file1=open("best_threshold.txt")
# best_thresh=0
# for line in thresh_file1:
# 	best_thresh=float(line.rstrip())
test_score=False
res_array=[]
res_array_1=[]
for line in open("yago_new.log"):
	if "Test scores" in line:
		test_score=True
		continue
	elif test_score==True:
		line1=(line.replace("("," ").rstrip().split(" "))
		# res_array.append(float(line1[9]))
		res_array.append(float(line1[6]))
		test_score=False
	# if float(line1[-1])<0.05:
	# 	count=count+1
for line in open("fb15k-237_remove_filter_neg.log"):
	if "Test scores" in line:
		test_score=True
		continue
	elif test_score==True:
		line1=(line.replace("("," ").rstrip().split(" "))
		# res_array.append(float(line1[9]))
		res_array_1.append(float(line1[6]))
		test_score=False
	# if float(line1[-1])<0.05:
	# 	count=count+1
res_array_2=[]
res_array_3=[]
for line in open("wn18rr_remove_filter_neg.log"):
	if "Test scores" in line:
		test_score=True
		continue
	elif test_score==True:
		line1=(line.replace("("," ").rstrip().split(" "))
		# res_array_2.append(float(line1[9]))
		res_array_2.append(float(line1[6]))
		test_score=False
	# if float(line1[-1])<0.05:
	# 	count=count+1
for line in open("nell_dev.log"):
	if "Test scores" in line:
		test_score=True
		continue
	elif test_score==True:
		line1=(line.replace("("," ").rstrip().split(" "))
		# res_array_2.append(float(line1[9]))
		res_array_3.append(float(line1[6]))
		test_score=False
	# if float(line1[-1])<0.05:
	# 	count=count+1
# res_array=[1623293,1853873,2005056,2083521,2083521,2135104,2300102,2450100,2532063,2656483,2750682]
plt.ylim(0.65,1)
plt.plot([1,2,3,4,5,6],res_array[:6],color='r',label="TypeE-ComplEx for yago", marker='o')
plt.plot([1,2,3,4,5,6],res_array_1[:6],color='b',label="TypeE-ComplEx for fb15k-237", marker='o')
plt.plot([1,2,3,4,5,6],res_array_2[:6],color='g',label="TypeE-ComplEx for wn18rr", marker='o')
plt.plot([1,2,3,4,5,6],res_array_3[:6],color='y',label="TypeE-ComplEx for nell", marker='o')
plt.legend()
plt.savefig("pdf_graph1.png")
plt.show()
test_score=False
res_array=[]
res_array_1=[]
for line in open("yago_ConvE_new.log"):
	if "Test scores" in line:
		test_score=True
		continue
	elif test_score==True:
		line1=(line.replace("("," ").rstrip().split(" "))
		# res_array.append(float(line1[9]))
		res_array.append(float(line1[6]))
		test_score=False
	# if float(line1[-1])<0.05:
	# 	count=count+1
for line in open("fb15k-237_ConvE.log"):
	if "Test scores" in line:
		test_score=True
		continue
	elif test_score==True:
		line1=(line.replace("("," ").rstrip().split(" "))
		# res_array.append(float(line1[9]))
		res_array_1.append(float(line1[6]))
		test_score=False
	# if float(line1[-1])<0.05:
	# 	count=count+1
res_array_2=[]
res_array_3=[]
for line in open("wn18rr_ConvE.log"):
	if "Test scores" in line:
		test_score=True
		continue
	elif test_score==True:
		line1=(line.replace("("," ").rstrip().split(" "))
		# res_array_2.append(float(line1[9]))
		res_array_2.append(float(line1[6]))
		test_score=False
	# if float(line1[-1])<0.05:
	# 	count=count+1
for line in open("nell_ConvE_dev.log"):
	if "Test scores" in line:
		test_score=True
		continue
	elif test_score==True:
		line1=(line.replace("("," ").rstrip().split(" "))
		# res_array_2.append(float(line1[9]))
		res_array_3.append(float(line1[6]))
		test_score=False
	# if float(line1[-1])<0.05:
	# 	count=count+1
# res_array=[1623293,1853873,2005056,2083521,2083521,2135104,2300102,2450100,2532063,2656483,2750682]
plt.ylim(0.65,1)
plt.plot([1,2,3,4,5,6],res_array[:6],color='r',label="TypeE-ConvE for yago", marker='o')
plt.plot([1,2,3,4,5,6],res_array_1[:6],color='b',label="TypeE-ConvE for fb15k-237", marker='o')
plt.plot([1,2,3,4,5,6],res_array_2[:6],color='g',label="TypeE-ConvE for wn18rr", marker='o')
plt.plot([1,2,3,4,5,6],res_array_3[:6],color='y',label="TypeE-ConvE for nell", marker='o')
plt.legend()
plt.savefig("pdf_graph2.png")
plt.show()