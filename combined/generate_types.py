# python3 generate_types.py cat_predicts.txt names.txt subclass.txt domain.txt range.txt out_types.txt

import sys
sys.setrecursionlimit(10**5)
names = dict([i.rstrip().split() for i in open(sys.argv[2])])
domains = dict([i.rstrip().split() for i in open(sys.argv[4])])
ranges = dict([i.rstrip().split() for i in open(sys.argv[5])])
subclass = dict([i.rstrip().split() for i in open(sys.argv[3])])
types = set(list(domains.values())+list(ranges.values()))
graph = {i:[] for i in types}
for i in subclass:
    if i not in graph: graph[i] = [subclass[i]]
    else: graph[i].append(subclass[i])
def depth(i,seen):
    res = 0
    seen.add(i)
    if i not in graph: return res
    for j in graph[i]: 
        if j not in seen:
            res= max(res,depth(j,seen)+1)
    return res

depths = {}
for i in graph:
    seen = set([])
    depths[i] = depth(i,seen)

all_types = {}
for line in open(sys.argv[1]):
    a,b,d = line.rstrip().split()
    if b not in depths:
            depths[b]=0
    if a not in all_types:
            if float(d)>=0.5: all_types[a]=b
    else:
            #print(b,all_types[a])
            if float(d)>=0.5 and depths[b]>depths[all_types[a]]: all_types[a]=b

with open('NN_data/types.txt','w') as f:
    for a in all_types:
        f.write('%s\t%s\n'%(names[a],names[all_types[a]]))

with open('NN_data/domain.txt','w') as f:
    for a in domains:
        f.write('%s\t%s\n'%(names[a],names[domains[a]]))        

with open('NN_data/range.txt','w') as f:
    for a in ranges:
        f.write('%s\t%s\n'%(names[a],names[ranges[a]]))        