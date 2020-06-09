import numpy
import torch


class kb(object):
    """
    Stores a knowledge base as an numpy array. Can be generated from a file. Also stores the entity/relation mappings
    (which is the mapping from entity names to entity id) and possibly entity type information.
    """
    def __init__(self, filename, em=None, rm=None, add_unknowns=True,domains = None,ranges = None,types = None):
        """
        Duh...
        :param filename: The file name to read the kb from
        :param em: Prebuilt entity map to be used. Can be None for a new map to be created
        :param rm: prebuilt relation map to be used. Same as em
        :param add_unknowns: Whether new entites are to be acknowledged or put as <UNK> token.
        """
        #print('loading ',filename)
        self.entity_map = {} if em is None else em
        self.relation_map = {} if rm is None else rm
        self.types = types
        if filename is None:
            return
        facts = []
        with open(filename) as f:
            lines = f.readlines()
            lines = [l.split() for l in lines]
            for l in lines:
                if(add_unknowns):
                    if(l[0] not in self.entity_map):
                        self.entity_map[l[0]] = len(self.entity_map)
                    if(l[2] not in self.entity_map):
                        self.entity_map[l[2]] = len(self.entity_map)
                    if(l[1] not in self.relation_map):
                        self.relation_map[l[1]] = len(self.relation_map)
                facts.append([self.entity_map.get(l[0], len(self.entity_map)-1), self.relation_map.get(l[1],
                        len(self.relation_map)-1), self.entity_map.get(l[2], len(self.entity_map)-1),(domains.get(l[1],0) if domains is not None else 0),(ranges.get(l[1],0) if ranges is not None else 0)])
                facts[-1].append(types[facts[-1][0]] if types is not None and (facts[-1][0] in types) else 0)
                facts[-1].append(types[facts[-1][2]] if types is not None and (facts[-1][2] in types) else 0)
            
        self.facts = numpy.array(facts, dtype='int64')

    def build_type_info(self,folder_name):
        # returns types,domains,ranges as no -> no thing
        # assuming data files are as follows
        # types.txt
        types = {}
        label_map = {'UNK':0}
        for line in open(folder_name+'/types.txt'):
            a,b = line.rstrip().split()
            if a not in self.entity_map: continue
            a = self.entity_map[a]
            if b not in label_map:
                label_map[b]=len(label_map)
            types[a] = label_map[b]
        for i in self.entity_map:
            if self.entity_map[i] not in types:
                types[self.entity_map[i]] = 0
        domains = {}
        ranges = {}
        for line in open(folder_name+'/domain.txt'):
            a,b = line.rstrip().split()
            if b not in label_map:
                label_map[b]=len(label_map)
            domains[a] = label_map[b]
            if a in self.relation_map: domains[self.relation_map[a]]=label_map[b]
        for line in open(folder_name+'/range.txt'):
            a,b = line.rstrip().split()
            if b not in label_map:
                label_map[b]=len(label_map)
            ranges[a] = label_map[b]
            if a in self.relation_map: ranges[self.relation_map[a]]=label_map[b]
        #print(self.relation_map)
        for i in range(len(self.facts)):
            #print(self.facts[i])
            self.facts[i][3] = domains.get(self.facts[i][1],0)
            self.facts[i][4] = ranges.get(self.facts[i][1],0)
            self.facts[i][5] = types[self.facts[i][0]]
            self.facts[i][6] = types[self.facts[i][2]]
        self.types = types
        self.rev_label_map = {i[1]:i[0] for i in label_map.items()}

        return self.types,domains,ranges,len(label_map)
    def augment_type_information(self, mapping):
        """
        Augments the current knowledge base with entity type information for more detailed evaluation.\n
        :param mapping: The maping from entity to types. Expected to be a int to int dict
        :return: None
        """
        self.type_map = mapping
        entity_type_matrix = numpy.zeros((len(self.entity_map), 1))
        for x in self.type_map:
            entity_type_matrix[self.entity_map[x], 0] = self.type_map[x]
        entity_type_matrix = torch.from_numpy(numpy.array(entity_type_matrix))
        self.entity_type_matrix = entity_type_matrix

    def compute_degree(self, out=True):
        """
        Computes the in-degree or out-degree of relations\n
        :param out: Whether to compute out-degree or in-degree
        :return: A numpy array with the degree of ith ralation at ith palce.
        """
        entities = [set() for x in self.relation_map]
        index = 2 if out else 0
        for i in range(self.facts.shape[0]):
            entities[self.facts[i][1]].add(self.facts[i][index])
        return numpy.array([len(x) for x in entities])
        

def union(kb_list):
    """
    Computes a union of multiple knowledge bases\n
    :param kb_list: A list of kb
    :return: The union of all kb in kb_list
    """
    l = [k.facts for k in kb_list]
    k = kb(None, kb_list[0].entity_map, kb_list[0].relation_map)
    k.facts = numpy.concatenate(l, 0)
    return k


def dump_mappings(mapping, filename):
    """
    Stores the mapping into a file\n
    :param mapping: The mapping to store
    :param filename: The file name
    :return: None
    """
    data = [[x, mapping[x]] for x in mapping]
    numpy.savetxt(filename, data)


def dump_kb_mappings(kb, kb_name):
    """
    Dumps the entity and relation mapping in a kb\n
    :param kb: The kb
    :param kb_name: The fine name under which the mappings should be stored.
    :return:
    """
    dump_mappings(kb.entity_map, kb_name+".entity")
    dump_mappings(kb.relation_map, kb_name+".relation")



