import numpy
import torch
import torch.autograd


class data_loader(object):
    """
    Does th job of batching a knowledge base and also generates negative samples with it.
    """
    def __init__(self, kb, load_to_gpu, first_zero=True):
        """
        Duh..\n
        :param kb: the knowledge base to batch
        :param load_to_gpu: Whether the batch should be loaded to the gpu or not
        :param first_zero: Whether the first entity in the set of negative samples of each fact should be zero
        """
        self.kb = kb
        self.load_to_gpu = load_to_gpu
        self.first_zero = first_zero

    def sample(self, batch_size=1000, negative_count=10):
        """
        Generates a random sample from kb and returns them as numpy arrays.\n
        :param batch_size: The number of facts in the batch or the size of batch.
        :param negative_count: The number of negative samples for each positive fact.
        :return: A list containing s, r, o and negative s and negative o of the batch
        """
        indexes = numpy.random.randint(0, self.kb.facts.shape[0], batch_size)
        facts = self.kb.facts[indexes]
        s = numpy.expand_dims(facts[:, 0], -1)
        r = numpy.expand_dims(facts[:, 1], -1)
        o = numpy.expand_dims(facts[:, 2], -1)
        r_d = numpy.expand_dims(facts[:, 3], -1)
        r_r = numpy.expand_dims(facts[:, 4], -1)
        types = self.kb.types
        ns = numpy.random.randint(0, len(self.kb.entity_map), (batch_size, negative_count))
        no = numpy.random.randint(0, len(self.kb.entity_map), (batch_size, negative_count))
        t_s = numpy.array([[types.get(j,0) for j in s[i]] for i in range(batch_size)]) 
        t_o = numpy.array([[types.get(j,0) for j in o[i]] for i in range(batch_size)]) 
        t_ns = numpy.array([[types.get(j,0) for j in ns[i]] for i in range(batch_size)]) 
        t_no = numpy.array([[types.get(j,0) for j in no[i]] for i in range(batch_size)]) 
        if self.first_zero:
            ns[:, 0] = len(self.kb.entity_map)-1
            no[:, 0] = len(self.kb.entity_map)-1
        return [s, r, o, ns, no,r_d,r_r,t_s,t_o,t_ns,t_no]

    def tensor_sample(self, batch_size=1000, negative_count=10):
        """
        Generates a random sample from kb and returns them as torch tensors. Internally uses sampe\n
        :param batch_size: The number of facts in the batch or the size of batch.
        :param negative_count: The number of negative samples for each positive fact.
        :return: A list containing s, r, o and negative s and negative o of the batch
        """
        ls = self.sample(batch_size, negative_count)
        if self.load_to_gpu:
            return [torch.autograd.Variable(torch.from_numpy(x).cuda()) for x in ls]
        else:
            return [torch.autograd.Variable(torch.from_numpy(x)) for x in ls]
