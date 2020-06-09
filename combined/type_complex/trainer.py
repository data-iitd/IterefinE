import numpy
import time
import evaluate
import torch
import kb
import utils
import os


class Trainer(object):
    def __init__(self, scoring_function, regularizer, loss, optim, train, valid, test, verbose=0, batch_size=1000,
                 hooks=None , eval_batch=100, negative_count=10, gradient_clip=None, regularization_coefficient=0.01,
                 save_dir="./logs"):
        super(Trainer, self).__init__()
        self.scoring_function = scoring_function
        self.loss = loss
        self.regularizer = regularizer
        self.train = train
        self.test = test
        self.valid = valid
        self.optim = optim
        self.batch_size = batch_size
        self.negative_count = negative_count
        self.ranker = evaluate.ranker(self.scoring_function, kb.union([train.kb, valid.kb, test.kb]))
        self.eval_batch = eval_batch
        self.gradient_clip = gradient_clip
        self.regularization_coefficient = regularization_coefficient
        self.save_directory = save_dir
        self.best_mrr_on_valid = {"valid":{"mrr":0.0}}
        self.verbose = verbose
        self.hooks = hooks if hooks else []

    def step(self):
        s, r, o, ns, no,r_d,r_r ,t_s,t_o,t_ns,t_no= self.train.tensor_sample(self.batch_size, self.negative_count)
        fp = self.scoring_function(s, r, o,r_d,r_r,t_s,t_o)
        fns = self.scoring_function(ns, r, o,r_d,r_r,t_ns,t_o)
        fno = self.scoring_function(s, r, no,r_d,r_r,t_s,t_no)
        if self.regularization_coefficient is not None:
            reg = self.regularizer(s, r, o) + self.regularizer(ns, r, o) + self.regularizer(s, r, no)
            reg = reg/(self.batch_size*self.scoring_function.embedding_dim*(1+2*self.negative_count))
        else:
            reg = 0
        loss = self.loss(fp, fns, fno) + self.regularization_coefficient*reg
        #print(loss)
        x = loss.item()
        rg = reg.item()
        self.optim.zero_grad()
        loss.backward()
        if(self.gradient_clip is not None):
            torch.nn.utils.clip_grad_norm(self.scoring_function.parameters(), self.gradient_clip)
        self.optim.step()
        debug = ""
        if "post_epoch" in dir(self.scoring_function):
            debug = self.scoring_function.post_epoch()
        return x, rg, debug

    def save_state(self, mini_batches):
        state = dict()
        state['mini_batches'] = mini_batches
        state['epoch'] = mini_batches*self.batch_size/self.train.kb.facts.shape[0]
        state['model_name'] = type(self.scoring_function).__name__
        state['model_weights'] = self.scoring_function.state_dict()
        state['optimizer_state'] = self.optim.state_dict()
        state['optimizer_name'] = type(self.optim).__name__
        filename = "model/model.pt"


        torch.save(state, filename)
        #try:
        #    if(state['valid_score_m']['mrr'] >= self.best_mrr_on_valid["valid_m"]["mrr"]):
        #        print("Best Model details:\n","valid_m",str(state['valid_score_m']), "test_m",str(state["test_score_m"]),
        #                                  "valid",str(state['valid_score_e2']), "test",str(state["test_score_e2"]),
        #                                  "valid_e1",str(state['valid_score_e1']),"test_e1",str(state["test_score_e1"]))
        #        best_name = os.path.join(self.save_directory, "best_valid_model.pt")
        #        self.best_mrr_on_valid = {"valid_m":state['valid_score_m'], "test_m":state["test_score_m"], 
        #                                  "valid":state['valid_score_e2'], "test":state["test_score_e2"],
        #                                  "valid_e1":state['valid_score_e1'], "test_e1":state["test_score_e1"]}

        #        if(os.path.exists(best_name)):
        #            os.remove(best_name)
        #        torch.save(state, best_name)#os.symlink(os.path.realpath(filename), best_name)
        #except:
        #    utils.colored_print("red", "unable to save model")

    def load_state(self, state_file):
        state = torch.load(state_file)
        self.scoring_function.load_state_dict(state['model_weights'])
        self.optim.load_state_dict(state['optimizer_state'])
        return state['mini_batches']

    def dump(self,dump_file,valid_labels=[],test_labels=[]):
        self.load_state(dump_file)
        # see test score 
        #test_score = evaluate.my_evaluate("test ",None, self.scoring_function, self.test.kb, self.eval_batch,
        #                                         verbose=self.verbose, hooks=self.hooks,labels = test_labels,eval_cnt = 0)
        #print('original rel wF1',test_score)
        _ = evaluate.my_evaluate("valid", 'train_preds.txt',self.scoring_function, self.valid.kb, self.eval_batch,
                                                    verbose=self.verbose, hooks=self.hooks,labels = [],eval_cnt = 0)
        #from sklearn.metrics.pairwise import cosine_similarity
        #a = self.scoring_function.E_t.weight.data.cpu().numpy()
        #print('starting computation')
        #import numpy as np
        #scores = np.abs(cosine_similarity(a,a))
        #print('ended computation') 
        #f = open('same_ent.txt','w')
        #print(len(self.train.kb.entity_map)*len(self.train.kb.entity_map))
        #for ent1,id1 in self.train.kb.entity_map.items():
        #    for ent2,id2 in self.train.kb.entity_map.items():
        #        if ent1!=ent2:
        #            f.write('%s\t%s\t%f\n'%(ent1,ent2,scores[id1][id2]))
        #print('same ent dumped')
    def start(self, steps=50, batch_count=(20, 10), mb_start=0,valid_labels = [], test_labels = [],dname=''):
        start = time.time()
        losses = []
        count = 0
        eval_cnt = 0
        best_valid_score = 0
        best_threshold = 0
        for i in range(mb_start, steps):
            l, reg, debug = self.step()
            losses.append(l)
            suffix = ("| Current Loss %8.4f | "%l) if len(losses) != batch_count[0] else "| Average Loss %8.4f | " % \
                                                                                         (numpy.mean(losses))
            suffix += "reg %6.3f | time %6.0f ||"%(reg, time.time()-start)
            suffix += debug
            prefix = "Mini Batches %5d or %5.1f epochs"%(i+1, i*self.batch_size/self.train.kb.facts.shape[0])
            #utils.print_progress_bar(len(losses), batch_count[0],prefix=prefix, suffix=suffix)
            if len(losses) >= batch_count[0]:
                losses = []
                count += 1
                if count == batch_count[1]:
                    self.scoring_function.eval()
                    threshold,valid_score = evaluate.my_evaluate("valid",None, self.scoring_function, self.valid.kb, self.eval_batch,
                                                   verbose=self.verbose, hooks=self.hooks,labels = valid_labels,eval_cnt = eval_cnt,threshold = None)
                    if valid_score>best_valid_score:
                            _ = evaluate.my_evaluate("valid", 'valid_preds.txt',self.scoring_function, self.valid.kb, self.eval_batch,
                                                    verbose=self.verbose, hooks=self.hooks,labels = valid_labels,eval_cnt = eval_cnt,threshold=threshold)
                            _ = evaluate.my_evaluate("test", 'test_preds.txt',self.scoring_function, self.test.kb, self.eval_batch,
                                                    verbose=self.verbose, hooks=self.hooks,labels = test_labels,eval_cnt = eval_cnt,threshold=threshold)
                            best_valid_score = valid_score
                            best_threshold = threshold
                            self.save_state(i)
                    eval_cnt+=1
                    self.scoring_function.train()
                    count = 0
                    #self.save_state(i, valid_score, test_score)
        #print(self.best_mrr_on_valid["valid"])
        #print(self.best_mrr_on_valid["test"])


