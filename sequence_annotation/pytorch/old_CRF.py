import torch
import torch.nn as nn
from keras.preprocessing.sequence import pad_sequences
from torch.nn.init import eye_,zeros_
import time
# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, vec.max(1)[1].item()]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class CRF(nn.Module):
#Reference from https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
    def __init__(self, tagset_size):
        super().__init__()
        self.tagset_size = tagset_size
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.reset_parameters()
    def _viterbi_decode(self, feats,length):
        pre_time = time.time()
        backpointers = []
        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), 0).cuda()
        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for index in range(length):
            feat = feats[index]
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = next_tag_var.max(1)[1].item()
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
        #Unifom transition to END
        best_tag_id = forward_var.max(1)[1].item()
        # Follow the back pointers to decode the best path.
        best_path = []
        #print(backpointers)
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        best_path.reverse()
        #print("CRF decode time",int(time.time()-pre_time))
        return best_path
    def forward(self, sentence,length=None):
        #CL.
        length = length or sentence.shape[1]
        return self._viterbi_decode(sentence.transpose(0,1),length)
    def reset_parameters(self):
        zeros_(self.transitions)
class BatchCRF(nn.Module):
    def __init__(self, CRF,ignore_index=-1):
        super().__init__()
        self.CRF = CRF
        self.ignore_index = ignore_index
        self.reset_parameters()
    def forward(self, output,lengths=None):
        #N,C,L
        pre_time = time.time()
        loss_sum = 0
        data = []
        for index in range(len(output)):
            observ = output[index]
            if lengths is None:
                length=None
            else:
                length = lengths[index]
            data.append(self.CRF(observ,length))
        data = pad_sequences(data,padding='post',value=self.ignore_index)
        data = torch.LongTensor(data).cuda()
        print("CRF decode time",int(time.time()-pre_time))
        return data
    def reset_parameters(self):
        self.CRF.reset_parameters()

class CRFLoss(nn.Module):
#Reference from https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
    def __init__(self,transitions):
        super().__init__()
        self.tagset_size = transitions.shape[0]
        self.transitions = transitions
    def _forward_alg(self, observes, length):
        #L,C
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor([0 for _ in range(self.tagset_size)]).cuda()
        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        # Iterate through the sentence
        
        for index in range(length):
            observe = observes[index]
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                #print(observe[next_tag])
                emit_score = observe[next_tag]#.expand(self.tagset_size)
                #print(emit_score)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag]
                #print(trans_score)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t)
            
        return log_sum_exp(forward_var)

    def _score_sentence(self, observes, tags, length=None):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).cuda()
        length = length or len(observes)
        for index in range(length-1):
            observe = observes[index]
            cur_tag = tags[index]
            next_tag = tags[index + 1]
            score += self.transitions[next_tag, cur_tag] + observe[next_tag]
        return score

    def forward(self, observes, tags, length=None):
        #C,L
        length = length or len(tags)
        observes = observes.transpose(0,1)
        forward_score = self._forward_alg(observes,length)
        gold_score = self._score_sentence(observes, tags,length)
        return forward_score-gold_score

class BatchCRFLoss(nn.Module):
    def __init__(self, CRF_loss):
        super().__init__()
        self.crf_loss = CRF_loss
    def forward(self, observes, target,lengths=None,**kwargs):
        #N,C,L
        N,C,L = observes.shape
        pre_time = time.time()
        #_,target = target.max(1)
        loss_sum = 0
        lengths = lengths or [L for _ in range(N)]
        for index in range(len(observes)):
            observe = observes[index]
            answer = target[index]
            length = lengths[index]
            loss_sum += self.crf_loss(observe,answer,length)
        print("CRF forward time",(time.time()-pre_time))
        return loss_sum/N