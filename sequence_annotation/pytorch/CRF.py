import torch
import time
from torch.nn.init import zeros_
from torch.nn.utils.rnn import PackedSequence,pack_padded_sequence,pad_packed_sequence
from keras.preprocessing.sequence import pad_sequences
from torch import nn

def batch_log_sum_exp(x):
    #input shape:N,C
    #output shape:N
    N,C = x.shape
    max_score,_ = x.max(1,keepdim=True)
    max_score_broadcast = max_score.expand(N,C)
    return max_score.view(-1) + torch.log(torch.sum(torch.exp(x - max_score_broadcast),dim=1))

class BatchCRFLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.transitions = None
    def _forward_alg(self, observes, lengths):
        _,batch_sizes = pack_padded_sequence(observes,lengths,batch_first=True)
        N,C,_ = observes.shape
        observes = observes.transpose(0,2)
        previous_score = torch.full((N, C), 0).cuda()
        forward_var = torch.full((N, C), 0).cuda()
        for index in range(max(lengths)):
            batch_size = batch_sizes[index].item()
            mask = torch.FloatTensor(([1]*batch_size)+[0]*(N-batch_size))
            mask = mask.repeat(C).reshape(C,N).transpose(0,1).cuda()
            observe = observes[index]
            alphas_t = []
            for next_tag in range(C):
                emit_score = observe[next_tag].repeat(C,1).transpose(0,1)
                trans_score = self.transitions[next_tag]
                next_tag_var = forward_var + trans_score + emit_score
                log_score = batch_log_sum_exp(next_tag_var)
                alphas_t.append(log_score.reshape(1,N))
            forward_var = torch.cat(alphas_t).transpose(0,1)*mask + (1-mask)*previous_score
            previous_score = forward_var
        return batch_log_sum_exp(forward_var)

    def _score_sentence(self, observes, answers,lengths):
        #N,C,L //  N,L
        #return N
        _,batch_sizes = pack_padded_sequence(observes,lengths,batch_first=True)
        N,_ = answers.shape
        score = torch.zeros(N).cuda()
        observes = observes.transpose(0,2)
        answers = answers.transpose(0,1)
        indice = list(range(N))
        for index in range(max(lengths)-1):
            batch_size = batch_sizes[index].item()
            mask = [1]*batch_size+[0]*(N-batch_size)
            mask = torch.FloatTensor(mask).cuda()
            observe = observes[index].transpose(0,1)
            next_tag = answers[index + 1]
            cur_tag = answers[index]
            observe_score = observe.transpose(0,1)[next_tag,indice]
            current_score = self.transitions[cur_tag, next_tag] +observe_score
            score += current_score*mask
        return score

    def forward(self, observes, answers, lengths=None,**kwargs):
        if self.transitions is None:
            raise Exception("Transitions have not been set yet")
        answers = answers.max(1)[1]
        pre_time = time.time()
        O_N,O_C,O_L = observes.shape
        A_N,A_L = answers.shape
        if lengths is None:
            lengths = [O_L for _ in range(O_N)]
        #if O_L!=A_L:
        #    raise Exception("Lengths are not the same")
        if O_N!=A_N or O_N != len(lengths):
            raise Exception("Numbers are not the same")
        tagset_size = self.transitions.shape[0]
        if O_C != tagset_size:
            raise Exception("Observed data's channel, "+str(O_C)+\
                            ", should be same as annotation size,"+str(tagset_size))
        forward_score = self._forward_alg(observes,lengths)
        gold_score = self._score_sentence(observes, answers,lengths)
        result = sum(forward_score-gold_score)/O_N
        print("CRF forward time",time.time()-pre_time)
        return result

class BatchCRF(nn.Module):
    def __init__(self, tagset_size,ignore_index=-1):
        super().__init__()
        self.tagset_size = tagset_size
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.ignore_index = ignore_index
        self.reset_parameters()
    def _viterbi_decode(self, observes,lengths):
        pre_time = time.time()
        _,batch_sizes = pack_padded_sequence(observes,lengths,batch_first=True)
        N,C,L = observes.shape
        observes = observes.transpose(0,2)
        backpointers = []
        forward_var = torch.full((N, C), 0).cuda()
        stored_path_end_nodes = []
        indice = list(range(N))
        max_length = max(lengths)
        for index in range(max_length):
            #N,C
            batch_size = batch_sizes[index].item()
            mask = torch.FloatTensor(([1]*batch_size)+[0]*(N-batch_size)).cuda()
            observe = observes[index].transpose(0,1)
            ids = []
            viterbi_vars = []
            for next_tag in range(C):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = [item.item() for item in next_tag_var.max(1)[1]]
                ids.append(best_tag_id)
                viterbi_var = next_tag_var.transpose(0,1)[best_tag_id,indice]
                viterbi_vars.append(viterbi_var)
            viterbi_vars = torch.cat(viterbi_vars).reshape(C,N).transpose(0,1)
            forward_var = viterbi_vars + observe
            #Unifom transition to END
            stored_path_end_nodes.append(forward_var.max(1)[1])
            backpointers.append(ids)
        #L,C,N
        backpointers = torch.LongTensor(backpointers).transpose(1,2).cuda()
        best_path = []
        best_tag_id = torch.zeros(N,dtype=torch.int64).cuda()
        ignore = torch.LongTensor([self.ignore_index for _ in range(N)]).cuda()
        for index in range(max_length-1,-1,-1):
            mask = torch.LongTensor([index!=(length-1) for length in lengths]).cuda()
            backpointer = backpointers[index]
            stored_path_end_node = stored_path_end_nodes[index]
            best_tag_id = best_tag_id * mask + (1-mask)*stored_path_end_node
            new_best_tag_id = backpointer.transpose(0,1)[best_tag_id,indice]
            best_tag_id = new_best_tag_id
            batch_size = batch_sizes[index].item()
            tag_mask = torch.LongTensor(([1]*batch_size)+[0]*(N-batch_size)).cuda()
            best_path.append(best_tag_id*tag_mask+(1-tag_mask)*ignore)
        best_path.reverse()
        best_path = torch.cat(best_path).cuda().reshape(max_length,N).transpose(0,1)
        print("CRF decode time",int(time.time()-pre_time))
        pre_time = time.time()
        return best_path
    def forward(self, observes,lengths=None):
        #N,C,L
        with torch.no_grad():
            N,C,L = observes.shape
            if lengths is None:
                lengths = [L for _ in range(N)]
            outputs = self._viterbi_decode(observes,lengths)
        return outputs
    def reset_parameters(self):
        zeros_(self.transitions)