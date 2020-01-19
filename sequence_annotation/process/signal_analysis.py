import torch
import numpy as np

def cnn_saliency_map(model,seq,length,channel_index):
    if len(seq.shape) != 3 or seq.shape[0] != 1:
        raise Exception("Wrong size")
    if seq.grad is not None:
        seq.grad.data.zero_()
    model.eval()
    out = model.feature_block(seq,[length])[0]
    out = out [0,channel_index,:]
    out.backward(torch.ones(1,1,length))
    return seq.grad

def saliency_map(model,seq,labels,length,channel_index):
    if len(seq.shape) != 3 or len(labels.shape) != 3:
        raise Exception("Wrong size")
    if seq.shape[0] != 1 or labels.shape[0] != 1:
        raise Exception("Wrong size")
    if seq.grad is not None:
        seq.grad.data.zero_()
    model.eval()
    index = labels.long()[0,channel_index,:].reshape(1,1,-1)
    out = model(seq,[length])
    if len(out) > 1:
        out = out[0]
    out = out.gather(1,index)
    out.backward(torch.ones(1,1,length))
    return seq.grad

def get_signal_ppm(signal,one_hot_seqs,lengths,ratio=None,radius=None):
    """Get PPM based on signals and seqeunces"""
    if ratio is None:
        ratio = 0.5
    if radius is None:
        radius = 16
    #Get signal threshold
    concat_signal = None
    for seq,length in zip(signal,lengths):
        subseq = list(seq[0,0,:length])
        if concat_signal is None:
            concat_signal = subseq
        else:
            concat_signal += subseq
    most_signal_threshold = max(concat_signal)*ratio
    subseqs = None
    ppm = None
    if most_signal_threshold > 0:
        #Get PFM
        for length,one_hot_seq,subsignal in zip(lengths,one_hot_seqs,signal):
            subsignal_ = subsignal[0][0]
            indice = np.where(subsignal_>=most_signal_threshold)[0]
            for index in indice:
                lb = max(0,index+1-radius)
                ub = min(length-1,index+1+radius)
                subseq = one_hot_seq[:,lb:ub+1]*subsignal_[index]
                if subseq.shape[1] == 2*radius+1:
                    if subseqs is None:
                        subseqs = subseq
                    else:
                        subseqs += subseq
        if subseqs is not None:
            #Get PPM
            ppm = subseqs/subseqs.sum(0)
    return ppm

def ppms2meme(ppms,names,path,alphabet=None,strands=None,background=None):
    """Convert the position probability matrixs to MEME motif fomrat. 
    For more inofmration please visit: http://meme-suite.org/doc/meme-format.html"""
    alphabet = alphabet or 'ACGT'
    alphabets_ = list(alphabet)
    strands = strands or '+ -'
    with open(path,"w") as fp:
        fp.write("MEME version {}\n\n".format(4))
        fp.write("ALPHABET= {}\n\n".format(alphabet))
        fp.write("strands: {}\n\n".format(strands))
        if background is not None:
            fp.write('Background letter frequencies\n')
            for alpha,freq in zip(alphabets_, background):
                fp.write('{} {} '.format(alpha,freq))
            fp.write("\n\n")

        for name,ppm in zip(names,ppms):
            lines = ppm[alphabets_].values.tolist()
            fp.write("MOTIF {}\n".format(name))
            fp.write('letter-probability matrix: alength= {} w= {} nsites= 20 E= 0 \n'.format(len(alphabet),len(lines)))
            for line in lines:
                fp.write('  '.join([str(v) for v in line]))
                fp.write('\n')
            fp.write('\n')
