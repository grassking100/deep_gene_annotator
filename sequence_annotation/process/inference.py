import torch
import numpy as np


def create_basic_inference(first_n_channel=None, before=True):
    first_n_channel = first_n_channel or 3

    def basic_inference(ann, mask=None):
        if first_n_channel > ann.shape[1]:
            raise Exception("Wrong channel size, got {} and {}".format(
                first_n_channel, ann.shape[1]))
        if before:
            ann = ann[:, :first_n_channel, :]
        else:
            ann = ann[:, first_n_channel:, :]
        if mask is not None:
            L = ann.shape[2]
            ann = ann.transpose(0, 1)
            ann = ann * (mask[:, :L].to(ann.dtype))
            ann = ann.transpose(0, 1)
        return ann

    return basic_inference


def seq_ann_inference(ann,
                      mask,
                      transcript_threshold=None,
                      intron_threshold=None):
    """
        Data shape is N,C,L (where C>=2)
        Input channel order: Transcription potential, Intron potential
        Output channel order: Exon, Intron , Other
    """
    N, C, L = ann.shape
    if C != 2:
        raise Exception(
            "Channel size should be equal to two, but got {}".format(C))

    if mask is not None:
        mask = mask[:, :L].unsqueeze(1).float()

    transcript_threshold = transcript_threshold or 0.5
    intron_threshold = intron_threshold or 0.5
    transcript_potential = ann[:, 0, :].unsqueeze(1)
    intron_potential = ann[:, 1, :].unsqueeze(1)

    transcript_mask = (transcript_potential >= transcript_threshold).float()
    intron_mask = (intron_potential >= intron_threshold).float()
    exon = transcript_mask * (1 - intron_mask)
    intron = transcript_mask * intron_mask
    other = 1 - transcript_potential
    if mask is not None:
        exon = exon * mask
        intron = intron * mask
        other = other * mask
    result = torch.cat([exon, intron, other], dim=1)
    return result


def index2one_hot(index, channel_size):
    if (np.array(index) < 0).any() or (np.array(index) >= channel_size).any():
        raise Exception("Invalid number")
    L = len(index)
    loc = list(range(L))
    onehot = np.zeros((channel_size, L))
    onehot[index, loc] = 1
    return onehot


def ann_vec2one_hot_vec(ann_vec, length=None):
    C, L = ann_vec.shape
    index = ann_vec.argmax(0)
    if length is not None:
        index = index[:length]
    return index2one_hot(index, C)
