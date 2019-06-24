import torch

def categorical_metric(outputs,labels,mask=None):
    #N,C,L
    N,C,L = outputs.shape
    if len(outputs.shape) != 3 or len(labels.shape) != 3:
        raise Exception("Wrong input shape",outputs.shape,labels.shape)
    if outputs.shape[0] != labels.shape[0] or outputs.shape[1] != labels.shape[1]:
        raise Exception("Inconsist batch size or channel size",outputs.shape,labels.shape)
    data = {}
    with torch.no_grad():
        if mask is None:
            mask = torch.ones(*outputs.shape)
        mask = mask[:,:L]
        labels = labels[:,:,:L]
        outputs = outputs.max(1)[1].contiguous().view(-1)
        labels = labels.max(1)[1].contiguous().view(-1)    
        T_ = (outputs == labels)
        F_ = (outputs != labels)
        T_ = T_*mask
        F_ = F_*mask
        data['T'] = T_.sum().item()
        data['F'] = F_.sum().item()
        data["TP"] = []
        data["FP"] = []
        data["TN"] = []
        data["FN"] = []
        for index in range(C):
            P = outputs == index
            R = labels == index
            TP = P & R
            TN = ~P & ~R
            FP = P & ~R
            FN = ~P & R
            TP = (TP*mask).sum().item()
            TN = (TN*mask).sum().item()
            FP = (FP*mask).sum().item()
            FN = (FN*mask).sum().item()
            data["TP"].append(TP)
            data["FP"].append(FP)
            data["TN"].append(TN)
            data["FN"].append(FN)

    return data

def accuracy(T,F):
    acc = T/(T+F)
    return acc

def precision(TPs,FPs):
    precisions = []
    for TP,FP in zip(TPs,FPs):
        P = (TP+FP)
        if P!=0:
            precision = TP/P
        else:
            precision = 0
        precisions.append(precision)
    return precisions

def recall(TPs,FNs):
    recalls = []
    for TP,FN in zip(TPs,FNs):
        RP = (TP+FN)
        if RP!=0:
            recall = TP/RP
        else:
            recall = 0
        recalls.append(recall)
    return recalls

def F1(TPs,FPs,FNs):
    f1s = []
    for TP,FP,FN in zip(TPs,FPs,FNs):
        denominator = (2*TP+FP+FN)
        if denominator!=0:
            f1 = 2*TP/denominator
        else:
            f1 = 0
        f1s.append(f1)
    return f1s