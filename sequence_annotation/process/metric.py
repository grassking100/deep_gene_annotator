def categorical_metric(predict,answer,mask):
    """
    The method would calculate and return categorical metric
    
    Parameters
    ----------
    predict : np.array
        A predict which shape is (number,channel,length)
    answer : np.array
        A answer which shape is (number,channel,length)
    mask : np.array
        A mask which shape is (number,length) to mask the valid region
    Returns
    -------
    categorical_metric : dict
        A dictionary represents TP, FP, FN, FP, F, T of each type at base level
    """
    N,C,L = predict.shape
    if len(predict.shape) != 3 or len(answer.shape) != 3:
        raise Exception("Wrong input shape",predict.shape,answer.shape)
    if predict.shape[0] != answer.shape[0] or predict.shape[1] != answer.shape[1]:
        raise Exception("Inconsist batch size or channel size",predict.shape,answer.shape)
    data = {}
    predict = predict.argmax(1).reshape(-1)
    answer = answer[:,:,:L].argmax(1).reshape(-1)
    mask = mask[:,:L].reshape(-1)
    T_ = (predict == answer)
    F_ = (predict != answer)
    T_ = T_*mask
    F_ = F_*mask
    data['T'] = T_.sum().item()
    data['F'] = F_.sum().item()
    data["TPs"] = []
    data["FPs"] = []
    data["TNs"] = []
    data["FNs"] = []
    for index in range(C):
        P = predict == index
        R = answer == index
        TP = P & R
        TN = ~P & ~R
        FP = P & ~R
        FN = ~P & R
        TP = (TP*mask).sum().item()
        TN = (TN*mask).sum().item()
        FP = (FP*mask).sum().item()
        FN = (FN*mask).sum().item()
        data["TPs"].append(TP)
        data["FPs"].append(FP)
        data["TNs"].append(TN)
        data["FNs"].append(FN)

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

def contagion_matrix(predict,answer,mask):
    """
    The method would return categorical metric
    
    Parameters
    ----------
    predict : np.array
        A predict which shape is (number,channel,length)
    answer : np.array
        A answer which shape is (number,channel,length)
    mask : np.array
        A mask which shape is (number,length) to mask the valid region
    Returns
    -------
    contagion_matrix_ : array
        The array about contagion matrix which every column represent prediction to the row represent answer
    """
    N,C,L = predict.shape
    if len(predict.shape) != 3 or len(answer.shape) != 3:
        raise Exception("Wrong input shape",predict.shape,answer.shape)
    if predict.shape[0] != answer.shape[0] or predict.shape[1] != answer.shape[1]:
        raise Exception("Inconsist batch size or channel size",predict.shape,answer.shape)
    data = [[0]*C for _ in range(C)]
    predict = predict.argmax(1).reshape(-1)
    answer = answer[:,:,:L].argmax(1).reshape(-1)
    mask = mask[:,:L].reshape(-1)
    for output,label,mask_bit in zip(predict,answer,mask):
        if mask_bit == 1:
            data[label][output] += 1
    return data

def calculate_metric(data,prefix=None,label_names=None,calculate_precision=True,
                     calculate_recall=True,calculate_F1=True,
                     calculate_accuracy=True,round_value=None):
    """
    The method would calculate and return metric at base level
    
    Parameters
    ----------
    data : dict
        A dictionay returned by method, categorical_metric
    prefix : str
        A prefix to add in returned key (default: "")
    label_names : list of str
        A list of each channel name, if it doesn't be provided, then the index would be used as its name
    round_value : int, optional
        rounded at the specified number of digits, if it is None then the result wouldn't be rounded (default is None)
    Returns
    -------
    categorical_metric : dict
        A dictionary represents recall, precision, accuracy ,F1, macro F1, macro recall, and macro precision of each type at base level
    """
    prefix = prefix or ''
    label_num = len(data['TPs'])
    if label_names is None:
        label_names = list(range(label_num))
    T,F = data['T'], data['F']
    TPs,FPs = data['TPs'], data['FPs']
    TNs,FNs = data['TNs'], data['FNs']
    data = {}
    names = ["{}{}".format(prefix,name) for name in ['T','F']]
    data.update(dict(zip(names,[T,F])))
    recall_ = recall(TPs,FNs)
    precision_ = precision(TPs,FPs)
    f1 = F1(TPs,FPs,FNs)
    macro_precision = sum(precision_)/label_num
    macro_recall = sum(recall_)/label_num
    
    if calculate_precision:
        names = ["{}precision_{}".format(prefix,name) for name in label_names]
        data.update(dict(zip(names,precision_)))
        data["{}macro_precision".format(prefix)] = macro_precision

    if calculate_recall:
        names = ["{}recall_{}".format(prefix,name) for name in label_names]
        data.update(dict(zip(names,recall_)))
        data["{}macro_recall".format(prefix)] = macro_recall
        
    if calculate_accuracy:
        data["{}accuracy".format(prefix)] = accuracy(T,F)
        
    if calculate_F1:
        for index,val in enumerate(f1):
            names = ["{}F1_{}".format(prefix,name) for name in label_names]
            data.update(dict(zip(names,f1)))
        macro_F1 = 0
        if macro_precision+macro_recall > 0:
            macro_F1 = (2*macro_precision*macro_recall)/(macro_precision+macro_recall)
        data["{}macro_F1".format(prefix)] = macro_F1

    if round_value is not None:
        for key,values in data.items():
            data[key] = round(values,round_value)
    return data

