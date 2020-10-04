from sklearn.metrics import confusion_matrix

def is_binary(arr):
    return ((arr==0) | (arr==1)).all()

def get_categorical_metric(predict, answer, mask):
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
    N, C, L = predict.shape
    if predict.shape != answer.shape:
        raise Exception("The shape of prediction ({}), answer ({})are not the same", predict.shape,answer.shape)
    if not is_binary(predict):
        raise Exception("Prediction should be binary data")
        
    if not is_binary(answer):
        raise Exception("Answer should be binary data")
        
    if not is_binary(mask):
        raise Exception("Mask should be binary data")
        
    if (N,L) != mask.shape:
        raise Exception("Got the wrong mask shape {}",mask.shape)
    data = {}
    predict = predict.argmax(1).reshape(-1)
    answer = answer.argmax(1).reshape(-1)
    mask = mask.reshape(-1)
    T_ = (predict == answer)
    F_ = (predict != answer)
    T_ = T_ * mask
    F_ = F_ * mask
    for type_ in ['TP','FP','FN']:
        data[type_] = []
        
    for index in range(C):
        P = predict == index
        R = answer == index
        TP = P & R
        FP = P & ~R
        FN = ~P & R
        TP = (TP * mask).sum().item()
        FP = (FP * mask).sum().item()
        FN = (FN * mask).sum().item()
        data["TP"].append(TP)
        data["FP"].append(FP)
        data["FN"].append(FN)
    return data


def calculate_precision(TPs, FPs,nan_to_zero=False):
    precisions = []
    for TP, FP in zip(TPs, FPs):
        P = (TP + FP)
        if P != 0:
            precision = TP / P
        else:
            if nan_to_zero:
                precision = 0
            else:
                precision = float('nan')
        precisions.append(precision)
    return precisions


def calculate_recall(TPs, FNs,nan_to_zero=False):
    recalls = []
    for TP, FN in zip(TPs, FNs):
        RP = (TP + FN)
        if RP != 0:
            recall = TP / RP
        else:
            if nan_to_zero:
                recall = 0
            else:
                recall = float('nan')
        recalls.append(recall)
    return recalls


def calculate_F1(TPs, FPs, FNs,nan_to_zero=True):
    f1s = []
    for TP, FP, FN in zip(TPs, FPs, FNs):
        denominator = (2 * TP + FP + FN)
        if denominator != 0:
            f1 = 2 * TP / denominator
        else:
            if nan_to_zero:
                f1 = 0
            else:
                f1 = float('nan')
        f1s.append(f1)
    return f1s


def get_confusion_matrix(predict, answer, mask):
    """
    The method would return confusion matrix

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
    returned : array
        The array about confusion matrix m which vlaue m[i][j] indicates count of answer i to the prediction j
    """
    N, C, L = predict.shape
    if predict.shape != answer.shape:
        raise Exception("The shape of prediction ({}), answer ({})are not the same", predict.shape,answer.shape)
    if not is_binary(predict) or not is_binary(answer) or not is_binary(mask):
        raise Exception("Input should be binary data")
    if (N,L) != mask.shape:
        raise Exception("Got the wrong mask shape {}",mask.shape)

    indice = list(range(C+1))
    predict = predict.argmax(1).reshape(-1)
    answer = answer.argmax(1).reshape(-1)
    mask = mask.reshape(-1)
    predict[mask != 1] = C
    answer[mask != 1] = C
    returned = confusion_matrix(answer, predict,indice)[:C,:C]
    return returned


class MetricCalculator:
    def __init__(self,label_num,prefix=None,label_names=None,F1=True,
                 macro_F1=True,precision=False,recall=False,
                 round_value=None,details=False):
        """
        prefix : str
            A prefix to add in returned key (default: "")
        label_names : list of str
            A list of each channel name, if it doesn't be provided, then the index would be used as its name
        round_value : int, optional
            rounded at the specified number of digits, if it is None then the result wouldn't be rounded (default is None)
        """
        self._label_num = label_num
        self._prefix = prefix or ''
        self._label_names = label_names or list(range(label_num))
        self._macro_F1 = macro_F1
        self._precision = precision
        self._recall = recall
        self._F1 = F1
        self._round_value = round_value
        self._details = details
        if self._label_num != len(self._label_names):
            raise
    
    def __call__(self,data):
        """
        The method would calculate and return metric at base level
        Parameters
        ----------
        data : dict
            A dictionay returned by method, get_categorical_metric
        Returns
        -------
        returned : dict
            A dictionary represents recall, precision, F1, macro F1, macro recall, and macro precision of each type at base level
        """
        if self._label_num != len(data['TP']):
            raise
        TP, FP, FN = data['TP'], data['FP'], data['FN']
        data = {}
        recall_ = calculate_recall(TP, FN,True)
        precision_ = calculate_precision(TP, FP,True)
        f1 = calculate_F1(TP, FP, FN)
        macro_precision = sum(precision_) / self._label_num
        macro_recall = sum(recall_) / self._label_num
        if self._details:
            for value_type,values in zip(['TP','FP','FN'],[TP,FP,FN]):
                for name,value in zip(self._label_names,values):
                    name = "{}{}_{}".format(self._prefix, value_type,name)
                    data[name] = value
        if self._precision:
            names = ["{}precision_{}".format(self._prefix, name) for name in self._label_names]
            data.update(dict(zip(names, precision_)))
            data["{}macro_precision".format(self._prefix)] = macro_precision

        if self._recall:
            names = ["{}recall_{}".format(self._prefix, name) for name in self._label_names]
            data.update(dict(zip(names, recall_)))
            data["{}macro_recall".format(self._prefix)] = macro_recall

        if self._F1:
            for index, val in enumerate(f1):
                names = ["{}F1_{}".format(self._prefix, name) for name in self._label_names]
                data.update(dict(zip(names, f1)))
            if self._macro_F1:
                macro_F1 = 0
                if macro_precision + macro_recall > 0:
                    macro_F1 = (2 * macro_precision * macro_recall) / (macro_precision + macro_recall)
                data["{}macro_F1".format(self._prefix)] = macro_F1

        if self._round_value is not None:
            for key, value in data.items():
                if isinstance(value, float):
                    data[key] = round(value, self._round_value)
        return data
