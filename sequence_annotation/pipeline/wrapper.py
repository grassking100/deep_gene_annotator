from .data_generator import DataGenerator
def fit_generator_wrapper_generator(batch_size=32,padding=None,epoch_shuffle=False,*args,**argws):
    def fit_generator_wrapper(model,data_):
        data = {}
        for data_kind,item in data_.items():
            data[data_kind] = DataGenerator(item['inputs'],item['answers'],
                                            batch_size,padding=padding,epoch_shuffle=epoch_shuffle)
        if 'validation' not in data.keys():
            data['validation'] = None
        return model.fit_generator(generator=data['training'],
                                   validation_data=data['validation'],*args,**argws)
    return fit_generator_wrapper

def evaluate_generator_wrapper_generator(batch_size=32,padding=None,epoch_shuffle=False,*args,**argws):
    def evaluate_generator_wrapper(model,data_):
        data = {}
        for data_kind,item in data_.items(): 
            data[data_kind] = DataGenerator(item['inputs'],item['answers'],
                                            batch_size,padding=padding,epoch_shuffle=epoch_shuffle)
        return model.evaluate_generator(data['testing'],*args,**argws)
    return evaluate_generator_wrapper

def fit_wrapper_generator(*args,**argws):
    def fit_wrapper(model,data):
        if 'validation' not in data.keys():
            val = None
        else:
            val = ([data['validation']['inputs']],[data['validation']['answers']])
        return model.fit(x=[data['training']['inputs']],
                         y=[data['training']['answers']],
                         validation_data=val,
                         *args,**argws)
    return fit_wrapper