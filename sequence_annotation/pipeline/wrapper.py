from ..model.data_generator import DataGenerator
def fit_generator_wrapper_generator(batch_size=32,*args,**argws):
    def fit_generator_wrapper(model,data_):
        data = {}
        for data_kind,item in data_.items(): 
            data[data_kind] = DataGenerator(item['inputs'],item['answers'],batch_size)
        if 'validation' not in data.keys():
            data['validation'] = None
        return model.fit_generator(generator=data['training'],
                                   validation_data=data['validation'],*args,**argws)
    return fit_generator_wrapper

def evaluate_generator_wrapper_generator(batch_size=32,*args,**argws):
    def evaluate_generator_wrapper(model,data_):
        data = {}
        for data_kind,item in data_.items(): 
            data[data_kind] = DataGenerator(item['inputs'],item['answers'],batch_size)
        return model.evaluate_generator(data['testing'],*args,**argws)
    return evaluate_generator_wrapper