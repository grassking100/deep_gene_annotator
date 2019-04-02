class FitGenerator:
    def __init__(self,*args,**kwargs):
        self.model = None
        self._args = args
        self._kwargs = kwargs
        self.train_generator = None
        self.val_generator = None
    def __call__(self):
        return self.model.fit_generator(generator=self.train_generator,
                                         validation_data=self.val_generator,
                                         *self._args,**self._kwargs)
class EvaluateGenerator:
    def __init__(self,*args,**kwargs):
        self.model = None
        self._args = args
        self._kwargs = kwargs
        self.generator = None
    def __call__(self):
        return self.model.evaluate_generator(generator=self.generator,
                                              *self._args,**self._kwargs)
