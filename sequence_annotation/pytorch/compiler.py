from ..process.compiler import Compiler

class SimpleCompiler(Compiler):
    def __init__(self,optimizer,loss_type):
        super().__init__()
        self._optimizer_wrapper = optimizer
        self._loss_type = loss_type
        self._record['loss_type'] = loss_type
    @property
    def loss(self):
        return self._loss_type
    @property
    def optimizer(self):
        return self._optimizer
    def process(self,model):
        self._optimizer = self._optimizer_wrapper(model.parameters())
        self._record['optimizer'] = self._optimizer
    def before_process(self,path=None):
        if path is not None:
            json_path = create_folder(path) + "/setting/compiler.json"
            with open(json_path,'w') as fp:
                json.dump(self._record,fp)