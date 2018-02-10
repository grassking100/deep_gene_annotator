def get_protected_attrs_names(object_):
    class_name = object_.__class__.__name__
    attrs = [attr for attr in dir(object_) if attr.startswith('_') 
             and not attr.endswith('__')
             and not attr.startswith('_'+class_name+'__')]
    return attrs