import os
from functools import wraps


class Registry:
    """

    """
    def __init__(self, name):
        self._registry_name = name
        self._obj_mapping = {}
        self.name = None

    def _register(self, name, obj):
        assert name not in self._obj_mapping, \
             f"Obj named {name} has been already registered in {self._registry_name} registry !!"
        self._obj_mapping[name] = obj

    def register(self, obj=None, name=None):
        self.name = name
        # used as decorator
        if obj is None:
            def decorate(func_or_class):
                if self.name is None:
                    self.name = func_or_class.__name__
                @wraps
                def wrapper(*args, **kwargs):
                    self._register(self.name, func_or_class)
                    return func_or_class(*args, **kwargs)
                    
                return wrapper
            
            return decorate

        # used as a function call
        if name is None:
            name = obj.__name__
        self._register(name, obj)
    
    def get(self, name):
        obj = self._obj_mapping.get(name)
        if obj is None:
            raise KeyError(f"No object named {name} in {self._registry_name} registry !!")
        return obj


    
