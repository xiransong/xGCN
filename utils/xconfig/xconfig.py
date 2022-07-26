from . import utils
from .parse import parse_arg_str_list


FIELD_SEP = '|'
ROOT_FIELD= 'root'


def join(field_str1, field_str2):
    return field_str1 + FIELD_SEP + field_str2


class xConfig:
    
    def __init__(self):
        self.d = {}
        
    def update(self, config_dict):
        self.d.update(config_dict)
    
    def load_yaml(self, file):
        self.update(
            utils.load_yaml(file)
        )
    
    def save_yaml(self, file):
        utils.save_yaml(file, self.d)
    
    def __getitem__(self, field_str):
        '''
            return value or subfield (dict)
        '''
        return self.get(field_str)
    
    def get(self, field_str):
        if field_str == ROOT_FIELD:
            return self.d
        else:
            _d, key = self._get_last_dict_and_key(field_str)
            return _d[key]
    
    def get_dict(self):
        return self.d
    
    def print(self):
        utils.print_dict(self.d)
    
    def _get_last_dict_and_key(self, field_str):
        lis = field_str.split(FIELD_SEP)
        _d = self.d
        for name in lis[:-1]:
            if name not in _d:
                raise KeyError
            _d = _d[name]
        return _d, lis[-1]
    
    def _set_field(self, field_str, value):
        _d, key = self._get_last_dict_and_key(field_str)
        _d[key] = value
    
    def set_field(self, *fields_and_values):
        for field_str, value in fields_and_values:
            self._set_field(field_str, value)
    
    def parse(self, arg_str_list):
        self.set_field(
            *parse_arg_str_list(arg_str_list)
        )
