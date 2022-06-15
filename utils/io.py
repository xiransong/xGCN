import pickle as pkl
import json
import yaml
import gc
import os
from datetime import datetime


def save_pickle(filename, obj):
    # with open(filename, "wb") as f:
    #     pkl.dump(obj, f)
    save_pkl_obj(obj, filename, protocol=4)

def save_pkl_obj(v, filename,  protocol=None):
    print('saving {0} at {1}'.format(os.path.basename(filename),  datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    with open(filename, 'wb') as f:
        # pkl.dump(v, f)
        ##--  also try this:  pickletools.optimize()
        ##--  https://towardsdatascience.com/the-power-of-pickletools-handling-large-model-pickle-files-7f9037b9086b
        if protocol is not None:
            p = pkl.Pickler(f, protocol=protocol)  ## before Py3.8, default is 3;  otherwise default is 4.  protocal=4 supports big object
        else:
            p = pkl.Pickler(f)
        p.fast = True
        p.dump(v)
    print('finish saving {0} at {1}'.format(os.path.basename(filename),  datetime.now().strftime("%d/%m/%Y %H:%M:%S")))


def load_pickle(filename):
    with open(filename, "rb") as f:
        gc.disable()
        obj = pkl.load(f)
        gc.enable()
    return obj


def save_json(filename, obj):
    with open(filename, 'w') as f:
        json.dump(obj, f, indent=4)


def load_json(filename):
    with open(filename, 'r') as f:
        obj = json.load(f)
    return obj


def load_yaml(filename):
    with open(filename, 'r') as f:
        obj = yaml.load(f, Loader=yaml.FullLoader)
    return obj


def save_yaml(filename, obj):
    with open(filename, 'w') as f:
        yaml.dump(obj, f, indent=4, sort_keys=False)
