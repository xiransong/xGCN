import yaml


def load_yaml(file):
    with open(file, 'r') as f:
        obj = yaml.load(f, Loader=yaml.FullLoader)
    return obj


def save_yaml(file, obj):
    with open(file, 'w') as f:
        yaml.dump(obj, f, indent=4, sort_keys=False)


def print_dict(d, indent=0):
    for key in d:
        _d = d[key]
        if isinstance(_d, dict):
            print('  '*indent + str(key) + ' : ')
            print_dict(_d, indent+1)
        else:
            print('  '*indent + str(key) + ' : ' + str(_d))
