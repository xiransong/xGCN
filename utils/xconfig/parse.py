def parse_arg_str(arg_str):
    # e.g. arg_str = 'model|emb_lr:float:0.001'
    arg_str = arg_str.split(':')
    field = arg_str[0]
    dtype = eval(arg_str[1])
    value = dtype(arg_str[2])
    return field, value

def parse_arg_str_list(arg_str_list):
    arg_list = []
    for arg_str in arg_str_list:
        arg_list.append(
            parse_arg_str(arg_str)
        )
    return arg_list
