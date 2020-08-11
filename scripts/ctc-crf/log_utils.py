import logging
from functools import reduce
import sys
def params_num(model):
    print('-' * 80)
    nparams = 0
    msg = []

    for name, param in model.named_parameters():
        param_shape = param.size()
        param_n = reduce(lambda x, y: x * y, list(param_shape))
        nparams += param_n
        msg.append('  {} ({}): {}'.format(list(param_shape), param_n, name))

    msg.append('  Total params: {} ({:.2f} M)'.format(nparams, float(nparams)  / (1000 * 1000)))
    return msg

def init_logging(name, file_path, level=logging.DEBUG, formatter='%(asctime)s - %(message)s'):
    logger = logging.getLogger('%slogger' % name)
    logger.setLevel(level)
    formatter = logging.Formatter(formatter)

    stream_hdl = logging.StreamHandler(sys.stdout)
    stream_hdl.setLevel(level)
    stream_hdl.setFormatter(formatter)
    logger.addHandler(stream_hdl)

    file_hdl = logging.FileHandler(file_path)
    file_hdl.setLevel(level)
    file_hdl.setFormatter(formatter)
    logger.addHandler(file_hdl)
    return logger