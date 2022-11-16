'''
This script is used for parsing the json schema for experiment settings.
'''

import importlib
from cat.shared import decoder as pn_zoo
from cat.shared import encoder as tn_zoo
from cat.shared import scheduler, SpecAug
from cat.shared import tokenizer as tknz
from cat.shared.scheduler import Scheduler
from cat.rnnt import joiner as joiner_zoo
from cat.rnnt.joiner import AbsJointNet
from cat.shared._constants import (
    SCHEMA_NN_CONFIG,
    SCHEMA_HYPER_CONFIG,
    F_NN_CONFIG,
    F_HYPER_CONFIG
)
from cat.utils.pipeline.common_utils import (
    readjson,
    dumpjson
)

import os
import argparse
import inspect
import json
import typing
from collections import OrderedDict
from typing import *

import torch
import torch.nn as nn
from torch.optim import Optimizer

F_VSC_SETTING = ".vscode/settings.json"


def modify_vsc_schema(filematch: str, sgm_url: str):
    if os.path.isfile(F_VSC_SETTING):
        vsc_setting = readjson(F_VSC_SETTING)
    else:
        os.makedirs(os.path.dirname(F_VSC_SETTING), exist_ok=True)
        vsc_setting = {}

    ## dump to setting
    json_schema = vsc_setting.get('json.schemas', [])
    match = False
    for i, _sgm in enumerate(json_schema):
        if _sgm.get('fileMatch', [None])[0] == filematch:
            json_schema[i] = {
                "fileMatch": [filematch],
                "url": f_schema
            }
            match = True
            break
    if not match:
        json_schema.append({
            "fileMatch": [filematch],
            "url": f_schema
        })

    vsc_setting['json.schemas'] = json_schema
    dumpjson(vsc_setting, F_VSC_SETTING)
    return F_VSC_SETTING


def add_property(d: Union[Dict, OrderedDict], new_property: Dict[str, Any]):
    if 'properties' not in d:
        d['properties'] = OrderedDict()
    d['properties'].update(new_property)
    return d


def gen_object(type, default=None, desc: str = None) -> OrderedDict:
    _out = OrderedDict()
    if type == dict or type == OrderedDict:
        _out['type'] = 'object'
        _out['properties'] = OrderedDict()
    elif type == int or type == float:
        _out['type'] = 'number'
    elif type == str:
        _out['type'] = 'string'
    elif type == bool:
        _out['type'] = 'boolean'
    elif type == list or type == tuple:
        _out['type'] = 'array'

    if desc is not None:
        _out['description'] = desc
    if default is not None:
        _out['default'] = default
    return _out


def get_func_args(func: Callable):
    fullargs = inspect.getfullargspec(func)
    names = [x for x in fullargs[0] if x != 'self']   # type: List[str]
    defaults = fullargs[3]
    if defaults is None:
        defaults = []
    else:
        defaults = list(defaults[::-1])
        for i in range(len(defaults)):
            try:
                json.dumps(defaults[i])
            except (TypeError, OverflowError):
                defaults[i] = None
    defaults += [None for _ in range(len(names)-len(defaults))]
    annos = fullargs[-1]    # type: Dict[str, str]
    defaults = defaults[::-1]
    return (names, defaults, annos)


def module_processing(processing: typing.Union[dict, OrderedDict], module_list: list):
    module_options = gen_object(str)
    module_options['examples'] = [m.__name__ for m in module_list]
    add_property(processing, {'type': module_options})

    # setup kwargs
    allOf = []
    for m in module_list:
        IfCondition = {'properties': {'type': {'const': m.__name__}}}
        ThenProcess = {'properties': {'kwargs': gen_object(dict)}}
        kwargs = ThenProcess['properties']['kwargs']

        names, defaults, annos = get_func_args(m)
        parsed = []
        for _arg, _default in zip(names, defaults):
            parsed.append(
                (_arg, gen_object(annos.get(_arg, None), _default))
            )
        kwargs['properties'] = OrderedDict(parsed)

        allOf.append(OrderedDict([
            ('if', IfCondition),
            ('then', ThenProcess)
        ]))

    if len(module_list) == 1:
        processing['properties'] = allOf[0]['then']['properties']['kwargs']['properties']
    else:
        processing['required'] = ['type', 'kwargs']
        processing['allOf'] = allOf
    return processing


def parser_processing(parser: argparse.ArgumentParser):
    options = OrderedDict()
    for action in parser._actions[1:]:
        k = action.__dict__['dest']
        options[k] = gen_object(
            type=action.__dict__['type'],
            default=action.__dict__['default'],
            desc=action.__dict__['help']
        )
        if action.__dict__['choices'] is not None:
            options[k]['examples'] = action.__dict__['choices']
    return options


def bin_processing(d_bin_parser: Dict[str, argparse.ArgumentParser], desc: str = None):
    _schema = gen_object(dict, desc=desc)

    bin_options = gen_object(
        str, desc="Modules that provides training interface.")
    bin_options['examples'] = list(d_bin_parser.keys())
    add_property(_schema, {'bin': bin_options})

    allOf = []
    for _bin, _parser in d_bin_parser.items():
        allOf.append({
            'if': {'properties': {'bin': {'const': _bin}}},
            "then": {
                "properties": {
                    "option": {
                        "type": "object",
                        "description": _parser.prog + ' options',
                        "properties": parser_processing(_parser)
                    }
                }
            }
        })

    if len(allOf) == 1:
        _schema['properties'] = allOf[0]['then']['properties']['option']['properties']
    else:
        _schema['required'] = ['bin', 'option']
        _schema['allOf'] = allOf

    return _schema


# Neural network schema
schema = gen_object(dict, desc="Settings of NN training.")
schema['required'] = ['scheduler']

# Transducer
add_property(schema, {'trainer': gen_object(
    dict, desc="Please refer to build_model() function in hyper:train:bin for configuring help.")})


# Encoder
processing = gen_object(
    dict, desc="Configuration of Transducer transcription network / encoder.")  # type:OrderedDict
modules = []
for m in dir(tn_zoo):
    _m = getattr(tn_zoo, m)
    if inspect.isclass(_m) and issubclass(_m, tn_zoo.AbsEncoder):
        modules.append(_m)

module_processing(processing, modules)
add_property(schema, {'encoder': processing})


# Predictor
processing = gen_object(
    dict, desc="Configuration of Transducer prediction network / LM.")  # type:OrderedDict
modules = []
for m in dir(pn_zoo):
    _m = getattr(pn_zoo, m)
    if inspect.isclass(_m) and issubclass(_m, pn_zoo.AbsDecoder):
        modules.append(_m)

module_processing(processing, modules)
add_property(schema, {'decoder': processing})


# SpecAug
processing = gen_object(dict,
                        desc="Configuration of SpecAugument.")  # type:OrderedDict
module_processing(processing, [SpecAug])
add_property(schema, {'specaug': processing})


# Joint network
processing = gen_object(dict,
                        desc="Configuration of Transducer joiner network.")  # type:OrderedDict
modules = []
for m in dir(joiner_zoo):
    _m = getattr(joiner_zoo, m)
    if inspect.isclass(_m) and issubclass(_m, AbsJointNet):
        modules.append(_m)
module_processing(processing, modules)
add_property(schema, {'joiner': processing})


# Scheduler
processing = gen_object(
    dict, desc="Configuration of Scheduler.")  # type:OrderedDict
modules = []
for m in dir(scheduler):
    _m = getattr(scheduler, m)
    if inspect.isclass(_m) and issubclass(_m, Scheduler):
        modules.append(_m)
module_processing(processing, modules)

# setup the optimizer
optim = gen_object(dict, desc="Configuration of optimizer.")
modules = []
for m in dir(torch.optim):
    _m = getattr(torch.optim, m)
    if inspect.isclass(_m) and issubclass(_m, Optimizer):
        modules.append(_m)
module_processing(optim, modules)
add_property(optim, {'zeroredundancy': gen_object(bool, default=True)})
add_property(processing, {'optimizer': optim})
add_property(schema, {'scheduler': processing})

# dump
## dump schema
f_schema = f".vscode/{SCHEMA_NN_CONFIG}"
dumpjson(schema, f_schema)
## update setting
modify_vsc_schema(f"exp/**/{F_NN_CONFIG}", f_schema)


# hyper-parameter schema
# part of the settings in this schema is handcrafted
f_schema = f".vscode/{SCHEMA_HYPER_CONFIG}"
if os.path.isfile(f_schema):
    hyper_schema = readjson(f_schema)
else:
    hyper_schema = gen_object(dict, desc="Settings of Hyper-parameters.")

# lm corpus packing
# fmt:off
from cat.utils.data.pack_corpus import _parser as corpus_parser
parser = corpus_parser()
add_property(
    hyper_schema['properties']['data'],
    {
        'packing-text-lm': add_property(
            gen_object(dict, desc=parser.prog),
            parser_processing(parser)
        )
    }
)
del parser
# fmt:on

# schema for tokenizer
# setting according to cat.shared.tokenizer.initialize()
processing = gen_object(
    dict, desc="Configuration of tokenizer")  # type:OrderedDict
modules = []
for m in dir(tknz):
    _m = getattr(tknz, m)
    if inspect.isclass(_m) and issubclass(_m, tknz.AbsTokenizer):
        modules.append(_m)

type_opts = gen_object(str, desc="Type of Tokenizer")
type_opts.update({
    'examples': [m.__name__ for m in modules]
})
add_property(processing, {'type': type_opts})
allof = []
for m in modules:
    ifcondition = {'properties': {'type': {'const': m.__name__}}}
    thenprocess = {'properties': {
        'option-init': gen_object(dict, desc="options for initializing the tokenizer."),
        'option-train': gen_object(dict, desc="options for traininig tokenizer.")
    }}

    names, defaults, annos = get_func_args(m)
    thenprocess['properties']['option-init'].update({
        'properties': OrderedDict([
            (n, gen_object(annos.get(n, None), d))
            for n, d in zip(names, defaults)
        ])
    })
    names, defaults, annos = get_func_args(m.train)
    thenprocess['properties']['option-train'].update({
        'properties': OrderedDict([
            (n, gen_object(annos.get(n, None), d))
            for n, d in zip(names, defaults)
        ])
    })
    allof.append(OrderedDict([
        ('if', ifcondition),
        ('then', thenprocess)
    ]))
processing['allOf'] = allof
processing['required'] = 'type'
add_property(hyper_schema, {'tokenizer': processing})


# schema for field:train
# if you want to add a new training script, add it here.
binlist = [
    'cat.rnnt.train_unified',
    # 'cat.rnnt.train_mwer',
    # 'cat.rnnt.train_nce',
    'cat.rnnt.train',
    'cat.lm.train',
    'cat.ctc.train',
    # 'cat.ctc.train_scrf'
]

add_property(hyper_schema, {
    'train': bin_processing({
        name: importlib.import_module(name)._parser()
        for name in binlist
    }, desc="Configuration of NN training")
})


# field:inference:infer
binlist = [
    'cat.rnnt.decode',
    'cat.lm.ppl_compute',
    'cat.lm.rescore',
    'cat.ctc.cal_logit',
    'cat.ctc.decode'
]

inference = hyper_schema['properties']['inference']
add_property(inference, {
    'infer': bin_processing({
        name: importlib.import_module(name)._parser()
        for name in binlist
    }, desc="Configuration for inference.")
})

# field:inference:er
# fmt: off
from cat.utils.wer import _parser as parser_wer
# fmt: on
add_property(inference, {
    'er': bin_processing({
        'cat.utils.wer': parser_wer()
    })
})

# dump
dumpjson(hyper_schema, f_schema)
## update setting
modify_vsc_schema(f"exp/**/{F_HYPER_CONFIG}", f_schema)
