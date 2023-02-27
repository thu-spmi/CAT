"""
Implementation of some common functions.

In this script, it is designed to avoid importing the module cat
"""

from multiprocessing import Process
from typing import *
import argparse
import json
import os
import sys
import glob
import subprocess

# fmt:off
try:
    import utils.pipeline
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/../..'))
from utils.pipeline._constants import *
# fmt:on


class StringFormatter:
    if sys.stdout.isatty():
        """
        NOTE: 
            the escape sequences do not support nested using
            which means COLOR1 + (COLOR2 + ENDC) + ENDC is not allowed.
        """
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
    else:
        HEADER = ''
        OKBLUE = ''
        OKCYAN = ''
        OKGREEN = ''
        WARNING = ''
        FAIL = ''
        ENDC = ''
        BOLD = ''
        UNDERLINE = ''

    @staticmethod
    def __call__(s: str, ct: Literal):
        return f"{ct}{s}{sfmt.ENDC}"

    @staticmethod
    def udl(s: str):
        return sfmt(s, sfmt.UNDERLINE)

    @staticmethod
    def warn(prompt: str, func: Optional[Callable] = None):
        if func is None:
            return f"{sfmt('WARNING:', sfmt.WARNING)} {prompt}"
        else:
            return f"{sfmt('WARNING:', sfmt.WARNING)} {func.__name__}() {prompt}"

    @staticmethod
    def error(prompt: str, func: Optional[Callable] = None):
        if func is None:
            return f"{sfmt('ERROR:', sfmt.FAIL)} {prompt}"
        else:
            return f"{sfmt('ERROR:', sfmt.FAIL)} {func.__name__}() {prompt}"

    @staticmethod
    def header(prompt: str):
        return "{0} {1} {0}".format('='*20, sfmt(prompt, sfmt.BOLD))

    @staticmethod
    def missing(property_name: str, field: Optional[Union[str, Iterable[str]]] = None, raiseerror: bool = True):
        if raiseerror:
            formatter = sfmt.error
        else:
            formatter = sfmt.warn

        if isinstance(field, str):
            field = [field]
        if field is None:
            return formatter(f"missing '{property_name}'")
        else:
            return formatter(f"missing '{property_name}' in {sfmt.udl(':'.join(field))}")

    @staticmethod
    def set(property_name: str, value: str, isPath: bool = True):
        if isPath:
            value = sfmt.udl(value)
        return f"set '{property_name}' -> {value}"


sfmt = StringFormatter()


def initial_datainfo():
    if not os.path.isfile(F_DATAINFO):
        from utils.data import resolvedata
        resolvedata.main()


def spawn(target: Callable, args: Union[tuple, argparse.Namespace]):
    """Spawn a new process to execute the target function with given args."""
    if isinstance(args, argparse.Namespace):
        args = (args, )
    worker = Process(target=target, args=args)

    worker.start()
    worker.join()
    if worker.exitcode is not None and worker.exitcode != 0:
        sys.stderr.write("Worker unexpectedly terminated. See above info.\n")
        exit(1)


def readjson(file: str) -> dict:
    checkExist('f', file)
    with open(file, 'r') as fi:
        data = json.load(fi)
    return data


def dumpjson(obj: dict, target: str):
    assert os.access(os.path.dirname(target),
                     os.W_OK), f"{target} is not writable."
    with open(target, 'w') as fo:
        json.dump(obj, fo, indent=4)


def checkExist(f_type: Literal['d', 'f'], f_list: Union[str, List[str]]):
    """Check whether directory/file exist and raise error if it doesn't.
    """
    if f_type == 'd':
        check = os.path.isdir
    elif f_type == 'f':
        check = os.path.isfile
    else:
        raise RuntimeError(sfmt.error(
            f"unknown f_type: {f_type}, expected one of ['d', 'f']",
            checkExist
        ))

    if isinstance(f_list, str):
        f_list = [f_list]
    assert len(f_list) > 0, sfmt.error(
        f"expect the file/dir list to have at least one element, but found empty.",
        checkExist
    )

    hints = {'d': 'Directory', 'f': 'File'}
    not_founds = []

    for item in f_list:
        if not check(item):
            not_founds.append(item)

    if len(not_founds) > 0:
        o_str = f"{hints[f_type]} checking failed:"
        for item in not_founds:
            o_str += f"\n\t{item}"
        raise FileNotFoundError(o_str)
    else:
        return


class TextUtterances:
    """Read files with uid and sort the utterances in order by uid."""

    def __init__(self, files: Union[str, List[str]]) -> None:
        if isinstance(files, str):
            files = [files]

        checkExist('f', files)
        # [(uid, seek, file_id), ...]
        self._seeks = []    # type: List[Tuple[str, int, int]]
        self._files = files

        for idf, f in enumerate(files):
            with open(f, 'r') as fi:
                while True:
                    loc = fi.tell()
                    line = fi.readline()
                    if line == '':
                        break
                    uid = line.split(maxsplit=1)[0]
                    self._seeks.append(
                        (uid, loc, idf)
                    )
        self._seeks = sorted(self._seeks, key=lambda x: x[0])

    def __len__(self) -> int:
        return len(self._seeks)

    def __getitem__(self, index: int):
        return self._seeks[index]

    def __iter__(self):
        opened = {}
        for uid, loc, idf in self._seeks:
            if idf not in opened:
                opened[idf] = open(self._files[idf], 'r')

            opened[idf].seek(loc)
            cont = opened[idf].readline()[:-1].split(maxsplit=1)

            if len(cont) == 1:
                yield (uid, '')
            else:
                yield (uid, cont[1])

        for f in opened.values():
            f.close()
        return


def recursive_rpl(src_dict: dict, target_key: str, rpl_val: Any):
    """(In-place) Recursively replace the value of given key to the new one.

    Args:
        src_dict (dict) : a dict-like obj, could be a nested one.
        target_key (str) : the key that is to be updated.
        rpl_val : value to be put in.
    
    Return:
        src_dict

    Example:
    >>> src = {'a': {'b': 1, 'c': 2}, 'b': -1, 'd': 5}
    >>> recursive_rpl(src, 'b', 0)
    >>> src
    {
        'a': {
            'b': 0,
            'c': 2
        },
        'b': 0,
        'd': 5
    }
    """
    if not isinstance(src_dict, dict):
        return

    if target_key in src_dict:
        src_dict[target_key] = rpl_val
    else:
        for k, v in src_dict.items():
            recursive_rpl(v, target_key, rpl_val)

    return src_dict


def parse_args_from_var(parser: argparse.ArgumentParser, options: Dict[str, Any] = {}, positionals: List = []):
    """Similar to parser.parse_args(), but parse args from variables.

    args: A B -c -d E

    where -c is an action option 'store_true'

    -> `args = parse_args_from_var(parser, {'c': True, 'd': E}, [A, B])`
    """
    args = argparse.Namespace()
    args.__dict__.update((k.replace('-', '_'), v) for k, v in options.items())
    return parser.parse_args(args=positionals, namespace=args)


def set_visible_gpus(N: int) -> str:
    assert N >= 0
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(N))
    else:
        seen_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(seen_gpus[:N])
    return os.environ['CUDA_VISIBLE_DEVICES']


def get_free_port():
    """Return a free available port on local machine."""
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]


def find_text(dataset: Union[str, List[str]]) -> Tuple[List[str], List[str]]:
    """File text file location for given input. The searching order is:

    1. Check whether item in F_DATAINFO, if found, return F_DATAINFO[item]['trans'];

    2. Try to get the file by the item as it's a path, if found, return item;

    3. If both 1 & 2 failed, raise error.

    Args:
        dataset (str, list): dataset(s)

    Returns:
        Tuple[List[str], List[str]]

        which is (items_from_F_DATAINFO, items_read_as_path)
    """
    if isinstance(dataset, str):
        dataset = [dataset]

    datainfo = readjson(F_DATAINFO)

    items_datainfo = []  # find in src data, assume uid before each utterance
    items_rf_path = []   # find in local path, assume NO uid before each utterance
    for _set in dataset:
        if _set in datainfo:
            items_datainfo.append(datainfo[_set]['trans'])
        elif os.path.isfile(_set):
            items_rf_path.append(_set)
        else:
            raise FileNotFoundError(f"request dataset: '{_set}' not found.")
    return items_datainfo, items_rf_path


def train_nn(working_dir: str, prompt: str = '{}\n'):

    f_hyper_p = os.path.join(working_dir, F_HYPER_CONFIG)
    f_nnconfig = os.path.join(working_dir, F_NN_CONFIG)
    checkExist('f', [f_hyper_p, f_nnconfig])

    settings = readjson(f_hyper_p)
    assert 'train' in settings, sfmt.missing(
        'train', sfmt.udl(f_hyper_p)
    )
    for item in ['bin', 'option']:
        assert item in settings['train'], sfmt.missing(
            item, (sfmt.udl(f_hyper_p), 'train')
        )

    if 'tokenizer' not in settings:
        sys.stderr.write(
            sfmt.missing('tokenizer', raiseerror=False) + '\n' +
            sfmt.warn(
                f"you have to ensure the 'num_classes' in {sfmt.udl(f_nnconfig)} is correct.\n",
                train_nn
            )
        )
    else:
        if '|V|' in settings['tokenizer']:
            vocab_size = settings['tokenizer']['|V|']
        else:
            import cat.shared.tokenizer as tknz
            checkExist('f', settings['tokenizer']['file'])
            vocab_size = tknz.load(settings['tokenizer']['file']).vocab_size

        nnconfig = readjson(f_nnconfig)
        # recursively search for 'num_classes'
        recursive_rpl(nnconfig, 'num_classes', vocab_size)
        dumpjson(nnconfig, f_nnconfig)

    train_options = settings['train']['option']
    fmt_data = os.path.join(working_dir, "pkl/{}.pkl")
    for option, val in [('trset', 'train'), ('devset', 'dev')]:
        if option not in train_options:
            f_data = fmt_data.format(val)
            checkExist('f', f_data)
            train_options[option] = f_data
            sys.stdout.write(prompt.format(sfmt.set(option, f_data)))
            del f_data

    if 'dir' not in train_options:
        train_options['dir'] = working_dir
        sys.stdout.write(prompt.format(sfmt.set('dir', working_dir)))
    if 'dist-url' not in train_options:
        train_options['dist-url'] = f"tcp://localhost:{get_free_port()}"
        sys.stdout.write(prompt.format(sfmt.set(
            'dist-url', train_options['dist-url'], False)))

    import importlib
    interface = importlib.import_module(settings['train']['bin'])
    spawn(interface.main, parse_args_from_var(
        interface._parser(), train_options))

    # after the training is done. plot tb to image.
    tfevents = glob.glob(os.path.join(
        f"{working_dir}/{D_LOG}/**/", "events.out.tfevents.*"))
    if len(tfevents) == 0:
        sys.stderr.write(prompt.format(
            sfmt.error(f"No log data found in {working_dir}/{D_LOG}", train_nn)
        ))
        sys.exit(1)

    subprocess.run(
        f"{sys.executable} utils/plot_tb.py "
        f"{' '.join(tfevents)} "
        f"-o {os.path.join(working_dir, F_MONITOR_FIG)}",
        shell=True, check=True, stdout=subprocess.PIPE
    )


def get_corpus(
        f_hyper: str = None,
        std_data: Literal['train', 'dev', 'test', 'all', 'none'] = 'none',
        merge: bool = False,
        _iter: bool = False,
        adding_data: Iterable[str] = [],
        ops: List[Literal['rm-id']] = [],
        append_op: Callable[[str], str] = None,
        skipid: bool = False):
    """
    A common interface for reading text corpus, with normalization operations (if needed).

    Args:
        f_hyper (str): path to F_HYPER_CONFIG, could be None if std_data = 'none'
        std_data (str): one of `['train', 'dev', 'test', 'all', 'none']`, if `'train'/'dev'/'test'`,
                get data from `F_HYPER_CONFIG['data']['train'/'dev'/'test']`; if `'all'`, 
                use all of them; if `'none'`, use none of them.
        merge (bool): if there're multiple files, return them as one.
        _iter (bool): if True, return iteratively; if False, save output(s) to D_CACHE.
                if `merge=False`, this option is forced to `False`.
        adding_data (list): list of text files, besides the `'train'/'dev'/'test'` ones.
        ops (list): list of support normalized operations:
                'rm-id' : rm utt id. For text from F_DATAINFO, this will always be done unless
                    `skipid = True`. So, this op would only affect text from other than F_DATAINFO.
                    This op MUST be placed at the first place. See find_text() for defails;

        append_op (function): any customized callable function to do further text normalization.
        skipid (bool): treat first column as the id, normalizatino op would skip it.
            Once this is True, `rm-id` op won't take effect.
    
    Return
        if `return_as_iterator=True`:
            return a generator, whose __iter__() return str representing a line
        else:
            return a list of paths to output cached files
    """

    assert std_data in ['train', 'dev', 'test', 'all', 'none'], \
        sfmt.error(f"std_data got an unexpected value: {std_data}", get_corpus)
    assert std_data != 'none' or len(adding_data) > 0, \
        sfmt.error("std_data='none' and adding_data is empty", get_corpus)

    def rm_id(s: str) -> str:
        if skipid:
            return s
        else:
            s = s.split(maxsplit=1)
            if len(s) < 2:
                return ''
            else:
                return s[1]

    def lower(s: str) -> str:
        return s.lower()

    def upper(s: str) -> str:
        return s.upper()

    def add_bos(s: str) -> str:
        return '<s> ' + s

    def add_eos(s: str) -> str:
        return s + '</s>'

    def rm_space(s: str) -> str:
        # rm all spaces
        return s.replace(' ', '')

    def _sep(s: str):
        i = 0
        prev = ('\u4e00' <= s[0] <= '\u9fa5') or ('\u3400' <= s[0] <= '\u4db5')
        for j in range(1, len(s)):
            if prev ^ (prev := ('\u4e00' <= s[j] <= '\u9fa5') or ('\u3400' <= s[j] <= '\u4db5')):
                if s[j] != ' ' and s[j-1] != ' ':
                    yield s[i:(i := j)]
        yield s[i:] if i > 0 else s

    def seg_cn_other(s: str) -> str:
        # add space between chinese characters and other symbols (except space)
        if len(s) < 2:
            return s

        return ' '.join(_sep(s))

    corpus_trans, corpus_raw = [], []
    if std_data != 'none':
        assert f_hyper is not None, sfmt.error(
            "f_hyper is not set.", get_corpus)
        if os.path.isdir(f_hyper):
            f_hyper = os.path.join(f_hyper, F_HYPER_CONFIG)
        assert os.path.isfile(f_hyper), sfmt.error(f"{f_hyper} is not a file.")
        hyper_config = readjson(f_hyper)
        if std_data == 'all':
            f_list = sum((hyper_config['data'][item]
                         for item in ['train', 'dev', 'test']), [])
        else:
            f_list = hyper_config['data'][std_data]
        corpus_trans, corpus_raw = find_text(f_list)

    corpus_raw += list(adding_data)

    if not merge:
        _iter = False

    processor = []
    noop = True
    if len(ops) > 0:

        noop = False

    has_rmop = False
    for i, p in enumerate(ops):
        p = p.replace('-', '_')
        if p == 'rm_id':
            assert i == 0, sfmt.error(
                f"'rm-id' op MUST be placed at the first place, instead of {i}", get_corpus)
            has_rmop = True
            continue
        elif p == 'lower':
            f = lower
        elif p == 'upper':
            f = upper
        elif p == 'add_bos':
            f = add_bos
        elif p == 'add_eos':
            f = add_eos
        elif p == 'rm_space':
            f = rm_space
        elif p == 'seg_cn_other':
            f = seg_cn_other
        else:
            raise ValueError(f"op={p} not found.")
        processor.append(f)

    if append_op is not None:
        processor.append(append_op)

    def file_reader(f: str, isrmid: bool = False):
        if skipid:
            isrmid = False

        with open(f, 'r') as fi:
            for line in fi:
                line = line.strip('\n')
                if skipid:
                    uid, line = line.split(maxsplit=1)
                elif isrmid:
                    line = rm_id(line)

                for p in processor:
                    line = p(line)
                if skipid:
                    yield uid + '\t' + line + '\n'
                else:
                    yield line + '\n'

    import uuid
    cachedir = os.path.join(os.getcwd(), D_CACHE)
    if not _iter:
        os.makedirs(cachedir, exist_ok=True)

    n_trans = len(corpus_trans)
    corpus = corpus_trans + corpus_raw

    if _iter:
        for i in range(len(corpus)):
            for s in file_reader(corpus[i], ((i < n_trans) or has_rmop)):
                yield s
        return
    elif merge:
        f_out = os.path.join(cachedir, str(uuid.uuid4()))
        with open(f_out, 'w') as fo:
            for i in range(len(corpus)):
                for s in file_reader(corpus[i], ((i < n_trans) or has_rmop)):
                    fo.write(s)
        yield f_out
        return
    else:
        for i in range(len(corpus)):
            f_out = os.path.join(cachedir, str(uuid.uuid4()))
            with open(f_out, 'w') as fo:
                for s in file_reader(corpus[i], ((i < n_trans) or has_rmop)):
                    fo.write(s)
            yield f_out
        return


def train_tokenizer(f_hyper: str):
    def update_conf(_tok, path):
        # store some info about the tokenizer to the file
        cfg_hyper = readjson(f_hyper)
        cfg_hyper['tokenizer']['|V|'] = _tok.vocab_size
        cfg_hyper['tokenizer']['file'] = path
        dumpjson(cfg_hyper, f_hyper)

    checkExist('f', f_hyper)
    cfg_hyper = readjson(f_hyper)

    import cat.shared.tokenizer as tknz

    assert 'tokenizer' in cfg_hyper, sfmt.missing(
        'tokenizer', sfmt.udl(f_hyper))

    if 'file' not in cfg_hyper['tokenizer']:
        f_tokenizer = os.path.join(
            os.path.dirname(f_hyper), F_TOKENIZER)
        sys.stdout.write(
            "train_tokenizer(): " +
            sfmt.set('tokenizer:file', f_tokenizer)+'\n')
    else:
        f_tokenizer = cfg_hyper['tokenizer']['file']
        if os.path.isfile(f_tokenizer):
            update_conf(tknz.load(f_tokenizer), f_tokenizer)
            sys.stderr.write(
                sfmt.warn(
                    f"['tokenizer']['file'] exists: {sfmt.udl(f_tokenizer)}\n"
                    "... skip tokenizer training. If you want to do tokenizer training anyway,\n"
                    "... remove the ['tokenizer']['file'] in setting\n"
                    f"... or remove the file:{sfmt.udl(f_tokenizer)} then re-run the script.\n",
                    train_tokenizer
                )
            )
            return

    assert 'data' in cfg_hyper, sfmt.missing('data', sfmt.udl(f_hyper))
    assert 'train' in cfg_hyper['data'], sfmt.missing(
        'train', (sfmt.udl(f_hyper), 'data'))

    assert os.access(os.path.dirname(f_tokenizer), os.W_OK), \
        f"tokenizer:file is not writable: '{sfmt.udl(cfg_hyper['tokenizer']['file'])}'"

    f_text = None
    # combine the transcripts and remove the ids if needed.
    if 'option-train' in cfg_hyper['tokenizer']:
        if 'f_text' not in cfg_hyper['tokenizer']['option-train']:
            f_text = list(get_corpus(f_hyper, 'train', merge=True))[0]
            cfg_hyper['tokenizer']['option-train']['f_text'] = f_text

    tokenizer = tknz.initialize(cfg_hyper['tokenizer'])
    tknz.save(tokenizer, f_tokenizer)
    if f_text is not None:
        os.remove(f_text)

    update_conf(tokenizer, f_tokenizer)


def model_average(
        setting: dict,
        checkdir: str,
        returnifexist: bool = False) -> Tuple[str, str]:
    """Do model averaging according to given setting, return the averaged model path."""

    assert 'mode' in setting, sfmt.error(
        "'mode' not specified.", model_average)
    assert 'num' in setting, sfmt.error(
        "'num' not specified.", model_average)
    avg_mode, avg_num = setting['mode'], setting['num']

    import torch
    from utils.avgmodel import select_checkpoint, average_checkpoints

    suffix_avgmodel = f"{avg_mode}-{avg_num}"
    checkpoint = os.path.join(checkdir, suffix_avgmodel)
    tmp_check = checkpoint + ".pt"
    if os.path.isfile(tmp_check) and returnifexist:
        return tmp_check, suffix_avgmodel
    i = 1
    while os.path.isfile(tmp_check):
        tmp_check = checkpoint + f".{i}.pt"
        i += 1
    checkpoint = tmp_check

    if avg_mode in ['best', 'last']:
        params = average_checkpoints(
            select_checkpoint(checkdir, avg_num, avg_mode))
    else:
        raise NotImplementedError(
            f"Unknown model averaging mode: {avg_mode}, expected in ['best', 'last']")

    # delete the parameter of optimizer for saving disk.
    for k in list(params.keys()):
        if k != 'model':
            del params[k]
    torch.save(params, checkpoint)
    return checkpoint, suffix_avgmodel


def log_commit(f_hyper: str):
    if subprocess.run('command -v git', shell=True, capture_output=True).returncode != 0:
        sys.stderr.write(
            sfmt.warn(
                "git command not found, skip logging commit.\n",
                log_commit
            )
        )
    else:
        process = subprocess.run(
            "git log -n 1 --pretty=format:\"%H\"", shell=True, check=True, stdout=subprocess.PIPE)

        orin_settings = readjson(f_hyper)
        orin_settings['commit'] = process.stdout.decode('utf-8')
        dumpjson(orin_settings, f_hyper)
