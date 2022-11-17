"""
Implementation of tokenizer
"""
from ..shared.coreutils import randstr
from typing import *
from collections import OrderedDict

import os
import io
import re
import sys
import pickle
import sentencepiece as sp
import jieba
jieba.default_logger.setLevel(jieba.logging.ERROR)


def gen_cache_path() -> str:
    return os.path.join('/tmp', randstr())


def file2bin(f_text: str) -> bytes:
    assert os.path.isfile(f_text), f"no such file: '{f_text}'"
    with open(f_text, 'rb') as fi:
        data = fi.read()
    return data


def bin2file(bindata: bytes, f_dest: Optional[str] = None) -> str:
    if f_dest is None:
        f_dest = gen_cache_path()
    with open(f_dest, 'wb') as fo:
        fo.write(bindata)
    return f_dest


class AbsTokenizer:

    def encode(self, strings: Union[str, Iterable[str]]) -> Union[List[int], List[List[int]]]:
        """Encode string to indices
        """
        if isinstance(strings, str):
            return self._enc(strings)
        try:
            iterator = iter(strings)
        except TypeError:
            raise RuntimeError(
                f"{self.__class__.__name__}.encode: input is neither str nor iterable.")

        cut_words = []
        for s in strings:
            cut_words.append(self._enc(s))
        return cut_words

    def decode(self, indices: Union[Iterable[int], Iterable[Iterable[int]]]) -> Union[str, Iterable[str]]:
        """Decode index to string."""
        try:
            iterator = iter(indices)
        except TypeError:
            raise RuntimeError(
                f"{self.__class__.__name__}.decode(): input is not iterable.")

        if isinstance(next(iterator), int):
            return self._dec(indices)

        try:
            iter(next(iter(indices)))
        except TypeError:
            raise RuntimeError(
                f"{self.__class__.__name__}.decode(): element of input is neither int nor iterable.")

        return [self._dec(x) for x in indices]

    def _enc(self, s: str) -> List[int]:
        """Implementation of encoding, will be invoked by self.encode()"""
        raise NotImplementedError

    def _dec(self, indices: Iterable[int]) -> str:
        """Implementation of decoding, will be invoked by self.decode()"""
        raise ImportError

    @staticmethod
    def train(*args, **kwargs):
        """Implementation of tokenizer training (if needed)"""
        return

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        raise NotImplementedError

    def dump_vocab(self, fileio: Optional[Union[str, "io.TextIOWrapper"]] = None) -> Union[None, Dict[int, str]]:
        """Dump vocabulary into a fileobject or return as dictionary"""
        vocab = self._vocab_to_dict()
        if fileio is None:
            return vocab
        elif isinstance(fileio, str):
            with open(fileio, 'w') as fo:
                for k, v in vocab.items():
                    fo.write(f"{v} {k}\n")
        elif isinstance(fileio, io.TextIOWrapper):
            for k, v in vocab.items():
                fileio.write(f"{v} {k}\n")
        else:
            raise ValueError(
                f"Unsupport file object of type: {type(fileio)}, expoected one of: str, TextIOWrapper")
        return None

    def _vocab_to_dict(self) -> Dict[int, str]:
        raise NotImplementedError

    def state_dict(self) -> OrderedDict:
        """Serialize tokenzier to dict object."""
        raise NotImplementedError

    def load_state_dict(self, state_dict: OrderedDict):
        """Load tokenizer from serialized object"""
        raise NotImplementedError

    def __getstate__(self):
        return self.state_dict()

    def __setstate__(self, state: dict):
        self.load_state_dict(state)


class SimpleTokenizer(AbsTokenizer):
    """Passing a file, a list of words, or a word-to-index mapping to build a simple tokenizer.

    When passed a file, assume one word per line, 
    if `read_index_from_file=False`, use the first one as word, and the lineid+1 as token id;
    if `read_index_from_file=True`, take first column as words, second column as token id.
    """

    def __init__(self, dmap: Union[str, List[str], Dict[str, int]], read_index_from_file: bool = False) -> None:
        super().__init__()
        self._r_vocab = OrderedDict([('<s>', 0), ('<unk>', 1)])
        if isinstance(dmap, str) and os.path.isfile(dmap):
            if read_index_from_file:
                consturctd = {}
                with open(dmap, 'r') as fi:
                    for line in fi:
                        line = line.strip()
                        if line == '':
                            continue
                        w, i = line.split(maxsplit=2)[:2]
                        consturctd[w] = i
            else:
                consturctd = []
                with open(dmap, 'r') as fi:
                    for line in fi:
                        line = line.strip()
                        if line == '':
                            continue
                        consturctd.append(line.split(maxsplit=1)[0])
            dmap = consturctd

        if isinstance(dmap, list):
            offset = 2
            for c in dmap:
                if c in self._r_vocab:
                    continue
                self._r_vocab[c] = offset
                offset += 1
        elif isinstance(dmap, dict):
            if '<s>' in dmap:
                del dmap['<s>']
            if '<unk>' in dmap:
                del dmap['<unk>']
            self._r_vocab.update(dmap)
        else:
            raise ValueError(str(type(dmap)))
        self._i_vocab = [c for c in self._r_vocab.keys()]

    @property
    def vocab_size(self) -> int:
        return len(self._r_vocab)

    def _enc(self, s: str) -> List[int]:
        return [self._r_vocab.get(c, 1) for c in s.split()]

    def _dec(self, indices: Iterable[int]) -> str:
        return ' '.join(self._i_vocab[i] for i in indices)

    def _vocab_to_dict(self) -> Dict[int, str]:
        return {i: c for i, c in enumerate(self._i_vocab)}

    def state_dict(self) -> OrderedDict:
        return self._r_vocab

    def load_state_dict(self, state_dict: OrderedDict):
        self._r_vocab = state_dict
        self._i_vocab = [c for c in self._r_vocab.keys()]


class JiebaTokenizer(AbsTokenizer):
    def __init__(self, userdict: Optional[Union[str, bytes]] = None, bos_id: int = 0) -> None:
        super().__init__()
        self._tokenizer = jieba.Tokenizer()
        if userdict is None:
            self._tokenizer.initialize()
            self._vocabulary = list(self._tokenizer.FREQ.items())
            self.byte_dict = None
        else:
            if isinstance(userdict, str):
                assert os.path.isfile(
                    userdict), f"{userdict} is not a valid file."
                self._tokenizer.set_dictionary(userdict)
                self.byte_dict = file2bin(userdict)
                self._tokenizer.initialize()
            elif isinstance(userdict, bytes):
                self.byte_dict = userdict
                cachefile = bin2file(userdict)
                self._tokenizer.set_dictionary(cachefile)
                self._tokenizer.initialize()
                os.remove(cachefile)
            else:
                raise ValueError(f"Unknown userdict type: {type(userdict)}")
            # we only load user custom word
            self._vocabulary = [(w, None) for w, freq in self._tokenizer.FREQ.items(
            ) if freq > 0]  # type: Dict[str, int]

        if bos_id == 1:
            unk_id = 0
        else:
            unk_id = 1
        if bos_id == -1:
            bos_id = len(self._vocabulary) + 1
        assert bos_id < len(self._vocabulary) + 2

        self._vocabulary = OrderedDict(
            [('<s>', bos_id), ('<unk>', unk_id)] + self._vocabulary)
        self._reverse_vocab = tuple(self._vocabulary.keys())    # type: tuple
        for idx, w in enumerate(self._vocabulary):
            self._vocabulary[w] = idx

    def cut(self, s: str) -> Generator[str, None, None]:
        """Do segmentation"""
        for w in self._tokenizer.cut(s.strip(), HMM=False):
            if w == ' ':
                continue
            yield w
        return

    def _enc(self, s: str) -> List[int]:
        rt_indices = []     # type: List[int]
        for w in self.cut(s):
            if w not in self._vocabulary:
                rt_indices.append(self._vocabulary["<unk>"])
            else:
                rt_indices.append(self._vocabulary[w])
        return rt_indices

    def _dec(self, indices: Iterable[int]) -> str:
        return ''.join([self._reverse_vocab[i] for i in indices])

    @property
    def vocab_size(self) -> int:
        return len(self._vocabulary)

    def _vocab_to_dict(self) -> Dict[int, str]:
        return {idx: self._reverse_vocab[idx] for idx in range(self.vocab_size)}

    def state_dict(self) -> OrderedDict:
        return OrderedDict([
            ('vocab', self._vocabulary),
            ('reverse-vocab', self._reverse_vocab),
            ('dict-data', self.byte_dict)
        ])

    def load_state_dict(self, state_dict: OrderedDict):
        assert 'vocab' in state_dict
        assert 'reverse-vocab' in state_dict
        assert 'dict-data' in state_dict

        self._vocabulary = state_dict['vocab']
        self._reverse_vocab = state_dict['reverse-vocab']
        self.byte_dict = state_dict['dict-data']
        self._tokenizer = jieba.Tokenizer()
        if self.byte_dict is not None:
            cachefile = bin2file(self.byte_dict)
            self._tokenizer.set_dictionary(cachefile)
            self._tokenizer.initialize()
            os.remove(cachefile)
        else:
            self._tokenizer.initialize()


class JiebaComposeLexiconTokenizer(JiebaTokenizer):
    """Tokenizer composing jieba segmentation and word2phone mapping for Chinese."""

    def __init__(
            self,
            lexicon: str,
            add_special_token: bool = True,
            bos_interface: str = '<s>',
            unk_interface: str = '<unk>',
            userdict: Optional[Union[str, bytes]] = None) -> None:
        """
        Args:
            lexicon (str) : file contains the mapping of word to phone. Usually annotated as 'lexicon.txt'
            add_special_token (bool) : add <s>, <unk> to the lexicon, otherwise they're assumed to be in the lexicon.
            bos_interface (str) : start of an utterance, usually '<s>' or '<bos>'
            unk_interface (str) : unknown word representation in the mapping. usually '<unk>' or '<UNK>'
            userdict (str, optional) : custom dictionary file, if not set, use Jieba default one.
        """
        super().__init__(userdict, bos_id=0)

        self._w2p_tokenizer = LexiconTokenizer(
            lexicon=lexicon,
            add_special_token=add_special_token,
            bos_interface=bos_interface,
            unk_interface=unk_interface
        )
        self._check_vocab_cover()

        # now the vocabulary is useless
        self._vocabulary.clear()
        self._reverse_vocab = tuple()

    def _check_vocab_cover(self):
        absense = list(set(self._vocabulary)-set(self._w2p_tokenizer._w2pid))
        promt = ' '.join(f"'{x}'" for x in absense[:5])
        if len(absense) > 5:
            promt += "..."
        if len(absense) > 0:
            sys.stderr.write(
                f"WARNING: {len(absense)} word(s) defined in vocabulary but missing in lexicon:\n"
                f"    {promt}\n"
            )

    @property
    def vocab_size(self) -> int:
        return self._w2p_tokenizer.vocab_size

    def state_dict(self) -> OrderedDict:
        parent_state = super().state_dict()
        parent_state.update({
            'w2p': pickle.dumps(self._w2p_tokenizer)
        })
        return parent_state

    def load_state_dict(self, state_dict: OrderedDict):
        super().load_state_dict(state_dict)
        self._w2p_tokenizer = pickle.loads(state_dict['w2p'])

    def _vocab_to_dict(self) -> Dict[int, str]:
        raise NotImplementedError

    def _enc(self, s: str) -> List[int]:
        # do segmentation
        cut_words = self.cut(s)
        # encoder text to ids
        return sum(self._w2p_tokenizer.encode(cut_words), [])

    def decode(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support decode() method.")


class LexiconTokenizer(AbsTokenizer):
    """Lexicon-based tokenizer. A light wrapper."""

    def __init__(
            self,
            lexicon: str,
            add_special_token: bool = True,
            bos_interface: str = '<s>',
            unk_interface: str = '<unk>') -> None:
        """
        Args:
            lexicon (str) : file contains the mapping of word to units. Usually annotated as 'lexicon.txt'
                            Each line should be as 
                            <word> <unit0> <unit1> ...
            add_special_token (bool) : add <s>, <unk> to the lexicon, otherwise they're assumed to be in the lexicon.
            bos_interface (str) : start of an utterance, usually '<s>' or '<bos>'
            unk_interface (str) : unknown word representation in the mapping. usually '<unk>' or '<UNK>'
        """
        super().__init__()
        assert os.path.isfile(lexicon), f"given w2p_map='{lexicon}' not exist."
        self._bos_token = bos_interface
        self._unk_token = unk_interface
        self.init_lexicon(lexicon, add_special_token)

    def init_lexicon(self, f_mapping: str, add_special_token: bool):
        p_rm_consecutive_space = re.compile(r"\s+")

        lexicon = {}
        with open(f_mapping, 'r') as fi:
            for line in fi:
                if line == '':
                    continue
                utt = re.sub(p_rm_consecutive_space, ' ', line).strip().split()
                if utt[0] not in lexicon:
                    # for words with multiple pronounciations, keep the first.
                    lexicon[utt[0]] = utt[1:]

        units = {}
        word2unitid = OrderedDict()
        if add_special_token:
            units[self._bos_token] = 0
            units[self._unk_token] = 1
            word2unitid[self._bos_token] = (0,)
            word2unitid[self._unk_token] = (1,)

        raw_units = set().union(*(lexicon.values()))
        lu = len(units)
        units.update({
            _unit: id+lu
            for id, _unit in enumerate(raw_units)
        })
        for word, utt in lexicon.items():
            word2unitid[word] = tuple(units[x] for x in utt)

        self._w2pid = word2unitid
        if not add_special_token and (self._bos_token not in self._w2pid or self._unk_token not in self._w2pid):
            raise RuntimeError(
                f"{self._bos_token} and(or) {self._unk_token} are not found in '{f_mapping}', "
                "it's your duty to add these special tokens."
            )

        self._units = units
        if self._w2pid[self._bos_token] != (0,):
            sys.stderr.write(
                f"warning: {self.__class__.__name__} bos token is set to {self._w2pid[self._bos_token]},\n"
                "... but for most of the cases, we assume <s>=<blk>=0, so it might cause some underminted error.\n")

    @property
    def vocab_size(self) -> int:
        return len(self._units)

    def state_dict(self) -> OrderedDict:
        return {
            'w2pid': self._w2pid,
            'units': self._units,
            'bos': self._bos_token,
            'unk': self._unk_token
        }

    def load_state_dict(self, state_dict: OrderedDict):
        self._w2pid = state_dict['w2pid']
        self._units = state_dict['units']
        self._bos_token = state_dict['bos']
        self._unk_token = state_dict['unk']

    def _vocab_to_dict(self) -> Dict[int, str]:
        raise NotImplementedError

    def _enc(self, s: str) -> List[int]:
        cut_words = s.split()
        unkid = self._w2pid[self._unk_token]
        rt_indices = [
            list(self._w2pid.get(w, unkid))
            for w in cut_words
        ]     # type: List[List[int]]
        return sum(rt_indices, [])

    def decode(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support decode() method.")


class RawTokenizer(AbsTokenizer):
    """A wrapper tokenizer that do nothing. encode('1 2 3') -> [1, 2, 3]"""

    def __init__(self, num_units: int) -> None:
        super().__init__()
        assert isinstance(num_units, (float, int))
        num_units = int(num_units)
        assert num_units > 0
        self.n_vocab = num_units

    def _enc(self, s: str) -> List[int]:
        return [int(x) for x in s.split()]

    def _dec(self, indices: Iterable[int]) -> str:
        return ' '.join(str(x) for x in indices)

    @property
    def vocab_size(self) -> int:
        return self.n_vocab

    def _vocab_to_dict(self) -> Dict[int, str]:
        return {i: str(i) for i in range(self.n_vocab)}

    def state_dict(self) -> OrderedDict:
        return OrderedDict([('n_vocab', self.n_vocab)])

    def load_state_dict(self, state_dict: OrderedDict):
        self.n_vocab = state_dict['n_vocab']
        return


class SentencePieceTokenizer(AbsTokenizer):
    """SentencePiece tokenizer wrapper."""

    def __init__(self, model_file: str = None) -> None:
        super().__init__()
        if model_file is None:
            return

        assert model_file is not None
        if not os.path.isfile(model_file):
            raise RuntimeError(
                f"{self.__class__.__name__}: sentencepiece model path \'{model_file}\' is invalid.")
        self._tokenzier = sp.SentencePieceProcessor(model_file=model_file)
        # FIXME (Huahuan): 
        #     I cannot figure out a way to export SentencePieceProcessor object to a file
        #     so here I make a redundant copy of the source file.
        self.byte_model = file2bin(model_file)

    def encode(self, strings: Union[str, Iterable[str]]) -> Union[List[int], List[List[int]]]:
        return self._tokenzier.Encode(strings)

    def decode(self, idx_tokens: Union[List[int], List[List[int]]]) -> Union[str, Iterable[str]]:
        return self._tokenzier.Decode(idx_tokens)

    @property
    def vocab_size(self) -> int:
        return self._tokenzier.vocab_size()

    def _vocab_to_dict(self) -> Dict[int, str]:
        return {idx: self._tokenzier.IdToPiece(idx) for idx in range(self.vocab_size)}

    def state_dict(self) -> OrderedDict:
        return OrderedDict([
            ('model-data', self.byte_model)
        ])

    def load_state_dict(self, state_dict: OrderedDict):
        assert 'model-data' in state_dict
        self.byte_model = state_dict['model-data']
        cachefile = bin2file(self.byte_model)
        self._tokenzier = sp.SentencePieceProcessor(model_file=cachefile)
        os.remove(cachefile)

    @staticmethod
    def train(
            f_text: str,
            model_prefix: str,
            vocab_size: int,
            add_dummy_prefix: bool = True,
            character_coverage: float = 0.9995,
            model_type: Literal['unigram', 'bpe', 'word', 'char'] = 'unigram',
            use_all_vocab: bool = False,
            bos_id: int = 0,
            unk_id: int = 1,
            eos_id: int = -1,
            unk_surface: str = "<unk>",
            minloglevel: int = 1,
            user_defined_symbols: str = "",
            train_extremely_large_corpus: bool = False,
            **options):
        """Train the sentencepiece tokenizer.
        
        For full options, take a hook at
        https://github.com/google/sentencepiece/blob/master/doc/options.md
        """
        assert os.path.isfile(f_text), f"Given text file '{f_text}' not found."
        if user_defined_symbols != "":
            if isinstance(user_defined_symbols, str) and os.path.isfile(user_defined_symbols):
                user_defined_symbols = [x.strip() for x in open(
                    user_defined_symbols, 'r').readlines()]

        os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
        sp.SentencePieceTrainer.Train(
            input=f_text,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            add_dummy_prefix=add_dummy_prefix,
            character_coverage=character_coverage,
            model_type=model_type,
            use_all_vocab=use_all_vocab,
            bos_id=bos_id,
            unk_id=unk_id,
            eos_id=eos_id,
            unk_surface=unk_surface,
            minloglevel=minloglevel,
            user_defined_symbols=user_defined_symbols,
            train_extremely_large_corpus=train_extremely_large_corpus,
            **options
        )


def initialize(cfg: Dict) -> AbsTokenizer:
    """ Initialize tokenizer according to the configurations, 
        which should be in following format:
        {
            'type': <Class name of tokenizer>,
            'option-init': {
                <options passed to ABCTokenizer.__init__()>
                ...
            },
            'option-train': {
                <options passed to ABCTokenizer.train()>
                ...
            }
        }
    """
    assert 'type' in cfg, "'type' is missing in tokenizer initialization."
    assert isinstance(cfg['type'], str), f"invalid type: '{cfg['type']}'"
    assert cfg['type'].isidentifier(), f"invalid type str: '{cfg['type']}'"

    try:
        eval(cfg['type'])
    except NameError:
        raise NameError(f"'{cfg['type']}' is not defined.")

    abctknz = eval(cfg['type'])     # type: AbsTokenizer
    if 'option-train' in cfg:
        abctknz.train(**cfg['option-train'])

    init_opts = cfg.get('option-init', {})
    if abctknz == SentencePieceTokenizer:
        if 'model_file' not in init_opts:
            # if path to the sp model is not configured, use the trained one.
            model_prefix = cfg.get(
                'option-train', {}).get('model_prefix', None)
            if model_prefix is None:
                raise ValueError(
                    f"For {SentencePieceTokenizer.__name__}, one of the settings is required:\n"
                    "1. If you already have sp model, set 'model_file' in 'option-init';\n"
                    "2. If you want to train a new one, set 'option-train' up to your request."
                )
            init_opts['model_file'] = model_prefix+'.model'

    return abctknz(**init_opts)


def save(obj: AbsTokenizer, target: str):
    """Save Tokenizer object at target location."""
    assert isinstance(
        obj, AbsTokenizer), "tokenizer.save: input object is not AbsTokenizer instance."
    with open(target, 'wb') as fo:
        pickle.dump(obj, fo)


def load(src: str) -> AbsTokenizer:
    assert os.path.isfile(src)
    with open(src, 'rb') as fi:
        tokenizer = pickle.load(fi)

    assert isinstance(
        tokenizer, AbsTokenizer), "tokenizer.load: loaded object is not AbsTokenizer instance."
    return tokenizer
