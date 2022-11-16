# Author: Huahuan Zheng (maxwellzh@outlook.com)


import os
import sys
import uuid
import argparse
from typing import *
from multiprocessing import Process, Queue


def chunkize_file(files: List[str], num_workers: int) -> List[Tuple[int, int]]:

    f_sizes = []
    for f in files:
        assert os.path.isfile(f)
        f_sizes.append(os.path.getsize(f))

    tot_size = sum(f_sizes)
    chunk_size = tot_size // num_workers

    c_beg = 0
    c_end = chunk_size
    offset = 0
    splits = []

    for _size, f in zip(f_sizes, files):
        with open(f, 'r') as fio:
            while c_end - offset <= _size:
                fio.seek(c_end-offset)
                # read to '\n'
                while True:
                    try:
                        fio.readline()
                        break
                    except UnicodeDecodeError:
                        fio.seek(fio.tell()-1)

                c_end = fio.tell()
                splits.append((c_beg, c_end))
                c_beg = c_end
                c_end = c_beg + chunk_size
        offset += _size
    if c_end != tot_size:
        splits.append((c_beg, tot_size))
    return splits


def dispatch_jobs(num_jobs: int, num_workers: int) -> List[Tuple[int, int]]:
    if num_workers > num_jobs:
        num_workers = num_jobs

    interval = num_jobs // num_workers
    indices = [interval * i for i in range(num_workers+1)]
    indices[-1] = num_jobs
    return [(indices[i], indices[i+1]) for i in range(num_workers)]


class TextLoader:
    def __init__(self, f_tknz: str) -> None:
        self._tknz = tknz.load(f_tknz)

    @staticmethod
    def count_size(file) -> int:
        return os.path.getsize(file)

    def __call__(self, corpus: str, _offset: int = 0, _end: int = -1):
        if _end == -1:
            _end = os.path.getsize(corpus)
        if _offset >= _end:
            return

        with open(corpus, 'r') as fi:
            fi.seek(_offset)
            while line := fi.readline():
                yield self._tknz.encode(line.strip())
                if fi.tell() >= _end:
                    break
        return


class BinLoader:
    @staticmethod
    def count_size(file) -> int:
        return len(CorpusDataset(file))

    def __call__(self, corpus: str, _offset: int = 0, _cnt: int = -1):
        data = CorpusDataset(corpus)
        if _offset >= len(data):
            return
        for i in range(len(data)):
            if i < _offset:
                continue
            if i == _cnt:
                break

            yield data[i][0].tolist()
        return


def process_worker(args: argparse.Namespace, p_range: Tuple[int, int], q_out: Queue, mapping: Dict[int, str] = {}):
    def _int2str(x: int) -> str:
        return mapping.get(x, str(x))

    rm_empty = not args.keep_empty_line
    if args.istext:
        loader = TextLoader(args.tokenizer)
    else:
        loader = BinLoader()

    cache = f'/tmp/corpus2index-{uuid.uuid4()}.tmp'
    assert os.access('/tmp', os.W_OK), f"cannot write to /tmp"
    offset = 0
    idx_beg, idx_end = p_range
    with open(cache, 'w') as fo:
        for file in args.input:
            for tokens in loader(file, idx_beg - offset, idx_end - offset):
                if tokens == [] and rm_empty:
                    continue
                fo.write(' '.join(_int2str(x) for x in tokens)+'\n')
            offset += loader.count_size(file)
            if offset >= idx_end:
                break

    q_out.put(cache, block=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, nargs='+',
                        help="Input corpus dataset file.")
    parser.add_argument("-t", action="store_true", dest="istext",
                        help="Identify the input to be text instead of binary file. Used with --tokenizer")
    parser.add_argument("--tokenizer", type=str,
                        help="Tokenizer model file. See cat/shared/tokenizer.py for details.")
    parser.add_argument("--keep-empty-line", action="store_true",
                        help="Keep empty lines instead removing them (default).")
    parser.add_argument("--map", nargs='*', type=str,
                        help="Map index to str, split by ':'. "
                        "e.g. map 0 to whitespace '--map 0:'; "
                        "     map 0 to whitespace and map 1 to <unk> '--map 0: \"1:<unk>\"'")
    args = parser.parse_args()

    stdin = (len(args.input) == 1 and args.input[0] == '/dev/stdin')
    mapping = {}
    if args.map is not None:
        for _m in args.map:
            if ':' not in _m:
                raise ValueError(f"No colon ':' found in --map={_m}")
            index, string = _m.split(':', maxsplit=1)
            try:
                mapping[int(index)] = string
            except ValueError:
                raise ValueError(
                    f"failed to read from mapping string \"--mapping={mapping}\"")

    from cat.shared.data import CorpusDataset
    from cat.shared import tokenizer as tknz
    if not stdin:
        if args.istext:
            tot_size = sum(os.path.getsize(f) for f in args.input)
            num_process = max(min(os.cpu_count()//2, tot_size//(1024*1024)), 1)
            workerloads = chunkize_file(args.input, num_process)
        else:
            tot_lines = sum(len(CorpusDataset(dataset))
                            for dataset in args.input)
            num_process = max(min(os.cpu_count()//2, tot_lines//1000), 1)
            workerloads = dispatch_jobs(tot_lines, num_process)

    if stdin:
        try:
            assert args.tokenizer is not None
            tokenizer = tknz.load(args.tokenizer)
            for line in sys.stdin:
                sys.stdout.write(' '.join(mapping.get(i, str(i))
                                 for i in tokenizer.encode(line)))
                sys.stdout.write('\n')
        except IOError:
            sys.exit(0)
        sys.exit(0)
    try:
        q = Queue(maxsize=1)
        p = []
        for i in range(num_process):
            p.append(Process(
                target=process_worker,
                args=(args, workerloads[i], q, mapping)
            ))
            p[-1].start()

        for _ in range(num_process):
            f_cache = q.get(block=True)
            with open(f_cache, 'r') as fi:
                for line in fi:
                    sys.stdout.write(line)
            os.remove(f_cache)

        for _p in p:
            _p.join()

    except IOError:
        pass
    finally:
        for _p in p:
            _p.terminate()
        del q
