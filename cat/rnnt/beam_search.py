"""Beam search for Transducer sequence.

For the performance/speed of decoding, only
... RNA (Recurrent neural aligner, a.k.a. monototic topo)
... decoding is support.

For other algorithm, check previous commits
https://github.com/maxwellzh/Transducer-dev/blob/e711c3b5582d981afe40b8453a1f268025f8de9a/cat/rnnt/beam_search_transducer.py

where there are:
- vallina decoding (with prefix merge)
- latency controlled decoding (with prefix merge)
- alignment-length synchronous decoding

Author: Huahuan Zhengh (maxwellzh@outlook.com)
"""

from .joiner import AbsJointNet
from cat.shared.decoder import (
    LSTM,
    AbsDecoder,
    AbsStates
)

import os
from typing import *

import torch


def logaddexp(a: torch.Tensor, b: torch.Tensor):
    if a.dtype == torch.float:
        return torch.logaddexp(a, b)
    elif a.dtype == torch.half:
        if a < b:
            a, b = b, a
        # a + log(1 + exp(b-a))
        return a + (1 + (b-a).exp()).log()
    else:
        raise ValueError


def hash_tensor(t: torch.LongTensor) -> Tuple[int]:
    return tuple(t.cpu().tolist())


class Hypothesis():
    def __init__(
            self,
            pred: torch.LongTensor,
            log_prob: Union[torch.Tensor, float],
            cache: Dict[str, Union[AbsStates, torch.Tensor]],
            lm_score: Union[torch.Tensor, float] = 0.0) -> None:

        self._last_token = pred[-1:]
        self.pred = hash_tensor(pred)
        self.log_prob = log_prob + 0.   # implictly clone
        self.cache = cache
        self.lm_score = lm_score + 0.

    @property
    def score(self):
        return self.log_prob + self.lm_score

    def get_pred_token(self, return_tensor: bool = False):
        if return_tensor:
            return self._last_token.new_tensor(self.pred)
        else:
            return list(self.pred)

    def clone(self):
        new_hypo = Hypothesis(
            self._last_token,
            self.log_prob,
            self.cache.copy(),
            self.lm_score
        )
        new_hypo.pred = self.pred[:]
        return new_hypo

    def __add__(self, rhypo: "Hypothesis"):
        new_hypo = self.clone()
        new_hypo.log_prob = logaddexp(new_hypo.log_prob, rhypo.log_prob)
        return new_hypo

    def add_(self, rhypo: "Hypothesis"):
        '''in-place version of __add__'''
        self.log_prob = logaddexp(self.log_prob, rhypo.log_prob)
        return self

    def add_token(self, tok: torch.LongTensor):
        self._last_token = tok.view(1)
        self.pred += (tok.item(),)

    def __len__(self) -> int:
        return len(self.pred)

    def __repr__(self) -> str:
        return f"Hypothesis({self.pred}, score={self.score:.2f})"


class PrefixCacheDict():
    """
    This use a map-style way to store the cache.
    Compared to tree-like structure, thie would be less efficient when the tree is 
    quite large. But more efficient when it is small.
    """

    def __init__(self) -> None:
        self._cache = {}    # type: Dict[Tuple[int], Dict]

    def __contains__(self, pref: Tuple[int]) -> bool:
        return pref in self._cache

    def update(self, pref: Tuple[int], new_cache: dict):
        if pref in self._cache:
            self._cache[pref].update(new_cache.copy())
        else:
            self._cache[pref] = new_cache.copy()

    def fetch(self, pref: Tuple[int]) -> Union[None, dict]:
        '''Get cache. If there isn't such prefix, return None.
        '''
        if pref in self._cache:
            return self._cache[pref]
        else:
            return None

    def prune_except(self, legal_prefs: List[Tuple[int]]):
        new_cache = {}
        for pref in legal_prefs:
            if pref in self._cache:
                new_cache[pref] = self._cache[pref]
        del self._cache
        self._cache = new_cache

    def prune_shorterthan(self, L: int):
        torm = [key for key in self._cache if len(key) < L]
        for k in torm:
            del self._cache[k]

    def __str__(self) -> str:
        cache = {}
        for k in self._cache.keys():
            cache[k] = {}
            for _k in self._cache[k].keys():
                cache[k][_k] = '...'

        return str(cache)

# TODO:
# 1. add a interface of decoder
# 2.[done] batch-fly the decoding
# 3.[done] interface for introducing external LM(s)
# 4.[done] rename tn -> encoder


class BeamSearcher:

    def __init__(
        self,
        predictor: AbsDecoder,
        joiner: AbsJointNet,
        blank_id: int = 0,
        bos_id: int = 0,
        beam_size: int = 5,
        nbest: int = -1,
        lm_module: Optional[AbsDecoder] = None,
        alpha: Optional[float] = 0.,
        beta: Optional[float] = 0.,
        est_ilm: bool = False,
        ilm_weight: Optional[float] = 0.
    ):
        assert blank_id == bos_id

        if alpha == 0.0:
            # NOTE: alpha = 0 will disable LM interation whatever beta is.
            lm_module = None

        if lm_module is None:
            alpha = 0.0
            beta = 0.0

        self.predictor = predictor
        self.joiner = joiner
        self.blank_id = blank_id
        self.bos_id = bos_id
        self.beam_size = beam_size
        if nbest == -1:
            nbest = beam_size
        self.nbest = min(nbest, beam_size)
        self.lm = lm_module
        self.alpha_ = alpha

        self.beta_ = beta
        self.est_ilm = est_ilm
        self.ilm_weight = ilm_weight
        if ilm_weight == 0.0:
            self.est_ilm = False

    def __call__(self, enc_out: torch.Tensor, frame_lens: Optional[torch.Tensor] = None) -> List[Tuple[List[List[int]], List[float]]]:
        hypos = self.batch_decode(enc_out, frame_lens)

        return [
            (
                [hypo.get_pred_token()[1:] for hypo in _hyps],
                [hypo.score.item() for hypo in _hyps]
            )
            for _hyps in hypos
        ]

    def batch_decode(self, encoder_out: torch.Tensor, frame_lens: Optional[torch.Tensor] = None) -> List[List[Hypothesis]]:
        """
        An implementation of batched RNA decoding

        encoder_out: (N, T, H)
        """
        use_lm = self.lm is not None
        if isinstance(self.predictor, LSTM):
            if use_lm and not isinstance(self.lm, LSTM):
                fixlen_state = False
            else:
                fixlen_state = True
        else:
            fixlen_state = False

        n_batches = encoder_out.size(0)
        dummy_token = encoder_out.new_empty(1, dtype=torch.long)
        idx_seq = torch.arange(n_batches)
        if frame_lens is None:
            n_max_frame_length = encoder_out.size(1)
            frame_lens = dummy_token.new_full(
                (n_batches,), fill_value=n_max_frame_length)
        else:
            frame_lens = frame_lens.clone()
            n_max_frame_length = frame_lens.max().int()

        Beams = [
            [
                Hypothesis(
                    pred=dummy_token.new_tensor([self.bos_id]),
                    log_prob=0.0,
                    cache={'pn_state': self.predictor.init_states()}
                )
            ]
            for _ in range(n_batches)
        ]

        if use_lm:
            for idx in range(n_batches):
                Beams[idx][0].cache.update(
                    {'lm_state': self.lm.init_states()})
        pref_cache = PrefixCacheDict()

        for t in range(n_max_frame_length):
            # concat beams in the batch to one group
            idx_ongoing_seq = idx_seq[frame_lens > 0]
            n_seqs = idx_ongoing_seq.size(0)
            batched_beams = sum((Beams[i_] for i_ in idx_ongoing_seq), [])
            # n_beams = len(batched_beams)
            group_uncached, group_cached = group_to_batch(
                batched_beams,
                dummy_token,
                pref_cache,
                statelen_fixed=fixlen_state)
            group_beams = group_uncached + group_cached

            idxbeam2srcidx = []   # len: n_beams
            group_pn_out = []     # len: len(group_beams)
            group_lm_out = []

            n_group_uncached = len(group_uncached)
            # In following loop, we do:
            # 1. compute predictor output for beams not in cache
            # 2. fetch output for beams in cache
            for i, (g_index, g_tokens, g_states) in enumerate(group_beams):
                idxbeam2srcidx += g_index
                if i < n_group_uncached:
                    pn_out, pn_state = self.predictor(
                        g_tokens, g_states['pn_state']())
                    if use_lm:
                        lm_out, lm_state = self.lm.get_log_prob(
                            g_tokens, g_states['lm_state']())
                    # add into cache
                    for bid, absidx in enumerate(g_index):
                        cur_cache = {
                            'pn_out': pn_out[bid:bid+1],
                            'pn_state': self.predictor.get_state_from_batch(pn_state, bid)
                        }
                        if use_lm:
                            cur_cache.update({
                                'lm_out': lm_out[bid:bid+1],
                                'lm_state': self.lm.get_state_from_batch(lm_state, bid)
                            })
                        pref_cache.update(
                            batched_beams[absidx].pred, cur_cache)
                else:
                    pn_out = g_tokens['pn_out']
                    pn_state = g_states['pn_state']()
                    if use_lm:
                        lm_out = g_tokens['lm_out']
                        lm_state = g_states['lm_state']()

                group_pn_out.append(pn_out)
                if use_lm:
                    group_lm_out.append(lm_out)

            # pn_out: (n_beams, 1, H)
            pn_out = torch.cat(group_pn_out, dim=0)
            '''
            Since we merge all hypos in a batch into one group, 
            Frames in enc_out may be mapped to various number of frames of pn_out
            e.g. a batch with 2 utterances, enc_out: (2, T, H), where the first utt 
                has 2 hypos and the second one has 6 hypos (i.e. n_beams = 2 + 6 = 8)
                To get the expand_enc_out:
                    enc_out[0, ...] expand twice -> (2, T, H)
                    enc_out[1, ...] expand 6 times -> (6, T, H)
                    -> concat them -> (8, T, H)
                note that indices in merged beams are not consistent to that of expand_enc_out,
                so we have to re-arange it via expand_enc_out[idxbeam2srcidx]
            '''
            # expand_enc_out: (n_beams, 1, H)
            expand_enc_out = torch.cat(
                [encoder_out[b:b+1, t:t+1, :].expand(len(Beams[b]), -1, -1)
                 for b in idx_ongoing_seq], dim=0)[idxbeam2srcidx]
            # log_prob: (n_beams, 1, 1, V) -> (n_beams, V)
            log_prob = self.joiner(
                expand_enc_out, pn_out).squeeze(1).squeeze(1)

            # combine_score: (n_beams, V)
            combine_score = log_prob + 0.
            for i, b in enumerate(idxbeam2srcidx):
                combine_score[i] += batched_beams[b].log_prob + \
                    batched_beams[b].lm_score

            if use_lm:
                # lm_score: (n_beams, V)
                lm_score = self.beta_ + self.alpha_ * \
                    torch.cat(group_lm_out, dim=0).squeeze(1)
                if self.blank_id == 0:
                    combine_score[:, 1:] += lm_score[:, 1:]
                else:
                    raise NotImplementedError

            if self.est_ilm:
                ilm_score = self.joiner.impl_forward(
                    torch.zeros_like(expand_enc_out), pn_out).squeeze(1).squeeze(1)
                if self.blank_id == 0:
                    # rm the blank symbol
                    ilm_score[:, 0] = 0.
                    ilm_score[:, 1:] = self.ilm_weight * \
                        ilm_score[:, 1:].log_softmax(dim=1)
                else:
                    raise NotImplementedError

                combine_score[:, 1:] += ilm_score[:, 1:]

            V = combine_score.size(-1)
            offset = 0
            min_len = n_max_frame_length
            srcidx2beamidx = {i_: idx for idx, i_ in enumerate(idxbeam2srcidx)}
            for s_ in range(n_seqs):
                idxinbatch = idx_ongoing_seq[s_]
                # map2rearangeidx tells which beam in combine_score derived from
                # ... the same utterance in the batch.
                map2rearangeidx = [srcidx2beamidx[offset+beamidx]
                                   for beamidx in range(len(Beams[idxinbatch]))]
                # flattened_pos: (K, )
                flatten_score = combine_score[map2rearangeidx].flatten()
                k = min(self.beam_size, flatten_score.numel())
                _, flatten_pos = torch.topk(flatten_score, k=k)
                flatten_pos = torch.sort(flatten_pos)[0]
                # idx_beam, tokens: (K, )
                idx_beam = [map2rearangeidx[i_] for i_ in torch.div(
                    flatten_pos, V, rounding_mode='floor')]
                tokens = flatten_pos % V
                A = {}      # type: Dict[Tuple[int], Hypothesis]
                for i in range(k):
                    hasduplicate = (i < k-1) and (idx_beam[i] == idx_beam[i+1])

                    cur_hypo = batched_beams[idxbeam2srcidx[idx_beam[i]]]
                    new_log_prob = log_prob[idx_beam[i], tokens[i]]
                    if tokens[i] == self.blank_id:
                        if cur_hypo.pred in A:
                            A[cur_hypo.pred].log_prob = logaddexp(
                                A[cur_hypo.pred].log_prob, cur_hypo.log_prob+new_log_prob)
                            continue
                        elif hasduplicate:
                            cur_hypo = cur_hypo.clone()
                    else:
                        if (new_pred := cur_hypo.pred + (tokens[i].item(), )) in A:
                            A[new_pred].log_prob = logaddexp(
                                A[new_pred].log_prob, cur_hypo.log_prob+new_log_prob)
                            continue
                        elif hasduplicate:
                            cur_hypo = cur_hypo.clone()

                        # the order of following two lines cannot be changed
                        cur_hypo.cache = pref_cache.fetch(cur_hypo.pred)
                        # cur_hypo.add_token(tokens[i])
                        cur_hypo._last_token = tokens[i].view(1)
                        cur_hypo.pred = new_pred
                        if self.est_ilm:
                            cur_hypo.lm_score += ilm_score[idx_beam[i], tokens[i]]
                        if use_lm:
                            cur_hypo.lm_score += lm_score[idx_beam[i], tokens[i]]
                    cur_hypo.log_prob += new_log_prob
                    A[cur_hypo.pred] = cur_hypo

                offset += len(Beams[idxinbatch])
                Beams[idxinbatch] = list(A.values())
                min_len = min(min_len, min(len(pred) for pred in A))
            pref_cache.prune_shorterthan(min_len)

            frame_lens -= 1

        return [
            sorted(B_, key=lambda item: item.score, reverse=True)[:self.nbest]
            for B_ in Beams
        ]


def group_to_batch(hypos: List[Hypothesis], dummy_tensor: torch.Tensor = None, prefix_cache: PrefixCacheDict = None, statelen_fixed: bool = False) -> Tuple[List[int], torch.Tensor, Dict[str, AbsStates]]:
    """Group the hypothesis in the list into batch with their hidden states

    Args:
        hypos
        dummy_tensor : claim the device of created batches, if None, use cpu

    Returns:
        if prefix_cache=None:
            [(indices, batched_tokens, batched_states), ... ]
        else
            [(indices, batched_tokens, batched_states), ... ], [(indices, batched_output, batched_states), ...]
        indices (list(int)): index of hypo in the original input hypos after batching
        batched_tokens (torch.LongTensor): [N, 1]
        batched_states (Dict[str, AbsStates]): the hidden states of the hypotheses, depending on the prediction network type.
        batched_output (Dict[str, torch.Tensor]): the cached output being batched
        statelen_fixed (bool, default False): whether to group the states by hypo lengths, 
            if set True, this would slightly speedup training, however it requires the cache state to be of fixed length with variable seq lengths (like LSTM)
    """
    if dummy_tensor is None:
        dummy_tensor = torch.empty(1)

    hypos_with_index = list(enumerate(hypos))
    # split hypos into two groups, one with cache hit and the other the cache doesn't.
    if prefix_cache is not None:
        in_cache = []
        for id, hypo in hypos_with_index:
            if hypo.pred in prefix_cache:
                in_cache.append((id, hypo))
        for id, _ in in_cache[::-1]:
            hypos_with_index.pop(id)

    # group that cache doesn't hit
    batched_out = []
    if statelen_fixed:
        groups_uncached = [hypos_with_index] if len(
            hypos_with_index) > 0 else []
    else:
        groups_uncached = groupby(
            hypos_with_index, key=lambda item: len(item[1]))
    for _hypos_with_index in groups_uncached:
        _index, _hypos = list(zip(*_hypos_with_index))
        _batched_tokens = torch.cat(
            [hyp._last_token for hyp in _hypos], dim=0).view(-1, 1)
        _batched_states = {
            _key: _state.batching(
                [_hyp.cache[_key]for _hyp in _hypos]
            ) for _key, _state in _hypos[0].cache.items()
            if isinstance(_state, AbsStates)}     # type: Dict[str, AbsStates]

        batched_out.append((list(_index), _batched_tokens, _batched_states))

    if prefix_cache is None:
        return batched_out
    elif in_cache == []:
        return batched_out, []
    else:
        cached_out = []
        if statelen_fixed:
            groups_cached = [in_cache] if len(in_cache) > 0 else []
        else:
            groups_cached = groupby(in_cache, key=lambda item: len(item[1]))
        for _hypos_with_index in groups_cached:
            _index, _hypos = list(zip(*_hypos_with_index))
            # type: List[Dict[str, Union[torch.Tensor, AbsStates]]]
            caches = [prefix_cache.fetch(_hyp.pred) for _hyp in _hypos]
            _batched_out = {}
            _batched_states = {}
            for k in caches[0].keys():
                if isinstance(caches[0][k], AbsStates):
                    _batched_states[k] = caches[0][k].batching(
                        [_cache[k] for _cache in caches])
                else:
                    # [1, 1, H]
                    _batched_out[k] = torch.cat(
                        [_cache[k] for _cache in caches], dim=0)
            cached_out.append((list(_index), _batched_out, _batched_states))

        return batched_out, cached_out


def groupby(item_list: Iterable, key: Callable) -> List[List[Any]]:
    odict = {}  # type: Dict[Any, List]
    for item in item_list:
        _k = key(item)
        if _k not in odict:
            odict[_k] = [item]
        else:
            odict[_k].append(item)
    return list(odict.values())
