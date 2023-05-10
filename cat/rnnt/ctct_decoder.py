# Copyright 2023 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""CTC-Transducer decoder with prefix beam search impl.

Assume <blk> = <s> = 0. 

CTC-T model is identity to CTC if the 
... predictor and joiner are none.
"""


from .rnnt_decoder import *
import math


NEGINF = float("-inf")


def logaddexp(a, b):
    a = float(a)
    b = float(b)

    if a == b == NEGINF:
        return NEGINF

    if a < b:
        a, b = b, a

    return a + math.log1p(math.exp(b - a))


class CTCHypo:
    def __init__(
        self,
        tokens: torch.LongTensor,
        state: Dict[str, Union[AbsStates, torch.Tensor]],
        pred: Optional[Tuple[int]] = None,
    ) -> None:
        self.last_tok = tokens[-1:]
        self.pred = tuple(tokens.tolist()) if pred is None else pred
        self.pn_state = state
        self.am_score = 0.0

    @property
    def score(self) -> Union[float, torch.FloatTensor]:
        return self.am_score

    def clone(self):
        new_hypo = self.__class__(self.last_tok, self.pn_state, self.pred)
        new_hypo.am_score = self.am_score + 0.0
        return new_hypo

    def get_pred_token(self, return_tensor: bool = False):
        if return_tensor:
            return self.last_tok.new_tensor(self.pred)
        else:
            return list(self.pred)

    def __len__(self) -> int:
        return len(self.pred)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.pred}"


class DualStateHypo(CTCHypo):
    def __init__(
        self,
        tokens: torch.LongTensor,
        state: Dict[str, Union[AbsStates, torch.Tensor]],
        pred: Optional[Tuple[int]] = None,
    ) -> None:
        super().__init__(tokens, state, pred)
        # clarify the prob end with non-blank label
        self.am_score_nb = 0.0

    @property
    def score(self) -> Union[float, torch.FloatTensor]:
        return logaddexp(self.am_score, self.am_score_nb)

    def clone(self):
        new_hypo = super().clone()
        new_hypo.am_score_nb = self.am_score_nb + 0.0
        return new_hypo

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(score_b={self.am_score:.3f}, score_nb={self.am_score_nb:.3f}, hypo={self.pred})"


class CTCTDecoder(RNNTDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.lm is None
        ), f"{self.__class__.__name__}: lm fusion is not support yet."

    def batch_decode(self, *args, **kwargs) -> List[List[Hypothesis]]:
        return self.batch_decode_intuitive(*args, **kwargs)

    def batch_decode_intuitive(
        self, encoder_out: torch.Tensor, frame_lens: Optional[torch.Tensor] = None
    ) -> List[List[CTCHypo]]:
        """Beam search decoding in intuitive way."""
        prev_token = encoder_out.new_empty((1, 1), dtype=torch.long)
        prev_token[0][0] = self.bos_id
        if frame_lens is None:
            frame_lens = prev_token.new_full(
                (encoder_out.size(0),), fill_value=encoder_out.size(1)
            )
        frame_lens = frame_lens.to(torch.int)

        res = []
        for n in range(encoder_out.size(0)):
            state = self.predictor.init_states()()
            h_init = DualStateHypo(prev_token[0], state, (self.bos_id,))
            h_init.am_score = NEGINF
            h_init.am_score_nb = 0.0

            A = {(self.bos_id,): h_init}  # type: Dict[Tuple[int], DualStateHypo]
            A_next = {}  # type: Dict[Tuple[int], DualStateHypo]
            A_cache = {}  # type: Dict[Tuple[int], DualStateHypo]
            for t in range(frame_lens[n].item()):
                for l, hyp in A.items():
                    prev_token[0][0] = l[-1]
                    pn_out, pn_state = self.predictor(prev_token, hyp.pn_state)
                    v = self.joiner(encoder_out[n, t], pn_out[0][0])

                    k = min(self.beam_size, v.size(0) - 1)
                    lprobs, indices = torch.topk(v[1:], k)
                    # blank is sliced from topk, here we must add 1 back
                    indices += 1

                    for i in range(k):
                        tok = indices[i].item()
                        lp = l + (tok,)

                        if tok == l[-1]:
                            prob_cur_path = hyp.am_score + lprobs[i]
                        else:
                            prob_cur_path = hyp.score + lprobs[i]

                        # add new hypo into A_next
                        if lp in A_next:
                            new_hypo = A_next[lp]
                            new_hypo.am_score_nb = logaddexp(
                                new_hypo.am_score_nb, prob_cur_path
                            )
                        else:
                            new_hypo = DualStateHypo(
                                indices[i].unsqueeze(0), pn_state, lp
                            )
                            new_hypo.am_score_nb = prob_cur_path
                            new_hypo.am_score = NEGINF
                            A_next[lp] = new_hypo

                    # add blank
                    if l in A_next:
                        orin_hyp = A_next[l]
                        orin_hyp.am_score = logaddexp(
                            orin_hyp.am_score, hyp.score + v[0]
                        )

                        """
                            Since we use the same token for both <sos> and <blk>,
                            we have to consider the situation, otherwise the probs are incorrect.
                        """
                        if len(l) > 1:
                            orin_hyp.am_score_nb = logaddexp(
                                orin_hyp.am_score_nb, hyp.am_score_nb + v[l[-1]]
                            )
                    else:
                        # re-use the hypo in A
                        hyp.am_score = hyp.score + v[0]
                        if len(l) > 1:
                            hyp.am_score_nb = hyp.am_score_nb + v[l[-1]]
                        else:
                            hyp.am_score_nb = NEGINF
                        A_next[l] = hyp

                A_cache = A_next
                A_next = {
                    l: hyp
                    for l, hyp in sorted(
                        A_cache.items(), key=lambda item: item[1].score, reverse=True
                    )[: self.beam_size]
                }
                A = A_next
                A_next = {}

            res.append(
                sorted(A.values(), key=lambda item: item.score, reverse=True)[
                    : self.nbest
                ]
            )
        return res
