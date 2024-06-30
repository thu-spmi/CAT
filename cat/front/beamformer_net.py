from collections import OrderedDict
from typing import Optional

import torch
from torch_complex.tensor import ComplexTensor

from .dnn_beamformer import DNN_Beamformer
#from dnn_wpe import DNN_WPE
from .dnn_wpe_new import DNN_WPE
from .stft import Stft

class BeamformerNet(torch.nn.Module):
    """TF Masking based beamformer"""

    def __init__(
        self,
        num_spk: int = 1,
        normalize_input: bool = False,
        mask_type: str = "IPM^2",
        loss_type: str = "spectrum",
        # STFT options
        n_fft: int = 512,
        # win_length: int = None,
        # hop_length: int = 128,
        # center: bool = True,
        # window: Optional[str] = "hann",
        # normalized: bool = False,
        # onesided: bool = True,
        # Dereverberation options
        use_wpe: bool = False,
        wnet_type: str = "blstmp",
        wlayers: int = 3,
        wunits: int = 300,
        wprojs: int = 320,
        wdropout_rate: float = 0.0,
        taps: int = 5,
        delay: int = 3,
        use_dnn_mask_for_wpe: bool = False,
        wnonlinear: str = "crelu",
        # Beamformer options
        use_beamformer: bool = True,
        bnet_type: str = "blstmp",
        blayers: int = 3,
        bunits: int = 300,
        bprojs: int = 320,
        badim: int = 320,
        bdropout_rate=0.0,
        fnet_type: str = "blstmp",
        flayers: int = 3,
        funits: int = 300,
        fprojs: int = 320,
        fdropout_rate=0.0,
        ref_channel: int = 0,
        use_noise_mask: bool = True,
        bnonlinear: str = "sigmoid",
        beamformer_type="mvdr",
    ):
        super(BeamformerNet, self).__init__()

        self.mask_type = mask_type
        self.beamformer_type = beamformer_type
        self.loss_type = loss_type
        if loss_type not in ("mask_mse", "spectrum"):
            raise ValueError("Unsupported loss type: %s" % loss_type)

        self.num_spk = num_spk
        self.num_bin = n_fft // 2 + 1

        # self.stft = Stft(
        #     n_fft=n_fft,
        #     win_length=win_length,
        #     hop_length=hop_length,
        #     center=center,
        #     window=window,
        #     normalized=normalized,
        #     onesided=onesided,
        # )

        self.normalize_input = normalize_input
        self.use_beamformer = use_beamformer
        self.use_wpe = use_wpe

        if self.use_wpe:
            if use_dnn_mask_for_wpe:
                # Use DNN for power estimation
                iterations = 1
            else:
                # Performing as conventional WPE, without DNN Estimator
                iterations = 2

            self.wpe = DNN_WPE(
                wtype=wnet_type,
                widim=self.num_bin,
                wunits=wunits,
                wprojs=wprojs,
                wlayers=wlayers,
                taps=taps,
                delay=delay,
                dropout_rate=wdropout_rate,
                iterations=iterations,
                use_dnn_mask=use_dnn_mask_for_wpe,
                nonlinear=wnonlinear,
            )
        else:
            self.wpe = None

        self.ref_channel = ref_channel
        if self.use_beamformer:
            self.beamformer = DNN_Beamformer(
                idim=self.num_bin,
                
                btype=bnet_type,
                bunits=bunits,
                bprojs=bprojs,
                blayers=blayers,
                bdropout_rate=bdropout_rate,
               
                ftype=fnet_type,
                funits=funits,
                fprojs=fprojs,
                flayers=flayers,
                fdropout_rate=fdropout_rate,
              
                num_spk=num_spk,
                use_noise_mask=use_noise_mask,
                badim=badim,
                ref_channel=ref_channel,
                beamformer_type=beamformer_type,
                btaps=taps,
                bdelay=delay,
            )
        else:
            self.beamformer = None

    def forward(self, input_spectrum: torch.Tensor, flens: torch.Tensor):
        """Forward.
        Args:
            input (torch.Tensor): mixed spectrum
            flens (torch.Tensor): input lengths [Batch]
        Returns:
            enhanced speech  (single-channel):
                torch.Tensor or List[torch.Tensor]
            output lengths
            predcited masks: OrderedDict[
                'dereverb': torch.Tensor(Batch, Frames, Channel, Freq),
                'spk1': torch.Tensor(Batch, Frames, Channel, Freq),
                'spk2': torch.Tensor(Batch, Frames, Channel, Freq),
                ...
                'spkn': torch.Tensor(Batch, Frames, Channel, Freq),
                'noise1': torch.Tensor(Batch, Frames, Channel, Freq),
            ]
        """
        # (Batch, Frames, Freq) or (Batch, Frames, Channels, Freq)
        input_spectrum = ComplexTensor(input_spectrum[..., 0], input_spectrum[..., 1])
        if self.normalize_input:
            input_spectrum = input_spectrum / abs(input_spectrum).max()

        # Shape of input spectrum must be (B, T, F) or (B, T, C, F)
        assert input_spectrum.dim() in (3, 4), input_spectrum.dim()
        enhanced = input_spectrum
        masks = OrderedDict()

        if self.training and self.loss_type.startswith("mask"):
            # Only estimating masks for training
            if self.use_wpe:
                if input_spectrum.dim() == 3:
                    mask_w, flens = self.wpe.predict_mask(
                        input_spectrum.unsqueeze(-2), flens
                    )
                    mask_w = mask_w.squeeze(-2)
                elif input_spectrum.dim() == 4:
                    if self.use_beamformer:
                        enhanced, flens, mask_w, _ = self.wpe(input_spectrum, flens)
                    else:
                        mask_w, flens = self.wpe.predict_mask(input_spectrum, flens)

                if mask_w is not None:
                    masks["dereverb"] = mask_w

            if self.use_beamformer and input_spectrum.dim() == 4:
                masks_b, flens = self.beamformer.predict_mask(enhanced, flens)
                for spk in range(self.num_spk):
                    masks["spk{}".format(spk + 1)] = masks_b[spk]
                if len(masks_b) > self.num_spk:
                    masks["noise1"] = masks_b[self.num_spk]

            return None, flens, masks

        else:
            # Performing both mask estimation and enhancement
            if input_spectrum.dim() == 3:
                print("single")
                # single-channel input (B, T, F)
                if self.use_wpe:
                    enhanced, flens, mask_w, _ = self.wpe(
                        input_spectrum.unsqueeze(-2), flens
                    )
                    enhanced = enhanced.squeeze(-2)
                    if mask_w is not None:
                        masks["dereverb"] = mask_w.squeeze(-2)
            else:
                # multi-channel input (B, T, C, F)
                #print("B,T,C,F")
                #print(input_spectrum.shape)
                #print("multich")
                # 1. WPE
                if self.use_wpe:
                    #print("wpe!")
                    # print(input_spectrum)
                    enhanced, flens, mask_w, _  = self.wpe(input_spectrum, flens)
                    if mask_w is not None:
                        masks["dereverb"] = mask_w

                # 2. Beamformer
                if self.use_beamformer:
                    # enhanced: (B, T, C, F) -> (B, T, F)
                    if self.beamformer_type == "filter":
                        enhanced = self.beamformer(enhanced, flens)
                        if isinstance(enhanced, list):
                        # multi-speaker output
                            enhanced = [torch.stack([enh.real, enh.imag], dim=-1) for enh in enhanced]
                        else:
                        # single-speaker output
                            enhanced = torch.stack([enhanced.real, enhanced.imag], dim=-1).float()
                        return enhanced
                    else:
                        enhanced, flens, masks_b = self.beamformer(enhanced, flens)
                        for spk in range(self.num_spk):
                            masks["spk{}".format(spk + 1)] = masks_b[spk]
                        if len(masks_b) > self.num_spk:
                            masks["noise1"] = masks_b[self.num_spk]

        # Convert ComplexTensor to torch.Tensor
        # (B, T, F) -> (B, T, F, 2)
        if isinstance(enhanced, list):
            # multi-speaker output
            enhanced = [torch.stack([enh.real, enh.imag], dim=-1) for enh in enhanced]
        else:
            # single-speaker output
            enhanced = torch.stack([enhanced.real, enhanced.imag], dim=-1).float()
        return enhanced, flens, masks

    # def forward_rawwav(self, input: torch.Tensor, flens: torch.Tensor, ilens):
    #     """Output with wavformes.
    #     Args:
    #         input (torch.Tensor): mixed speech [Batch, Nsample, Channel]
    #         ilens (torch.Tensor): input lengths [Batch]
    #     Returns:
    #         predcited speech wavs (single-channel):
    #             torch.Tensor(Batch, Nsamples), or List[torch.Tensor(Batch, Nsamples)]
    #         output lengths
    #         predcited masks: OrderedDict[
    #             'dereverb': torch.Tensor(Batch, Frames, Channel, Freq),
    #             'spk1': torch.Tensor(Batch, Frames, Channel, Freq),
    #             'spk2': torch.Tensor(Batch, Frames, Channel, Freq),
    #             ...
    #             'spkn': torch.Tensor(Batch, Frames, Channel, Freq),
    #             'noise1': torch.Tensor(Batch, Frames, Channel, Freq),
    #         ]
    #     """

    #     # predict spectrum for each speaker
    #     predicted_spectrums, flens, masks = self.forward(input, flens)
    #     #print(predicted_spectrums.shape)
    #     if predicted_spectrums is None:
    #         predicted_wavs = None
    #     elif isinstance(predicted_spectrums, list):
    #         # multi-speaker input
    #         predicted_wavs = [
    #             self.stft.inverse(ps, ilens)[0] for ps in predicted_spectrums
    #         ]
    #     else:
    #         # single-speaker input
    #         predicted_wavs = self.stft.inverse(predicted_spectrums, ilens)[0]

    #     return predicted_wavs, ilens, masks
