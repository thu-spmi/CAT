# Copyright 2023 Tsinghua University
# Apache 2.0.
# Author: Xiangzhu Kong (kongxiangzhu99@gmail.com)

# Acknowledgment:
#   This code is adapted from the TrochAudio project. The original code can be found at https://pytorch.org/audio/0.13.1/compliance.kaldi.html.

# Description:
#   This script provides functions for feature extraction from audio signals, including Short-Time Fourier Transform (STFT),
#   Mel filter bank calculation, and other related operations. It includes the Feature_Trans class for handling various audio
#   transformations and feature extraction steps.

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torchaudio.compliance.kaldi import _get_epsilon,\
                    _get_waveform_and_window_properties,\
                    _subtract_column_mean,get_mel_banks,\
                    _get_log_energy,\
                    _next_power_of_2
import math
from typing import Tuple

import torch
import torchaudio
from torch import Tensor


# numeric_limits<float>::epsilon() 1.1920928955078125e-07
EPSILON = torch.tensor(torch.finfo(torch.float).eps)
# 1 milliseconds = 0.001 seconds
MILLISECONDS_TO_SECONDS = 0.001

# window types
HAMMING = "hamming"
HANNING = "hanning"
POVEY = "povey"
RECTANGULAR = "rectangular"
BLACKMAN = "blackman"
WINDOWS = [HAMMING, HANNING, POVEY, RECTANGULAR, BLACKMAN]

def _feature_window_function(
    window_type: str,
    window_size: int,
    blackman_coeff: float,
    device: torch.device,
    dtype: int,
) -> Tensor:
    r"""Returns a window function with the given type and size"""
    if window_type == HANNING:
        return torch.hann_window(window_size, periodic=False, device=device, dtype=dtype)
    elif window_type == HAMMING:
        return torch.hamming_window(window_size, periodic=False, alpha=0.54, beta=0.46, device=device, dtype=dtype)
    elif window_type == POVEY:
        # like hanning but goes to zero at edges
        return torch.hann_window(window_size, periodic=False, device=device, dtype=dtype).pow(0.85)
    elif window_type == RECTANGULAR:
        return torch.ones(window_size, device=device, dtype=dtype)
    elif window_type == BLACKMAN:
        a = 2 * math.pi / (window_size - 1)
        window_function = torch.arange(window_size, device=device, dtype=dtype)
        # can't use torch.blackman_window as they use different coefficients
        return (
            blackman_coeff
            - 0.5 * torch.cos(a * window_function)
            + (0.5 - blackman_coeff) * torch.cos(2 * a * window_function)
        ).to(device=device, dtype=dtype)
    else:
        raise Exception("Invalid window type " + window_type)


def _get_strided(waveform: Tensor, window_size: int, window_shift: int, snip_edges: bool) -> Tensor:
    """
    Returns a tensor containing strided windows from the input waveform.

    Args:
        waveform (Tensor): Input tensor of shape (..., num_samples).
        window_size (int): Frame length.
        window_shift (int): Frame shift.
        snip_edges (bool): Whether to handle edge effects by only outputting frames that fit completely within the waveform.

    Returns:
        Tensor: Tensor of shape (..., m, window_size), where each row is a frame.
    """
    assert waveform.dim() >= 2

    # 计算用于分步查看的步幅
    strides = (window_shift * waveform.stride(-1), waveform.stride(-1))

    # 计算帧的数量（m）
    num_samples = waveform.size(-1)
    if snip_edges:
        if num_samples < window_size:
            return torch.empty(waveform.shape[:-1] + (0, window_size), dtype=waveform.dtype, device=waveform.device)
        else:
            m = 1 + (num_samples - window_size) // window_shift
    else:
        reversed_waveform = torch.flip(waveform, [-1])
        m = (num_samples + (window_shift // 2)) // window_shift
        pad = window_size // 2 - window_shift // 2
        pad_right = reversed_waveform
        if pad > 0:
            pad_left = reversed_waveform[..., -pad:]
            waveform = torch.cat((pad_left, waveform, pad_right), dim=-1)
        else:
            waveform = torch.cat((waveform[..., -pad:], pad_right), dim=-1)

    # 为分步查看创建尺寸
    sizes = waveform.shape[:-1] + (m, window_size)
    # 计算新的 strides
    strides = waveform.stride()[:-1] + (window_shift * waveform.stride(-1), waveform.stride(-1))
    # 使用 as_strided 保留重叠的帧
    return waveform.as_strided(sizes, strides)


def _get_window(
    waveform: Tensor,
    padded_window_size: int,
    window_size: int,
    window_shift: int,
    window_type: str,
    blackman_coeff: float,
    snip_edges: bool,
    raw_energy: bool,
    energy_floor: float,
    dither: float,
    remove_dc_offset: bool,
    preemphasis_coefficient: float,
) -> Tuple[Tensor, Tensor]:
    r"""Gets a window and its log energy

    Returns:
        (Tensor, Tensor): strided_input of size (m, ``padded_window_size``) and signal_log_energy of size (m)
    """
    device, dtype = waveform.device, waveform.dtype
    epsilon = _get_epsilon(device, dtype)

    # size (..., m, window_size)
    strided_input = _get_strided(waveform, window_size, window_shift, snip_edges)
    
    # pack batch
    shape = strided_input.size()
    strided_input = strided_input.reshape(-1, shape[-1])

    if dither != 0.0:
        # Returns a random number strictly between 0 and 1
        x = torch.max(epsilon, torch.rand(strided_input.shape, device=device, dtype=dtype))
        rand_gauss = torch.sqrt(-2 * x.log()) * torch.cos(2 * math.pi * x)
        strided_input = strided_input + rand_gauss * dither

    if remove_dc_offset:
        # Subtract each row/frame by its mean
        row_means = torch.mean(strided_input, dim=-1).unsqueeze(-1)  # size (m, 1)
        strided_input = strided_input - row_means

    if raw_energy:
        # Compute the log energy of each row/frame before applying preemphasis and
        # window function
        signal_log_energy = _get_log_energy(strided_input, epsilon, energy_floor)  # size (m)

    if preemphasis_coefficient != 0.0:
        # strided_input[i,j] -= preemphasis_coefficient * strided_input[i, max(0, j-1)] for all i,j
        offset_strided_input = torch.nn.functional.pad(strided_input.unsqueeze(0), (1, 0), mode="replicate").squeeze(
            0
        )  # size (m, window_size + 1)
        strided_input = strided_input - preemphasis_coefficient * offset_strided_input[:, :-1]

    # Apply window_function to each row/frame
    window_function = _feature_window_function(window_type, window_size, blackman_coeff, device, dtype).unsqueeze(
        0
    )  # size (1, window_size)
    strided_input = strided_input * window_function  # size (m, window_size)

    # Pad columns with zero until we reach size (m, padded_window_size)
    if padded_window_size != window_size:
        padding_right = padded_window_size - window_size
        strided_input = torch.nn.functional.pad(
            strided_input.unsqueeze(0), (0, padding_right), mode="constant", value=0
        ).squeeze(0)

    # Compute energy after window function (not the raw one)
    if not raw_energy:
        signal_log_energy = _get_log_energy(strided_input, epsilon, energy_floor)  # size (m)
    
    # unpack batch
    strided_input = strided_input.reshape(shape[:-1] + strided_input.shape[-1:])
    
    return strided_input, signal_log_energy



class Feature_Trans():
    def __init__(
        self,
        blackman_coeff: float = 0.42,
        channel: int = -1,
        dither: float = 1.0,
        energy_floor: float = 0.0,

        window_size: float = 400,
        window_shift: float = 160,
        
        high_freq: float = 0.0,
        htk_compat: bool = False,
        low_freq: float = 20.0,
        min_duration: float = 0.0,
        num_mel_bins: int = 80,
        preemphasis_coefficient: float = 0.97,
        raw_energy: bool = True,
        remove_dc_offset: bool = True,
        round_to_power_of_two: bool = True,
        sample_frequency: float = 16000.0,
        snip_edges: bool = True,
        subtract_mean: bool = False,
        use_energy: bool = False,
        use_log_fbank: bool = True,
        use_power: bool = True,
        vtln_high: float = -500.0,
        vtln_low: float = 100.0,
        vtln_warp: float = 1.0,
        window_type: str = POVEY,
        
        need_spectrum: bool = False
        
        ):
        """
        A class for handling various audio transformations and feature extraction steps.

        Methods:
            cal_stft(waveform: Tensor, ilens: Tensor) -> Tuple[Tensor, Tensor]:
                Calculates the Short-Time Fourier Transform (STFT) of the input waveform.

            cal_fbank(waveform: Tensor, ilens: Tensor) -> Tuple[Tensor, Tensor]:
                Calculates the Mel filter bank features of the input waveform.

            stft_to_fbank(input: Tensor, ilens: Tensor, signal_log_energy: Tensor = None) -> Tuple[Tensor, Tensor]:
                Converts STFT to Mel filter bank features.

            spectrum_to_fbank(input: Tensor, ilens: Tensor, signal_log_energy: Tensor = None) -> Tuple[Tensor, Tensor]:
                Converts spectrum to Mel filter bank features.
        """
        
        self.channel = channel
        
        self.window_size = window_size
        self.window_shift = window_shift
        self.frame_length = window_size // (sample_frequency * MILLISECONDS_TO_SECONDS)
        self.frame_shift = window_shift // (sample_frequency * MILLISECONDS_TO_SECONDS)
        
        
        self.sample_frequency = sample_frequency
        self.round_to_power_of_two = round_to_power_of_two
        self.preemphasis_coefficient = preemphasis_coefficient
        self.min_duration = min_duration
        self.window_type = window_type
        self.blackman_coeff = blackman_coeff
        self.snip_edges = snip_edges
        self.raw_energy = raw_energy
        self.energy_floor = energy_floor
        self.dither = dither
        self.remove_dc_offset = remove_dc_offset
        self.preemphasis_coefficient = preemphasis_coefficient
        self.subtract_mean = subtract_mean
        self.use_power = use_power
        
        self.num_mel_bins = num_mel_bins
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.vtln_low = vtln_low
        self.vtln_high = vtln_high
        self.vtln_warp = vtln_warp
        self.use_log_fbank = use_log_fbank
        self.use_energy = use_energy
        self.htk_compat = htk_compat
        #n-fft
        self.padded_window_size = _next_power_of_2(window_size) if round_to_power_of_two else window_size
        
        self.need_spectrum = need_spectrum

        pass
    
    def cal_stft(self,waveform, ilens):
        """
        Calculates the Short-Time Fourier Transform (STFT) of the input waveform.

        Args:
            waveform (Tensor): Input waveform tensor.
            ilens (Tensor): Input lengths tensor.

        Returns:
            Tuple[Tensor, Tensor]: STFT result and its lengths.
        """
        #device, dtype = waveform.device, waveform.dtype
        #epsilon = _get_epsilon(device, dtype)

        if len(waveform) < self.min_duration * self.sample_frequency:
            # signal is too short
            return torch.empty(0)

        strided_input, signal_log_energy = _get_window(
            waveform,
            self.padded_window_size,
            self.window_size,
            self.window_shift,
            
            self.window_type,
            self.blackman_coeff,
            self.snip_edges,
            self.raw_energy,
            self.energy_floor,
            self.dither,
            self.remove_dc_offset,
            self.preemphasis_coefficient,
        )
        # input: (B,C,T) output: (B,C,T,F,2)
        # size (m, padded_window_size // 2 + 1, 2)
        fft = torch.fft.rfft(strided_input)
        if fft.is_complex():
            fft = torch.stack([fft.real, fft.imag], dim=-1)

        if self.snip_edges:
            if ilens.min() < self.window_size:
                return torch.empty(waveform.shape[:-1] + (0, self.window_size), dtype=waveform.dtype, device=waveform.device)
            else:
                olens = 1 + (ilens - self.window_size) // self.window_shift
        else:
            olens = (ilens + (self.window_shift // 2)) // self.window_shift
        
        # # Convert the FFT into a power spectrum
        if self.need_spectrum:
            device, dtype = waveform.device, waveform.dtype
            epsilon = _get_epsilon(device, dtype)
            power_spectrum = torch.max(fft.abs().pow(2.0), epsilon).log()  # size (m, padded_window_size // 2 + 1)
            power_spectrum[:, 0] = signal_log_energy

            power_spectrum = _subtract_column_mean(power_spectrum, self.subtract_mean)
            
            return power_spectrum, olens
        
        return fft, olens#, power_spectrum
    
    def cal_fbank(self,waveform,ilens):
        """
        Calculates the Mel filter bank features of the input waveform.

        Args:
            waveform (Tensor): Input waveform tensor.
            ilens (Tensor): Input lengths tensor.

        Returns:
            Tuple[Tensor, Tensor]: Mel filter bank features and their lengths.
        """
        device, dtype = waveform.device, waveform.dtype

        strided_input, signal_log_energy = _get_window(
            waveform,
            self.padded_window_size,
            self.window_size,
            self.window_shift,
            
            self.window_type,
            self.blackman_coeff,
            self.snip_edges,
            self.raw_energy,
            self.energy_floor,
            self.dither,
            self.remove_dc_offset,
            self.preemphasis_coefficient,
        )
        # input: (B,C,T) output: (B,C,T,M)
        # size (..., m, padded_window_size // 2 + 1)
        spectrum = torch.fft.rfft(strided_input).abs()
        if self.use_power:
            spectrum = spectrum.pow(2.0)
        
        # pack batch
        shape = spectrum.size()
        spectrum = spectrum.reshape(-1, shape[-1])

        # size (num_mel_bins, padded_window_size // 2)
        mel_energies, _ = get_mel_banks(
            self.num_mel_bins, self.padded_window_size, self.sample_frequency, self.low_freq, self.high_freq, self.vtln_low, self.vtln_high, self.vtln_warp
        )
        mel_energies = mel_energies.to(device=device, dtype=dtype)

        # pad right column with zeros and add dimension, size (num_mel_bins, padded_window_size // 2 + 1)
        mel_energies = torch.nn.functional.pad(mel_energies, (0, 1), mode="constant", value=0)

        # sum with mel fiterbanks over the power spectrum, size (m, num_mel_bins)
        mel_energies = torch.mm(spectrum, mel_energies.T)
        if self.use_log_fbank:
            # avoid log of zero (which should be prevented anyway by dithering)
            mel_energies = torch.max(mel_energies, _get_epsilon(device, dtype)).log()

        # if use_energy then add it as the last column for htk_compat == true else first column
        if self.use_energy:
            signal_log_energy = signal_log_energy.unsqueeze(1)  # size (m, 1)
            # returns size (m, num_mel_bins + 1)
            if self.htk_compat:
                mel_energies = torch.cat((mel_energies, signal_log_energy), dim=1)
            else:
                mel_energies = torch.cat((signal_log_energy, mel_energies), dim=1)

        mel_energies = _subtract_column_mean(mel_energies, self.subtract_mean)
        
        # unpack batch
        mel_energies = mel_energies.reshape(shape[:-1] + mel_energies.shape[-1:])
        
        if self.snip_edges:
            if ilens < self.window_size:
                return torch.empty(waveform.shape[:-1] + (0, self.window_size), dtype=waveform.dtype, device=waveform.device)
            else:
                olens = 1 + (ilens - self.window_size) // self.window_shift
        else:
            olens = (ilens + (self.window_shift // 2)) // self.window_shift
        
        return mel_energies, olens
    
    def stft_to_fbank(
                self, input, ilens,signal_log_energy: torch.Tensor = None
                ):
        """
        Converts STFT to Mel filter bank features.

        Args:
            input (Tensor): STFT input tensor.
            ilens (Tensor): Input lengths tensor.
            signal_log_energy (Tensor, optional): Signal log energy tensor.

        Returns:
            Tuple[Tensor, Tensor]: Mel filter bank features and their lengths.
        """
        # input:(B,T,F,2)
        device, dtype = input.device, input.dtype
        if not input.is_complex():
            spectrum = torch.complex(input[...,0], input[...,1])
        
        spectrum = spectrum.abs()
        
        if self.use_power:
            spectrum = spectrum.pow(2.0)
            
        # pack batch
        shape = spectrum.size()
        spectrum = spectrum.reshape(-1, shape[-1])

        # size (num_mel_bins, padded_window_size // 2)
        mel_energies, _ = get_mel_banks(
            self.num_mel_bins, self.padded_window_size, self.sample_frequency, self.low_freq, self.high_freq, self.vtln_low, self.vtln_high, self.vtln_warp
        )
        mel_energies = mel_energies.to(device=device, dtype=dtype)

        # pad right column with zeros and add dimension, size (num_mel_bins, padded_window_size // 2 + 1)
        mel_energies = torch.nn.functional.pad(mel_energies, (0, 1), mode="constant", value=0)

        # sum with mel fiterbanks over the power spectrum, size (m, num_mel_bins)
        mel_energies = torch.mm(spectrum, mel_energies.T)
        if self.use_log_fbank:
            # avoid log of zero (which should be prevented anyway by dithering)
            mel_energies = torch.max(mel_energies, _get_epsilon(device, dtype)).log()

        # if use_energy then add it as the last column for htk_compat == true else first column
        if self.use_energy:
            assert signal_log_energy is not None
            signal_log_energy = signal_log_energy.unsqueeze(1)  # size (m, 1)
            # returns size (m, num_mel_bins + 1)
            if self.htk_compat:
                mel_energies = torch.cat((mel_energies, signal_log_energy), dim=1)
            else:
                mel_energies = torch.cat((signal_log_energy, mel_energies), dim=1)

        mel_energies = _subtract_column_mean(mel_energies, self.subtract_mean)
        
        # unpack batch
        mel_energies = mel_energies.reshape(shape[:-1] + mel_energies.shape[-1:])
        
        return mel_energies, ilens
    
    def spectrum_to_fbank(
                self, input, ilens,signal_log_energy: torch.Tensor = None
                ):
        """
        Converts spectrum to Mel filter bank features.

        Args:
            input (Tensor): Spectrum input tensor.
            ilens (Tensor): Input lengths tensor.
            signal_log_energy (Tensor, optional): Signal log energy tensor.

        Returns:
            Tuple[Tensor, Tensor]: Mel filter bank features and their lengths.
        """
        # input:(B,T,F,2)
        device, dtype = input.device, input.dtype
        
        spectrum = input
            
        # pack batch
        shape = spectrum.size()
        spectrum = spectrum.reshape(-1, shape[-1])

        # size (num_mel_bins, padded_window_size // 2)
        mel_energies, _ = get_mel_banks(
            self.num_mel_bins, self.padded_window_size, self.sample_frequency, self.low_freq, self.high_freq, self.vtln_low, self.vtln_high, self.vtln_warp
        )
        mel_energies = mel_energies.to(device=device, dtype=dtype)

        # pad right column with zeros and add dimension, size (num_mel_bins, padded_window_size // 2 + 1)
        mel_energies = torch.nn.functional.pad(mel_energies, (0, 1), mode="constant", value=0)

        # sum with mel fiterbanks over the power spectrum, size (m, num_mel_bins)
        mel_energies = torch.mm(spectrum, mel_energies.T)
        if self.use_log_fbank:
            # avoid log of zero (which should be prevented anyway by dithering)
            mel_energies = torch.max(mel_energies, _get_epsilon(device, dtype)).log()

        # if use_energy then add it as the last column for htk_compat == true else first column
        if self.use_energy:
            assert signal_log_energy is not None
            signal_log_energy = signal_log_energy.unsqueeze(1)  # size (m, 1)
            # returns size (m, num_mel_bins + 1)
            if self.htk_compat:
                mel_energies = torch.cat((mel_energies, signal_log_energy), dim=1)
            else:
                mel_energies = torch.cat((signal_log_energy, mel_energies), dim=1)

        mel_energies = _subtract_column_mean(mel_energies, self.subtract_mean)
        
        # unpack batch
        mel_energies = mel_energies.reshape(shape[:-1] + mel_energies.shape[-1:])
        
        return mel_energies, ilens


# 导入必要的库和模块
#import torch
if __name__ == "__main__":
    from torchaudio.transforms import Resample
    from torchaudio.utils import download_asset

    torch.random.manual_seed(0)

    # SAMPLE_SPEECH = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
    SAMPLE_SPEECH = r"C:\Users\kxz\Desktop\SACC\back.wav"

    # 创建 Feature_Trans 实例
    feature_transform = Feature_Trans(num_mel_bins=80)

    # 读取音频文件（示例中使用 torchaudio）
    waveform, sample_rate = torchaudio.load(SAMPLE_SPEECH,normalize=False)

    # 如果音频采样率不是 16kHz，进行重采样
    if sample_rate != 16000:
        resample = Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample(waveform)

    
    waveform = waveform.unsqueeze(0)
    # 获取输入音频的长度
    input_length = waveform.size(-1)

    # 计算梅尔频谱特征
    mel_energies1, output_length = feature_transform.cal_stft(waveform.to(dtype=torch.float32), input_length)
    mel_energies1,_ = feature_transform.stft_to_fbank(mel_energies1, output_length)

    mel_energies2, output_length = feature_transform.cal_fbank(waveform.to(dtype=torch.float32), input_length)

    # 打印输出结果的形状
    print("Input Shape:", waveform.shape)
    print("Mel Energies1 Shape:", mel_energies1.shape)
    print(mel_energies1[:,1,:])
    print("Mel Energies2 Shape:", mel_energies2.shape)
    print(mel_energies2[:,1,:])
    print("Output Length:", output_length)
    
    

