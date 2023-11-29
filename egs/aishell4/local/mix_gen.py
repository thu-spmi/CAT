# Copyright 2022 Tsinghua University
# Apache 2.0.
# Author: Xiangzhu Kong

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
#import IPython
import pyroomacoustics as pra
#from shapely.geometry import Point, Polygon
import random
import os
import math
import soundfile as sf
# import torch
# import torchaudio
import json
eps = np.finfo(np.float32).eps

seed = 0
# 设置伪随机数生成器的种子
random.seed(seed)
np.random.seed(seed)


c = 340
fs=16000 # Sampling frequency [Hz]
num_room = 10000
utt_per_room = 5
room_x = 8
room_y = 8
room_z = 3

mic_type = "circular" # circular or linear
channels = 8
phase = 0
fs = 16000

noise_num = 2
speech_num = 1

# 定义SNR范围[-5, 10 )
snr_range = (-5, 10)

def check_conditions(speech_source, noise_sources, mic_middle_point):
    """
    检查语音源、噪声源们与麦克风阵列之间的条件是否满足:
        1、声源与麦克风阵列的距离位于(0.5, 5.0)之内
        2、语音源与噪声源们之间的角度大于20°

    参数：
    - speech: 语音源的位置，格式为 [x, y, z]
    - noise_sources: 噪声源们的位置列表，每个噪声源的位置格式为 [x, y, z]
    - mic_middle_point: 麦克风阵列中心点的位置，格式为 [x, y, z]

    返回：
        如果满足条件，返回 True, 否则返回 False。
    """

    # 计算语音源与麦克风阵列的距离和角度
    speech_distance = np.linalg.norm(np.array(speech_source) - np.array(mic_middle_point))
    speech_angle = np.arctan2(speech_source[1] - mic_middle_point[1], speech_source[0] - mic_middle_point[0])
    speech_angle = np.degrees(speech_angle)

    # 计算每个噪声源与麦克风阵列的距离和角度
    noise_distances = [np.linalg.norm(np.array(noise_source) - np.array(mic_middle_point)) for noise_source in noise_sources]
    noise_angles = [np.arctan2(noise_source[1] - mic_middle_point[1], noise_source[0] - mic_middle_point[0]) for noise_source in noise_sources]
    noise_angles = np.degrees(noise_angles)

    # 计算语音源与噪声源之间的夹角
    angle_speech_noise = np.arccos(np.dot(np.array(speech_source) - np.array(mic_middle_point), np.array(noise_sources[0]) - np.array(mic_middle_point)) /
                                   (np.linalg.norm(np.array(speech_source) - np.array(mic_middle_point)) * np.linalg.norm(np.array(noise_sources[0]) - np.array(mic_middle_point))))
    angle_speech_noise = np.degrees(angle_speech_noise)

    # 检查条件
    if 0.5 <= speech_distance <= 5 and all(0.5 <= d <= 5 for d in noise_distances) and angle_speech_noise > 20:
        return True
    else:
        return False

def adjust_matrix_dimension(matrix, target_size, target_dimension):
    """
    Adjust the length of the specified dimension of the matrix to the target size.
    If the length is greater than the target size, it will be cropped.
    If the length is less than the target size, it will be padded.

    Parameters:
    - matrix: array-like
        The input matrix. Will be converted to a NumPy array.
    - target_size: int
        The target size for the specified dimension.
    - target_dimension: int
        The dimension index to adjust.

    Returns:
    - numpy.ndarray
        The adjusted matrix.
    """
    # Convert the input matrix to a NumPy array
    matrix = np.array(matrix)

    # Ensure target_dimension is a valid index
    if target_dimension < 0 or target_dimension >= matrix.ndim:
        raise ValueError("Invalid target_dimension index")

    # Get the shape of the input matrix
    shape = list(matrix.shape)

    # Calculate the padding or cropping size
    size_diff = shape[target_dimension] - target_size

    if size_diff > 0:
        # If the size is greater, crop the matrix
        start_idx = size_diff // 2
        end_idx = start_idx + target_size
        slices = [slice(None)] * len(shape)
        slices[target_dimension] = slice(start_idx, end_idx)
        adjusted_matrix = matrix[tuple(slices)]
    elif size_diff < 0:
        # If the size is smaller, pad the matrix
        pad_before = abs(size_diff) // 2
        pad_after = abs(size_diff) - pad_before
        pad_width = [(0, 0)] * len(shape)
        pad_width[target_dimension] = (pad_before, pad_after)
        adjusted_matrix = np.pad(matrix, pad_width, mode='constant', constant_values=0)
    else:
        # If the size is already equal, return the original matrix
        adjusted_matrix = matrix

    return adjusted_matrix

def rms(data):
    """
    calc rms of wav
    """
    energy = data ** 2
    max_e = np.max(energy)
    low_thres = max_e*(10**(-50/10)) # to filter lower than 50dB 
    rms = np.mean(energy[energy>=low_thres])
    #rms = np.mean(energy)
    return rms

def snr_mix(clean, noise, snr):
    '''
    mix clean and noise according to snr
    '''
    clean_rms = rms(clean)
    clean_rms = np.maximum(clean_rms, eps)
    noise_rms = rms(noise)
    noise_rms = np.maximum(noise_rms, eps)
    k = math.sqrt(clean_rms / (10**(snr/10) * noise_rms))
    new_noise = noise * k
    return new_noise

def get_one_spk_noise(clean, noise, snr, scale):
    """
    mix clean and noise according to the snr and scale
    args:
        clean: numpy.array, L x C  L is always segment_length
        noise: numpy.array, L' x C
        snr: float
        scale: float
    """
    clean = clean.T
    noise = noise.T
    gen_noise = snr_mix(clean, noise, snr)
    noisy = clean + gen_noise

    max_amp = np.max(np.abs(noisy))
    max_amp = np.maximum(max_amp, eps)
    noisy_scale = 1. / max_amp * scale
    clean = clean * noisy_scale
    noisy = noisy * noisy_scale
    
    clean = clean.T
    noisy = noisy.T
    return noisy, clean #noisy_scale

def get_speech_reverb(room, speech_source,speech_path):
    fs, audio = wavfile.read(speech_path)
    audio_len = len(audio)
    # 更新声源
    room.sources = []
    room.add_source(speech_source, signal=audio)
    room.simulate()
    mic_signals = room.mic_array.signals
    
    return mic_signals, audio_len
 
def get_noise_reverb(room, noise_source,noise_path,speech_len):
    fs, audio = wavfile.read(noise_path)
    audio_len = len(audio)
    if audio_len > speech_len:
        # 随机选择audio的起始位置
        start_position = np.random.randint(0, audio_len - speech_len + 1)
        # 选择长度为speech_len的片段
        audio = audio[start_position:start_position + speech_len]
    elif audio_len < speech_len:
        # 补零以达到speech_len
        zero_padding = np.zeros(speech_len - audio_len)
        audio = np.concatenate([audio, zero_padding])
    # 更新声源
    room.sources = []
    room.add_source(noise_source, signal=audio)
    room.simulate()
    mic_signals = room.mic_array.signals
    
    return mic_signals, start_position, start_position + speech_len

def main(speech_dir, noise_dir, output_dir): 

    # Output directory for generated WAV files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Paths to speech and noise directories
    speech_path = [os.path.join(speech_dir, file) for file in sorted(os.listdir(speech_dir)) if file.endswith(".wav")]
    noise_paths = [os.path.join(noise_dir, file) for file in os.listdir(noise_dir) if file.endswith(".wav")]
    
    # JSON file name
    json_file_name = os.path.join(output_dir, "all_samples_info.json")
    
    # 读取现有的 JSON 文件内容，如果存在的话
    all_samples_info = []
    if os.path.exists(json_file_name):
        with open(json_file_name, 'r') as json_file:
            all_samples_info = json.load(json_file)
            
    
    # 创建一个列表，用于存储 wav.scp 的内容
    wav_scp_entries = []
    
    # 读取现有的 wav.scp 文件内容，如果存在的话
    wav_scp_file_path = os.path.join(output_dir, "wav.scp")
    if os.path.exists(wav_scp_file_path):
        with open(wav_scp_file_path, 'r') as wav_scp_file:
            wav_scp_entries = wav_scp_file.read().splitlines()
    
    # 开始生成
    speech_idx = 0
    
    for i in range(num_room):
        x = np.random.uniform(3, room_x)
        y = np.random.uniform(3, room_y)
        z = room_z
        corners = np.array([[0,0],[0,y], [x,y], [x,0]]).T
        room = pra.Room.from_corners(corners, fs=fs, max_order=3, materials=pra.Material(0.2, 0.15), ray_tracing=True, air_absorption=True)
        room.extrude(z, materials=pra.Material(0.2, 0.15)) #天花板也要设置反射系数，否则会有很长的拖尾
        room.set_ray_tracing(receiver_radius=0.5, n_rays=10000, energy_thres=1e-5)
        for j in range(utt_per_room):
            # 从范围内随机选择一个SNR值
            snr = random.uniform(snr_range[0], snr_range[1])
            speech_source = [np.random.uniform(0, x),np.random.uniform(0, y),np.random.uniform(1.2, 1.9)]
            noise_sources = [None] * noise_num
            for k in range(noise_num):
                noise_sources[k] = [np.random.uniform(0, x),np.random.uniform(0, y),np.random.uniform(1.2, 1.9)]
            
            mic_distance = 0.05 #麦克风阵列的半径或者间距
            
            if mic_type == "circular":
                mic_func = pra.circular_2D_array
                mic_middle_point = [np.random.uniform(mic_distance*2+x/4, x-mic_distance*2-x/4),
                                    np.random.uniform(mic_distance*2+y/4, y-mic_distance*2-y/4),
                                    np.random.uniform(1.0, 1.5)]
                
            elif mic_type == "linear":
                mic_func = pra.linear_2D_array
                mic_middle_point = [np.random.uniform(mic_distance*4+x/4, x-mic_distance*4-x/4),
                                    np.random.uniform(mic_distance+y/4, y-mic_distance-y/4),
                                    np.random.uniform(1.0, 1.5)]
            
            
            mic_array = mic_func(mic_middle_point[:2], channels, phase, mic_distance)
            mic_array = np.pad(mic_array, ((0, 1), (0, 0)), constant_values=mic_middle_point[2])
            
            room.mic_array = None
            room.add_microphone_array(pra.MicrophoneArray(mic_array, fs))
            
            if check_conditions(speech_source, noise_sources, mic_middle_point):
                selected_noise_names = []  # 用于存储所选噪声文件的名称（去掉前缀和后缀）
                noise_interval = []
                selected_speech_names = [] # 储存语音文件的名称
                rev_speech, ori_length = get_speech_reverb(room, speech_source, speech_path[speech_idx])
                # 提取文件名（去掉前缀和后缀）
                speech_name = os.path.splitext(os.path.basename(speech_path[speech_idx]))[0]
                selected_speech_names.append(speech_name)  # 将语音文件名称添加到列表中

                rev_speech = adjust_matrix_dimension(rev_speech, ori_length, 1)
                rev_noise = []
                for m in range(noise_num):
                    selected_noise_path = random.choice(noise_paths)
                    # 提取文件名（去掉前缀和后缀）
                    noise_name = os.path.splitext(os.path.basename(selected_noise_path))[0]

                    rev_noise_source, noise_start, noise_end = get_noise_reverb(room, noise_sources[m], selected_noise_path,ori_length)
                    rev_noise_source = adjust_matrix_dimension(rev_noise_source, ori_length, 1)
                    rev_noise.append(rev_noise_source)
                    
                    selected_noise_names.append(noise_name+'_'+f"{(noise_start/16000):.2f}:{(noise_end/16000):.2f}" )  # 将选择的噪声文件名称添加到列表中
                    noise_interval.append(f"{noise_start}:{noise_end}")
                    
                # 创建一个用于存储所有噪声信号总和的数组
                total_noise = np.zeros_like(rev_noise[0])

                # 按通道对 rev_noise 中的所有噪声信号进行相加
                for n in range(noise_num):
                    total_noise += rev_noise[n]
                
                adjust_noise = snr_mix(rev_speech, total_noise, snr)
                mix = rev_speech + adjust_noise
                
                #signal = torch.tensor(mix).float()
                
                ################## 存储波形 ########################
                # 使用下划线连接所有噪声文件名称
                all_noise_names = '_'.join(selected_noise_names)
                all_speech_names = '_'.join(selected_speech_names)
                
                # 将当前语音信息添加到列表中
                sample_info = {
                    'idx': speech_idx,
                    'simu_file': f"{speech_idx}_{all_speech_names}_{all_noise_names}_{x:.2f}_{y:.2f}_{z:.2f}_{snr:.3f}_{mic_type}.wav",
                    'all_speech_names': all_speech_names,
                    'noise_names': selected_noise_names,
                    'room_size': {'x': x, 'y': y, 'z': z},
                    'snr': snr,
                    'mic_type': mic_type,
                    'speech_source': speech_source,
                    'noise_sources': noise_sources,
                    'noise_interal': noise_interval,
                    'channels': channels
                }
                all_samples_info.append(sample_info)
                
                # # 归一化信号
                # normalized_signal = signal / torch.max(torch.abs(signal))
                # # 保存归一化后的信号
                # audio_file_path = os.path.join(output_dir, f"{speech_idx}_{all_speech_names}_{all_noise_names}_{x:.2f}_{y:.2f}_{z:.2f}_{snr:.3f}_{mic_type}.wav")
                # torchaudio.save(audio_file_path, normalized_signal, sample_rate=fs)
                
                # 归一化麦克风信号
                normalized_signal_np = mix / np.max(np.abs(mix))
                # 映射归一化后的信号到整数范围
                int_signal = np.round(normalized_signal_np * np.iinfo(np.int16).max).astype(np.int16)

                # 保存归一化后的音频数据
                audio_file_path = os.path.join(output_dir, f"{speech_idx}_{all_speech_names}_{all_noise_names}_{x:.2f}_{y:.2f}_{z:.2f}_{snr:.3f}_{mic_type}.wav")
                sf.write(audio_file_path, int_signal.T, fs, subtype='PCM_16')
                
                # 将 wav.scp 的条目添加到列表中
                wav_scp_entries.append(f"{all_speech_names} {audio_file_path}")
                
                speech_idx += 1
                
                if speech_idx  >= len(speech_path):
                    # 将更新后的信息写入 JSON 文件
                    with open(json_file_name, 'w') as json_file:
                        json.dump(all_samples_info, json_file, indent=4)
                    # 将 wav.scp 的内容写入文件
                    with open(wav_scp_file_path, 'w') as wav_scp_file:
                        wav_scp_file.write('\n'.join(wav_scp_entries))    
                    
                    print(f"Progress: 100%")
                    print("Completed!")
                    return 
                # 手动显示进度
                progress = (speech_idx) / len(speech_path) * 100
                print(f"Progress: {progress:.2f}%")
                
                
            else :
                #print("The generated room does not meet the criteria! ")
                continue
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate room and save WAV files")
    parser.add_argument("--speech_dir", type=str, required=True, help="Path to the speech directory")
    parser.add_argument("--noise_dir", type=str, required=True, help="Path to the noise directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")

    args = parser.parse_args()
    

    main(args.speech_dir, args.noise_dir, args.output_dir)



      
        
