# Copyright 2020 Tsinghua SPMI Lab / Tasi
# Apache 2.0.
# Author: Xiangzhu Kong (kongxiangzhu99@gmail.com)
#
# Description:
#   This script simulates room acoustics and generates reverberant speech signals with noise for speech processing experiments.
#   The key functions include generating room configurations, placing microphones and sources, and mixing speech and noise at specified SNR levels.
#   The script saves the generated data and metadata, and supports resuming from saved progress.


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

from tqdm import tqdm

seed = 0
# 设置伪随机数生成器的种子
random.seed(seed)
np.random.seed(seed)


c = 340
fs=16000 # Sampling frequency [Hz]
num_room = 999999
utt_per_room = 5
room_x = 8
room_y = 8
room_z = 3

save_interval = 100  # 每处理100个语音文件后保存一次进度

# 定义可能的麦克风类型
mic_types = ["circular", "linear","square"] # "non_uniform_linear"
channels = 8
phase = 0
fs = 16000

# 假设的最大噪声源数量
total_noise_num = 2
speech_num = 1

# 定义SNR范围[-5, 10 )
snr_range = (-5, 10)

# 定义非线性阵列麦克风间的相对距离（以米为单位）
#relative_distances_m = np.array([0.15, 0.1, 0.05, 0.2, 0.05, 0.1, 0.15])
offsets_m = [-0.4, -0.25, -0.15, -0.1, 0.1, 0.15, 0.25, 0.41]  # 偏移量




def check_conditions(speech_source, noise_sources, mic_middle_point):
    """
    Check if the conditions between the speech source, noise sources, and microphone array are met:
        1. The distance between the sources and the microphone array is within (0.5, 5.0) meters.
        2. The angle between the speech source and the noise sources is greater than 20°.

    Args:
        speech_source (list): Coordinates of the speech source [x, y, z].
        noise_sources (list): List of coordinates for the noise sources [x, y, z].
        mic_middle_point (list): Coordinates of the microphone array center [x, y, z].

    Returns:
        bool: True if conditions are met, False otherwise.
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
        # start_idx = size_diff // 2  # 两边进行剪裁
        start_idx = 0 
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
    Calculate the root mean square (RMS) of the waveform data.

    Args:
        data (numpy.ndarray): Input waveform data.

    Returns:
        float: RMS value.
    """
    energy = data ** 2
    max_e = np.max(energy)
    low_thres = max_e*(10**(-50/10)) # to filter lower than 50dB 
    rms = np.mean(energy[energy>=low_thres])
    #rms = np.mean(energy)
    return rms

def snr_mix(clean, noise, snr):
    """
    Mix clean and noise signals according to the specified SNR.

    Args:
        clean (numpy.ndarray): Clean speech signal.
        noise (numpy.ndarray): Noise signal.
        snr (float): Signal-to-noise ratio (SNR).

    Returns:
        numpy.ndarray: Scaled noise signal to achieve the specified SNR.
    """
    clean_rms = rms(clean)
    clean_rms = np.maximum(clean_rms, eps)
    noise_rms = rms(noise)
    noise_rms = np.maximum(noise_rms, eps)
    k = math.sqrt(clean_rms / (10**(snr/10) * noise_rms))
    new_noise = noise * k
    return new_noise

def get_one_spk_noise(clean, noise, snr, scale):
    """
    Mix clean and noise signals according to the specified SNR and scale the result.

    Args:
        clean (numpy.ndarray): Clean speech signal, shape (L, C) where L is segment length.
        noise (numpy.ndarray): Noise signal, shape (L', C).
        snr (float): Signal-to-noise ratio (SNR).
        scale (float): Scaling factor for the resulting signal.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Scaled noisy and clean signals.
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

# 检查麦克风位置是否在房间内
def is_mic_inside_room(mic_array, room_x, room_y):
    """
    Check if the microphones are inside the room.

    Args:
        mic_array (numpy.ndarray): Microphone array positions.
        room_x (float): Room width.
        room_y (float): Room length.

    Returns:
        bool: True if all microphones are inside the room, False otherwise.
    """
    for mic in mic_array.T:  # 遍历每个麦克风
        x, y = mic[:2]  # 获取麦克风的 x 和 y 坐标
        if x < 0 or x > room_x or y < 0 or y > room_y:
            return False  # 如果麦克风不在房间内，返回 False
    return True  # 所有麦克风都在房间内，返回 True

def get_speech_reverb(room, speech_source,speech_path):
    """
    Simulate reverberant speech by placing the speech source in the room.

    Args:
        room (pyroomacoustics.Room): Room object.
        speech_source (list): Coordinates of the speech source [x, y, z].
        speech_path (str): Path to the speech file.

    Returns:
        Tuple[numpy.ndarray, int, bool]: Reverberant speech signals, original length, and a flag indicating if the room was skipped.
    """
    fs, audio = wavfile.read(speech_path)
    audio_len = len(audio)
    # 更新声源
    room.sources = []
    try:
        speech_source_array = np.array([speech_source], dtype=np.float32)
        room.add_source(speech_source_array.T, signal=audio)
        room.simulate()
        mic_signals = room.mic_array.signals
    
        return mic_signals, audio_len, False
    except:
        bbox = room.get_bbox()
        room_size = bbox[:, 1] - bbox[:, 0]
        mic_positions = room.mic_array.R
        print("房间坐标：", bbox)
        print("房间尺寸(x, y, z): ", room_size)
        print("源坐标",speech_source)
        print("麦克风位置：",mic_positions)
        #raise ValueError("The source must be added inside the room.")
        return False, False, True
        
    
 
def get_noise_reverb(room, noise_source,noise_path,speech_len):
    """
    Simulate reverberant noise by placing the noise source in the room.

    Args:
        room (pyroomacoustics.Room): Room object.
        noise_source (list): Coordinates of the noise source [x, y, z].
        noise_path (str): Path to the noise file.
        speech_len (int): Length of the speech signal.

    Returns:
        Tuple[numpy.ndarray, int, int, bool]: Reverberant noise signals, start position, end position, and a flag indicating if the room was skipped.
    """
    fs, audio = wavfile.read(noise_path)
    audio_len = len(audio)
    if audio_len > speech_len:
        # 随机选择audio的起始位置
        start_position = np.random.randint(0, audio_len - speech_len + 1)
        end_position = start_position + speech_len
        # 选择长度为speech_len的片段
        audio = audio[start_position:end_position]
    else :
        # 补零以达到speech_len
        zero_padding = np.zeros(speech_len - audio_len)
        audio = np.concatenate([audio, zero_padding])
        start_position = 0
        end_position = start_position + audio_len
    # 更新声源
    room.sources = []
    try:
        noise_source_array = np.array([noise_source], dtype=np.float32)
        room.add_source(noise_source_array.T, signal=audio)
        room.simulate()
        mic_signals = room.mic_array.signals
        
        return mic_signals, start_position, end_position, False
    except:
        bbox = room.get_bbox()
        room_size = bbox[:, 1] - bbox[:, 0]
        mic_positions = room.mic_array.R
        print("房间坐标：", bbox)
        print("房间尺寸(x, y, z): ", room_size)
        print("noise源坐标",noise_source)
        print("麦克风位置：",mic_positions)
        #raise ValueError("The source must be added inside the room.")
        return True, True, True, True
        
    

# 定义非均匀线性麦克风阵列的函数
def non_uniform_linear_array(center, offsets_m):
    """
    根据提供的偏移量在中心点附近创建非均匀线性麦克风阵列。

    Parameters:
    center: list or array, 阵列中心点的坐标[x, y, z]。
    offsets_m: list or array, 每个麦克风点相对于中心点的水平偏移量。

    Returns:
    numpy.ndarray: 麦克风阵列的坐标数组，形状为(3, M)。
    """
    # 计算每个麦克风的x坐标
    mic_positions_x_m = center[0] + np.array(offsets_m)
    # 所有麦克风的y和z坐标与中心点相同
    mic_positions_y_m = np.full_like(mic_positions_x_m, center[1])
    mic_positions_z_m = np.full_like(mic_positions_x_m, center[2])

    return np.vstack((mic_positions_x_m, mic_positions_y_m, mic_positions_z_m))

def is_inside_room(room_x, room_y, source):
    """
    Check if the source is inside the room.

    Args:
        room_x (float): Room width.
        room_y (float): Room length.
        source (list): Coordinates of the source [x, y, z].

    Returns:
        bool: True if the source is inside the room, False otherwise.
    """
    x, y, _ = source
    return 0 < x < room_x and 0 < y < room_y


def save_progress(progress_file, wav_scp_entries, all_samples_info, progress):
    """
    Save the current progress to a file.

    Args:
        progress_file (str): Path to the progress file.
        wav_scp_entries (list): List of wav.scp entries.
        all_samples_info (list): List of all sample information.
        progress (dict): Progress information to be saved.
    """
    with open(progress_file, 'w') as f:
        json.dump({
            'wav_scp_entries': wav_scp_entries,
            'all_samples_info': all_samples_info,
            'progress': progress
        }, f)

def load_progress(progress_file):
    """
    Load progress from a file.

    Args:
        progress_file (str): Path to the progress file.

    Returns:
        tuple: List of wav.scp entries, list of all sample information, and progress information.
    """
    try:
        with open(progress_file, 'r') as f:
            data = json.load(f)
            return data['wav_scp_entries'], data['all_samples_info'], data['progress']
    except FileNotFoundError:
        return [], [], None


def main(speech_dir, noise_dirs, output_dir): 

    # Output directory for generated WAV files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Paths to speech and noise directories
    #speech_path = sorted([os.path.join(root, file) for root, dirs, files in os.walk(speech_dir) for file in files if file.endswith(".wav")])
    #noise_paths = sorted([os.path.join(root, file) for root, dirs, files in os.walk(noise_dir) for file in files if file.endswith(".wav")])
    
    speech_path = []
    wav_scp_path = os.path.join(speech_dir, "wav.scp")
    with open(wav_scp_path, 'r') as file:
        for line in file:
            # 通常每行包含一个标识符和一个路径
            path = line.strip().split()[-1]
            speech_path.append(path)
    # 处理噪声文件，遍历所有噪声目录
    noise_paths = []
    for noise_dir in noise_dirs:
        noise_paths.extend(sorted([
            os.path.join(root, file) 
            for root, dirs, files in os.walk(noise_dir) 
            for file in files if file.endswith(".wav")
        ]))
    
    # 总迭代次数为所有语音文件的数量
    total_iterations = len(speech_path)
    
    # JSON file name
    json_file_name = os.path.join(output_dir, "all_samples_info.json")
    #pbar = tqdm(total=total_iterations, desc="Processing", unit="file")
    
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
    
    progress_file = os.path.join(output_dir, 'progress.json')  # 进度文件的路径

    # 尝试加载现有进度
    wav_scp_entries, all_samples_info, progress = load_progress(progress_file)
    if progress is not None:
        start_room_idx = progress['room_idx']
        speech_idx = progress['speech_idx']
    else:
        start_room_idx = 0
        speech_idx = 0
    
    # 创建进度条对象，设置初始值为已处理的语音文件数
    pbar = tqdm(total=total_iterations, initial=speech_idx, desc="Processing", unit="file")
    
    for i in range(start_room_idx, num_room):
        x = np.random.uniform(3, room_x)
        y = np.random.uniform(3, room_y)
        z = room_z
        corners = np.array([[0,0],[0,y], [x,y], [x,0]]).T
        room = pra.Room.from_corners(corners, fs=fs, max_order=3, materials=pra.Material(0.2, 0.15), ray_tracing=True, air_absorption=True)
        room.extrude(z, materials=pra.Material(0.2, 0.15)) #天花板也要设置反射系数，否则会有很长的拖尾
        room.set_ray_tracing(receiver_radius=0.5, n_rays=10000, energy_thres=1e-5)
        for j in range(utt_per_room):
            skip_room = False  # 设置标志变量
            # 从范围内随机选择一个SNR值
            snr = random.uniform(snr_range[0], snr_range[1])
            speech_source = [np.random.uniform(0.1, x-0.1),np.random.uniform(0.1, y-0.1),np.random.uniform(1.2, 1.9)]
            if not is_inside_room(x, y, speech_source):
                continue  # 如果声源不在房间内，则跳过当前循环
            noise_num = random.randint(1, total_noise_num)
            noise_sources = [None] * noise_num
            for k in range(noise_num):
                noise_sources[k] = [np.random.uniform(0.1, x-0.1),np.random.uniform(0.1, y-0.1),np.random.uniform(1.2, 1.9)]
                if not is_inside_room(x, y, noise_sources[k]):
                    skip_room = True  # 如果噪声源不在房间内，设置标志为 True
                    break  # 跳出 k 循环
            if skip_room:
                continue  # 如果标志为 True，跳出 j 循环，进入下一个 j 循环
            
            # 随机选择一个麦克风类型
            mic_type = random.choice(mic_types)
            
            # 计算麦克风阵列的坐标
            if mic_type == "circular":
                mic_distance = 0.05 #麦克风阵列的半径或者间距
                mic_middle_point = [
                    np.random.uniform(mic_distance * 2 + x / 4, x - mic_distance * 2 - x / 4),
                    np.random.uniform(mic_distance * 2 + y / 4, y - mic_distance * 2 - y / 4),
                    np.random.uniform(1.0, 1.5)
                ]
                # 这里应该定义圆形阵列的具体计算方式
                mic_array = pra.circular_2D_array(mic_middle_point[:2], channels, phase, mic_distance)
                mic_array = np.pad(mic_array, ((0, 1), (0, 0)), 'constant', constant_values=mic_middle_point[2])

            elif mic_type == "linear":
                mic_distance = 0.011 #麦克风阵列的半径或者间距
                mic_middle_point = [
                    np.random.uniform(mic_distance * 4 + x / 4, x - mic_distance * 4 - x / 4),
                    np.random.uniform(mic_distance + y / 4, y - mic_distance - y / 4),
                    np.random.uniform(1.0, 1.5)
                ]
                # 这里应该定义线性阵列的具体计算方式
                mic_array = pra.linear_2D_array(mic_middle_point[:2], channels, phase, mic_distance)
                mic_array = np.pad(mic_array, ((0, 1), (0, 0)), 'constant', constant_values=mic_middle_point[2])

            elif mic_type == "non_uniform_linear":
                # 生成随机中心点，确保麦克风阵列不会超出房间边界
                mic_middle_x = np.random.uniform(0.4, x - 0.4)
                mic_middle_y = np.random.uniform(0.4, y - 0.4)
                mic_middle_z = np.random.uniform(1.0, 1.5)  # 假设麦克风阵列高度在1.0到1.5米之间

                mic_middle_point = [mic_middle_x, mic_middle_y, mic_middle_z]

                # 直接使用中心点和相对距离计算非均匀线性阵列的坐标
                mic_array = non_uniform_linear_array(mic_middle_point, offsets_m)

            elif mic_type == "square":
                mic_distance = 0.011 #麦克风阵列的半径或者间距
                mic_middle_point = [
                    np.random.uniform(mic_distance * 4 + x / 4, x - mic_distance * 4 - x / 4),
                    np.random.uniform(mic_distance + y / 4, y - mic_distance - y / 4),
                    np.random.uniform(1.0, 1.5)
                ]
                # 这里应该定义矩形阵列的具体计算方式
                mic_array = pra.square_2D_array(mic_middle_point[:2], 2, 4, phase, mic_distance)
                mic_array = np.pad(mic_array, ((0, 1), (0, 0)), 'constant', constant_values=mic_middle_point[2])
            
            if not is_mic_inside_room(mic_array, room_x, room_y):
                continue  # 如果有麦克风不在房间内，跳过当前循环，进入下一个循环
            
            room.mic_array = None
            room.add_microphone_array(pra.MicrophoneArray(mic_array, fs))
            
            if check_conditions(speech_source, noise_sources, mic_middle_point):
                selected_noise_names = []  # 用于存储所选噪声文件的名称（去掉前缀和后缀）
                noise_interval = []
                selected_speech_names = [] # 储存语音文件的名称
                rev_speech, ori_length,skip_room = get_speech_reverb(room, speech_source, speech_path[speech_idx])
                if skip_room:
                    continue  # 如果标志为 True，跳出 j 循环，进入下一个 j 循环
                
                # 提取文件名（去掉前缀和后缀）
                speech_name = os.path.splitext(os.path.basename(speech_path[speech_idx]))[0]
                selected_speech_names.append(speech_name)  # 将语音文件名称添加到列表中

                rev_speech = adjust_matrix_dimension(rev_speech, ori_length, 1)
                rev_noise = []
                for m in range(noise_num):
                    selected_noise_path = random.choice(noise_paths)
                    # 提取文件名（去掉前缀和后缀）
                    noise_name = os.path.splitext(os.path.basename(selected_noise_path))[0]

                    rev_noise_source, noise_start, noise_end,skip_room = get_noise_reverb(room, noise_sources[m], selected_noise_path,ori_length)
                    rev_noise_source = adjust_matrix_dimension(rev_noise_source, ori_length, 1)
                    rev_noise.append(rev_noise_source)
                    
                    selected_noise_names.append(noise_name+'_'+f"{(noise_start/16000):.2f}:{(noise_end/16000):.2f}" )  # 将选择的噪声文件名称添加到列表中
                    noise_interval.append(f"{noise_start}:{noise_end}")
                    if skip_room:
                        break  # 如果标志为 True，跳出 m 循环
                
                if skip_room:
                    continue  # 如果标志为 True，跳出 j 循环，进入下一个 j 循环
                    
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
                    'idx': speech_idx+1,
                    'simu_file': f"{speech_idx+1}_{all_speech_names}_{all_noise_names}_{x:.2f}_{y:.2f}_{z:.2f}_{snr:.3f}_{mic_type}.wav",
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
                audio_file_path = os.path.join(output_dir, f"{speech_idx+1}_{all_speech_names}_{all_noise_names}_{x:.2f}_{y:.2f}_{z:.2f}_{snr:.3f}_{mic_type}.wav")
                sf.write(audio_file_path, int_signal.T, fs, subtype='PCM_16')
                
                # 将 wav.scp 的条目添加到列表中
                wav_scp_entries.append(f"{all_speech_names}\t{audio_file_path}")
                
                speech_idx += 1
                
                if speech_idx  >= len(speech_path):
                    # 将更新后的信息写入 JSON 文件
                    with open(json_file_name, 'w') as json_file:
                        json.dump(all_samples_info, json_file, indent=4)
                    # 将 wav.scp 的内容写入文件
                    with open(wav_scp_file_path, 'w') as wav_scp_file:
                        wav_scp_file.write('\n'.join(wav_scp_entries))  
                    
                    # 完成处理后删除进度文件
                    if os.path.exists(progress_file):
                        os.remove(progress_file)  
                    
                    #print(f"Progress: 100%")
                    #print("Completed!")
                    
                    pbar.close()
                    return 
                # 手动显示进度
                #progress = (speech_idx) / len(speech_path) * 100
                #print(f"Progress: {progress:.2f}%")
                # 更新进度、wav_scp_entries 和 all_samples_info 并保存
                if speech_idx % save_interval == 0:
                    progress = {'room_idx': i, 'speech_idx': speech_idx}
                    save_progress(progress_file, wav_scp_entries, all_samples_info, progress)
                pbar.update(1)

                               
            else :
                #print("The generated room does not meet the criteria! ")
                continue
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate room and save WAV files")
    # 读取语音目录下的wac.scp文件，其格式为uid + path
    parser.add_argument("--speech_dir", type=str, required=True, help="Path to the speech directory")
    # 读取所有噪声目录下的所有wav文件
    parser.add_argument("--noise_dir", type=str, nargs='+', required=True, help="Paths to the noise directories")
    # 输出文件夹
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")

    args = parser.parse_args()

    main(args.speech_dir, args.noise_dir, args.output_dir)



      
        
