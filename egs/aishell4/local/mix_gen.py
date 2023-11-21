import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
#import IPython
import pyroomacoustics as pra
#from shapely.geometry import Point, Polygon
import random
import math
import torch
import torchaudio
eps = np.finfo(np.float32).eps

c = 340
fs=16000 # Sampling frequency [Hz]
num_room = 1
utt_per_room = 1
room_x = 8
room_y = 8
room_z = 3

mic_type = "circular" # circular or linear
channels = 8
phase = 0
fs = 16000

noise_num = 1
speech_num = 1
nb_src = noise_num + speech_num

speech_path = "/mnt/workspace/kongxz/data/706/clean/F031-001.wav"
noise_paths = [
    "/mnt/workspace/kongxz/data/706/clean/F031-002.wav"
]

snr = -40

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

def get_reverb(room, speech_source,speech_path):
    fs, audio = wavfile.read(speech_path)
    audio_len = len(audio)
    # 更新声源
    room.sources = []
    room.add_source(speech_source, signal=audio)
    room.simulate()
    mic_signals = room.mic_array.signals
    
    return mic_signals, audio_len
 

idx = 1
   
for i in range(num_room):
    x = np.random.uniform(3, room_x)
    y = np.random.uniform(3, room_y)
    z = room_z
    corners = np.array([[0,0],[0,y], [x,y], [x,0]]).T
    room = pra.Room.from_corners(corners, fs=fs, max_order=3, materials=pra.Material(0.2, 0.15), ray_tracing=True, air_absorption=True)
    room.extrude(z, materials=pra.Material(0.2, 0.15)) #天花板也要设置反射系数，否则会有很长的拖尾
    room.set_ray_tracing(receiver_radius=0.5, n_rays=10000, energy_thres=1e-5)
    for j in range(utt_per_room):
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
            rev_speech, ori_length = get_reverb(room, speech_source, speech_path)
            rev_speech = adjust_matrix_dimension(rev_speech, ori_length, 1)
            rev_noise = []
            for m in range(noise_num):
                rev_noise_source,_ = get_reverb(room, noise_sources[m], noise_paths[m])
                rev_noise_source = adjust_matrix_dimension(rev_noise_source, ori_length, 1)
                rev_noise.append(rev_noise_source)
            
            # 创建一个用于存储所有噪声信号总和的数组
            total_noise = np.zeros_like(rev_noise[0])

            # 按通道对 rev_noise 中的所有噪声信号进行相加
            for n in range(noise_num):
                total_noise += rev_noise[n]
            
            adjust_noise = snr_mix(rev_speech, total_noise, snr)
            mix = rev_speech + adjust_noise
            
            signal = torch.tensor(mix).float()
            
            ################## 存储波形 ########################
            # 归一化信号
            normalized_signal = signal / torch.max(torch.abs(signal))

            # 保存归一化后的信号
            torchaudio.save("./test.wav", normalized_signal, sample_rate=fs)
            
        else :
            print("The generated room does not meet the criteria! ")
            continue
        
      
        
