import numpy as np
import os
import struct
import msgpack

from colorama import just_fix_windows_console
from termcolor import colored

ANALYSIS_EXTENSION = '.xad'

AUDIO_EXTENSIONS = ['.wav', '.mp3']

ANALYSIS_VERSION = 2

CHANNEL_KEYS = {
    0: 'rms',
    1: 'lows',
    2: 'mids',
    3: 'highs'
}


def audio_analysis_description(audio_analysis):
    continuous = audio_analysis['continuous']
    peaks = audio_analysis['peaks']
    return f'AudioAnalysis with {len(continuous)} continuous data points and {len(peaks)} peaks'


def signal_analysis_description(signal_analysis):
    rms_max = np.max(signal_analysis.rms)
    rms_min = np.min(signal_analysis.rms)
    num_peaks = len(signal_analysis.peaks)
    return f'SignalAnalysis with rms range ({rms_min} : {rms_max}) and {num_peaks} peaks'


def write_packed_with_header(data, header, file_path):
    if len(header) != 3:
        raise ValueError("header_integers must contain exactly three 32-bit integers")
    # Pack the header integers as 32-bit (4 bytes) integers
    header_packed = struct.pack('iii', *header)  # 'iii' means 3 int32 values
    packed_data = msgpack.packb(data, use_bin_type=True)
    combined_data = header_packed + packed_data
    print(colored(f'Packed data points into {len(combined_data)} bytes, written to {file_path} with header {header}',"green"))
    with open(file_path, 'wb') as file:
        file.write(combined_data)


def read_packed_with_header(file_path):
    # Open the file in binary read mode
    with open(file_path, 'rb') as file:
        # Read the header: 3 x 32-bit integers (12 bytes total)
        header_packed = file.read(12)
        if len(header_packed) != 12:
            raise ValueError("File is too short to contain a valid header")
        # Unpack the header: 'iii' means 3 int32 values
        header = struct.unpack('iii', header_packed)
        # Read the remaining data (msgpack data)
        packed_data = file.read()
        if not packed_data:
            raise ValueError("No data found after header")
        # Unpack the msgpack data
        data = msgpack.unpackb(packed_data, raw=False)
    return header, data


def input_to_filepaths(input_path, extensions):
    # If the specified path is a directory, get absolute file paths of all files in the directory
    if os.path.isdir(input_path):
        file_paths = [os.path.abspath(os.path.join(input_path, file)) for file in os.listdir(input_path)
                      if file.lower().endswith(extensions)]
    else:
        file_paths = [os.path.abspath(input_path)]
    return file_paths


def remap(array, in_min, in_max, out_min, out_max):
    # Clip the value to be within the input range
    array = np.clip(array, in_min, in_max)
    # Perform the linear transformation to the new range
    return (array - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def normalize_array(array):
    array_mean_centered = array - np.mean(array)
    max_abs_value = np.max(np.abs(array_mean_centered))
    normalized_array = array_mean_centered / max_abs_value
    return normalized_array


def normalize_array_01(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array


def change_extension(file_path, new_extension):
    base = os.path.splitext(file_path)[0]
    return base + new_extension


def clip_bin_signal(signal):
    min_value = signal.min()
    max_value = signal.max()
    center = (max_value + min_value) / 2
    clipped = np.clip(signal, center, max_value)
    return clipped


