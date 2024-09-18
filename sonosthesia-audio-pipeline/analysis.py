import argparse
import collections
import os
import struct
import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
import msgpack
import numpy as np

from colorama import just_fix_windows_console
from termcolor import colored
from scipy.signal import butter, sosfilt
from setup import input_to_filepaths
from utils import change_extension, remap

PeakData = collections.namedtuple('PeakData', ['channel', 'start', 'duration', 'magnitude', 'strength'])


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter(order, [lowcut, highcut], fs=fs, output='sos', btype='bandpass')
    y = sosfilt(sos, data)
    return y


def get_onset_envelope(y, sr, normalize):
    onset_envelope = librosa.onset.onset_strength(y=y, sr=sr)
    if normalize:
        onset_envelope = onset_envelope - np.min(onset_envelope, keepdims=True, axis=-1)
        onset_envelope /= np.max(onset_envelope, keepdims=True, axis=-1) + librosa.util.tiny(onset_envelope)
    return onset_envelope


def get_peaks(y, sr, rms, channel):
    onset_envelope = get_onset_envelope(y, sr, True)
    times = librosa.times_like(onset_envelope, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_envelope, sr=sr, backtrack=False, units='frames')
    backtracked_frames = librosa.onset.onset_backtrack(onset_frames, onset_envelope)
    peaks = []
    for i in range(len(onset_frames)):
        start = times[backtracked_frames[i]]
        duration = times[onset_frames[i]] - start
        magnitude = rms[onset_frames[i]]
        strength = onset_envelope[onset_frames[i]]
        peaks.append(PeakData(channel, float(start), float(duration), float(magnitude), float(strength)))
    return peaks


def get_rms(y):
    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.amplitude_to_db(librosa.feature.rms(S=S)[0])
    return rms


def write_packed_with_header(data, header, file_path):
    if len(header) != 3:
        raise ValueError("header_integers must contain exactly three 32-bit integers")
    # Pack the header integers as 32-bit (4 bytes) integers
    header_packed = struct.pack('iii', *header)  # 'iii' means 3 int32 values
    packed_data = msgpack.packb(data, use_bin_type=True)
    combined_data = header_packed + packed_data
    print(f'Packed {len(data.continuous)} continuous data points and {len(data.peaks)} peaks into {len(combined_data)} '
          f'bytes, written to {file_path} with header {header}')
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