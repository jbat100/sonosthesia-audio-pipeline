import numpy as np
import os


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
