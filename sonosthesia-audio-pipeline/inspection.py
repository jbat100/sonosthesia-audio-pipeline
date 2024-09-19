import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np
from colorama import just_fix_windows_console
from termcolor import colored
from utils import (input_to_filepaths, remap,
                   ANALYSIS_VERSION, CHANNEL_KEYS, MSGPACK_ANALYSIS_EXTENSION, JSON_ANALYSIS_EXTENSION,
                   read_packed_with_header, read_json_with_header, audio_analysis_description)

INSPECTION_DESCRIPTION = 'Inspect an analysis file.'

just_fix_windows_console()


def remap_dbs(array):
    return remap(array, -40, -5, 0, 1)


def plot_rms(continuous, extractor, label, ax):
    times = [data['time'] for data in continuous]
    values = remap_dbs(np.array([extractor(data) for data in continuous]))
    ax.plot(times, values, label=label)
    ax.legend()


def plot_peaks(peaks, channel, label, ax):
    channel_peaks = [peak for peak in peaks if peak['channel'] == channel]
    times = [peak['start'] for peak in channel_peaks]
    magnitudes = remap_dbs(np.array([peak['magnitude'] for peak in channel_peaks]))
    strengths = [peak['strength'] for peak in channel_peaks]
    ax.vlines(times, 0, magnitudes, color='r', alpha=0.9, linestyle='--', label=f'{label} magnitudes')
    ax.scatter(times, strengths, color='g', marker='o', label=f'{label} strengths')


def plot_analysis(continuous, peaks, channel, ax1, ax2):
    key = CHANNEL_KEYS[channel]
    plot_rms(continuous, lambda d: d[key], key.capitalize(), ax1)
    plot_peaks(peaks, channel, key.capitalize(), ax2)


def load_analysis(file_path, start, duration):
    end = float('inf') if duration is None else start + duration
    _, ext = os.path.splitext(file_path)
    header, data = None, None
    if ext == MSGPACK_ANALYSIS_EXTENSION:
        header, data = read_packed_with_header(file_path)
    elif ext == JSON_ANALYSIS_EXTENSION:
        header, data = read_json_with_header(file_path)
    if header[0] != ANALYSIS_VERSION:
        raise ValueError(f"Unexpected version {header[0]}, required {ANALYSIS_VERSION}")
    print(colored(f'Loaded analysis file from {file_path} read header {header}', "green"))
    print(colored(f'Found : {audio_analysis_description(data)}', "grey"))
    continuous = [point for point in data['continuous'] if start <= point['time'] <= end]
    peaks = [peak for peak in data['peaks'] if start <= peak['start'] <= end]
    if len(continuous) == len(peaks) == 0:
        print(colored('No data to plot', "red"))
        return
    print(colored(f'Plotting : {len(continuous)} continuous analysis points and {len(peaks)} peaks', "blue"))
    fig, ax = plt.subplots(nrows=8, sharex=True)
    plot_analysis(continuous, peaks, 0, ax[0], ax[1])
    plot_analysis(continuous, peaks, 1, ax[2], ax[3])
    plot_analysis(continuous, peaks, 2, ax[4], ax[5])
    plot_analysis(continuous, peaks, 3, ax[6], ax[7])
    plt.show()


def configure_inspection_parser(parser):
    parser.add_argument('-i', '--input', type=str, nargs='?', required=True,
                        help=f'path to an analysis {MSGPACK_ANALYSIS_EXTENSION} file or directory')
    parser.add_argument('-s', '--start', type=float, default=0.0,
                        help='start time in seconds')
    parser.add_argument('-d', '--duration', type=float, default=None,
                        help='duration in seconds')
    parser.add_argument('-j', '--json', action='store_true',
                        help='search .json files when input arg is a directory')


def inspection_with_args(args):
    extensions = [JSON_ANALYSIS_EXTENSION] if args.json else [MSGPACK_ANALYSIS_EXTENSION]
    file_paths = input_to_filepaths(args.input, extensions)
    for file_path in file_paths:
        load_analysis(file_path, args.start, args.duration)
    print('Done')


def inspection():
    parser = argparse.ArgumentParser(description=INSPECTION_DESCRIPTION)
    configure_inspection_parser(parser)
    args = parser.parse_args()
    inspection_with_args(args)


if __name__ == "__main__":
    inspection()
