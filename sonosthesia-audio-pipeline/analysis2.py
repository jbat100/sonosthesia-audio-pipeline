import argparse
import collections
import os
import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
import numpy as np

from colorama import just_fix_windows_console
from termcolor import colored
from setup import input_to_filepaths
from utils import change_extension, remap
from analysis import butter_bandpass_filter, get_rms, get_peaks, write_packed_with_header, read_packed_with_header

# using msgpack with nested named tuples is a bit of a pain especially for compatibility with other environments

SignalAnalysis = collections.namedtuple('SignalAnalysis', ['rms', 'peaks'])
ContinuousData = collections.namedtuple('ContinuousData', ['time', 'rms', 'lows', 'mids', 'highs', 'centroid'])
AudioAnalysis = collections.namedtuple('AudioAnalysis', ['continuous', 'peaks'])

AUDIO_EXTENSIONS = {'.wav', '.mp3'}

CHANNEL_KEYS = {
    0: 'rms',
    1: 'lows',
    2: 'mids',
    3: 'highs'
}

just_fix_windows_console()


def run_band_analysis(data, sr, band, channel):
    y = butter_bandpass_filter(data, band[0], band[1], sr, order=5)
    return run_signal_analysis(y, sr, channel)


def run_signal_analysis(y, sr, channel):
    rms = get_rms(y)
    peaks = get_peaks(y, sr, rms, channel)
    return SignalAnalysis(rms, peaks)


def signal_analysis_description(signal_analysis):
    rms_max = np.max(signal_analysis.rms)
    rms_min = np.min(signal_analysis.rms)
    num_peaks = len(signal_analysis.peaks)
    return f'SignalAnalysis with rms range ({rms_min} : {rms_max}) and {num_peaks} peaks'


def audio_analysis_description(audio_analysis):
    continuous = audio_analysis['continuous']
    peaks = audio_analysis['peaks']
    return f'AudioAnalysis with {len(continuous)} continuous data points and {len(peaks)} peaks'


def run_audio_analysis(y, sr, low_band, mid_band, high_band):

    peaks = []

    analysis_main = run_signal_analysis(y, sr, 0)
    times = librosa.times_like(analysis_main.rms, sr=sr)
    peaks.extend(analysis_main.peaks)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr).ravel()
    print(colored(f'Main analysis : {signal_analysis_description(analysis_main)}', "light_grey"))

    analysis_lows = run_band_analysis(y, sr, low_band, 1)
    peaks.extend(analysis_lows.peaks)
    print(colored(f'Lows analysis : {signal_analysis_description(analysis_lows)}', "light_grey"))

    analysis_mids = run_band_analysis(y, sr, mid_band, 2)
    peaks.extend(analysis_mids.peaks)
    print(colored(f'Mids analysis : {signal_analysis_description(analysis_mids)}', "light_grey"))

    analysis_highs = run_band_analysis(y, sr, high_band, 3)
    peaks.extend(analysis_highs.peaks)
    print(colored(f'Highs analysis : {signal_analysis_description(analysis_highs)}', "light_grey"))

    sorted_peaks = sorted(peaks, key=lambda p: p.start)
    count = len(times)

    continuous_dicts = []
    for frame in range(count):
        continuous_dicts.append({
            'time': float(times[frame]),
            'rms': float(analysis_main.rms[frame]),
            'lows': float(analysis_lows.rms[frame]),
            'mids': float(analysis_mids.rms[frame]),
            'highs': float(analysis_highs.rms[frame]),
            'centroid': float(cent[frame])
        })
    peak_dicts = []
    for peak in sorted_peaks:
        peak_dicts.append(peak._asdict())
    return AudioAnalysis(continuous_dicts, peak_dicts)


def run_analysis2(file_path, start, duration):
    y, sr = librosa.load(file_path, sr=None, offset=start, duration=duration)
    num_samples = y.shape[0] 
    duration = num_samples / sr
    print(colored(f'Loaded {file_path}, got {num_samples} samples at rate {sr}, estimated duration is {duration}'), "green")
    audio_analysis = run_audio_analysis(y, sr, [30, 200], [500, 2000], [4000, 16000])
    write_packed_with_header(audio_analysis._asdict(), [2, 0, 0], change_extension(file_path, '.xad'))


def remap_dbs(array):
    return remap(array, -40, -5, 0, 1)


def plot_rms(continuous, extractor, label, ax):
    times = [data['time'] for data in continuous]
    values = remap_dbs(np.array([extractor(data) for data in continuous]))
    ax.plot(times, values, label=label)
    ax.legend()


def plot_peaks(peaks, channel, label, ax):
    channel_peaks = [peak for peak in peaks if peak['channel'] == channel]
    count = len(channel_peaks)
    times = [peak['start'] for peak in channel_peaks]
    magnitudes = remap_dbs(np.array([peak['magnitude'] for peak in channel_peaks]))
    strengths = [peak['strength'] for peak in channel_peaks]
    ax.vlines(times, 0, magnitudes, color='r', alpha=0.9, linestyle='--', label=f'{label} magnitudes')
    ax.scatter(times, strengths, color='g', marker='o', label=f'{label} strengths')


def plot_analysis(continuous, peaks, channel, ax1, ax2):
    key = CHANNEL_KEYS[channel]
    plot_rms(continuous, lambda d: d[key], key.capitalize(), ax1)
    plot_peaks(peaks, channel, key.capitalize(), ax2)


def load_analysis2(file_path, start, duration):
    end = float('inf') if duration is None else start + duration
    header, data = read_packed_with_header(file_path)
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


def configure_analysis2_parser(parser):
    parser.add_argument('-i', '--input', type=str, nargs='?', required=True,
                        help='path to the file (.wav, .mp3, .xad) or directory')
    parser.add_argument('-s', '--start', type=float, default=0.0,
                        help='start time in seconds')
    parser.add_argument('-d', '--duration', type=float, default=None,
                        help='duration in seconds')


def analysis2():
    parser = argparse.ArgumentParser(description='Analyse an audio file or display an analysis file')
    configure_analysis2_parser(parser)
    args = parser.parse_args()
    file_paths = input_to_filepaths(args.input)
    for file_path in file_paths:
        _, ext = os.path.splitext(file_path)
        if ext.lower() in AUDIO_EXTENSIONS:
            run_analysis2(file_path, args.start, args.duration)
        elif ext.lower() == '.xad':
            load_analysis2(file_path, args.start, args.duration)
        else:
            print(colored(f'Skipped file : {file_path}', "yellow"))
    print('Done')


if __name__ == "__main__":
    analysis2()

