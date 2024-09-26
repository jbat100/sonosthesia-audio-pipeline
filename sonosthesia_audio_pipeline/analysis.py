import argparse
import collections
import librosa.display
import librosa.feature
import numpy as np

from scipy.signal import butter, sosfilt
from colorama import just_fix_windows_console
from termcolor import colored

from sonosthesia_audio_pipeline.utils import (input_to_filepaths, change_extension, remap,
                   AUDIO_EXTENSIONS, ANALYSIS_VERSION, MSGPACK_ANALYSIS_EXTENSION, JSON_ANALYSIS_EXTENSION,
                   write_packed_with_header, write_json_with_header, signal_analysis_description)

ANALYSIS_DESCRIPTION = 'Analyse an audio file and write to .axd file.'

LOW_BAND = [30, 100]
MID_BAND = [500, 2000]
HIGH_BAND = [4000, 16000]

# using msgpack with nested named tuples is a bit of a pain especially for compatibility with other environments

SignalAnalysis = collections.namedtuple('SignalAnalysis', ['rms', 'peaks'])
ContinuousData = collections.namedtuple('ContinuousData', ['time', 'rms', 'lows', 'mids', 'highs', 'centroid'])
AudioAnalysis = collections.namedtuple('AudioAnalysis', ['continuous', 'peaks', 'info'])
PeakData = collections.namedtuple('PeakData', ['channel', 'start', 'duration', 'magnitude', 'strength'])

# info tuples used to provide information on the analysis settings and statistics

RangeInfo = collections.namedtuple('RangeInfo', ['lower', 'upper'])
SignalInfo = collections.namedtuple('BandInfo', ['band', 'magnitude', 'peaks'])
AnalysisInfo = collections.namedtuple('AnalysisInfo', ['duration', 'main', 'lows', 'mids', 'highs', 'centroid'])

just_fix_windows_console()


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


def run_band_analysis(data, sr, band, channel):
    y = butter_bandpass_filter(data, band[0], band[1], sr, order=5)
    return run_signal_analysis(y, sr, channel)


def run_signal_analysis(y, sr, channel):
    rms = get_rms(y)
    peaks = get_peaks(y, sr, rms, channel)
    return SignalAnalysis(rms, peaks)


def signal_analysis_info(signal_analysis, band):
    rms_max = float(np.max(signal_analysis.rms))
    rms_min = float(np.min(signal_analysis.rms))
    num_peaks = len(signal_analysis.peaks)
    return SignalInfo(RangeInfo(float(band[0]), float(band[1]))._asdict(),
                      RangeInfo(rms_min, rms_max)._asdict(),
                      num_peaks)._asdict()


def range_info(array):
    return RangeInfo(float(np.min(array)), float(np.max(array)))._asdict()


def run_audio_analysis(y, sr, low_band, mid_band, high_band):

    peaks = []

    analysis_main = run_signal_analysis(y, sr, 0)
    times = librosa.times_like(analysis_main.rms, sr=sr)
    peaks.extend(analysis_main.peaks)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr).ravel()
    print(colored(f'Main analysis : {signal_analysis_description(analysis_main)}', "light_grey"))
    main_info = signal_analysis_info(analysis_main, [20, 8000])
    cent_info = range_info(cent)

    analysis_lows = run_band_analysis(y, sr, low_band, 1)
    peaks.extend(analysis_lows.peaks)
    print(colored(f'Lows analysis : {signal_analysis_description(analysis_lows)}', "light_grey"))
    lows_info = signal_analysis_info(analysis_lows, low_band)

    analysis_mids = run_band_analysis(y, sr, mid_band, 2)
    peaks.extend(analysis_mids.peaks)
    print(colored(f'Mids analysis : {signal_analysis_description(analysis_mids)}', "light_grey"))
    mids_info = signal_analysis_info(analysis_mids, mid_band)

    analysis_highs = run_band_analysis(y, sr, high_band, 3)
    peaks.extend(analysis_highs.peaks)
    print(colored(f'Highs analysis : {signal_analysis_description(analysis_highs)}', "light_grey"))
    highs_info = signal_analysis_info(analysis_highs, high_band)

    sorted_peaks = sorted(peaks, key=lambda p: p.start)
    count = len(times)

    analysis_info = AnalysisInfo(times[-1], main_info, lows_info, mids_info, highs_info, cent_info)._asdict()

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
        peak_dicts.append({
            'channel': int(peak.channel),
            'start': float(peak.start),
            'duration': float(peak.duration),
            'magnitude': float(peak.magnitude),
            'strength': float(peak.strength),
        })
    return AudioAnalysis(continuous_dicts, peak_dicts, analysis_info)._asdict()


def run_analysis(file_path, start, duration, write_json):
    y, sr = librosa.load(file_path, sr=None, offset=start, duration=duration)
    num_samples = y.shape[0] 
    duration = num_samples / sr
    print(colored(f'Loaded {file_path}, got {num_samples} samples at rate {sr}, estimated duration is {duration}', "green"))
    audio_analysis = run_audio_analysis(y, sr, LOW_BAND, MID_BAND, HIGH_BAND)
    msgpack_analysis_path = change_extension(file_path, MSGPACK_ANALYSIS_EXTENSION)
    audio_analysis_dict = audio_analysis
    write_packed_with_header(audio_analysis_dict, [ANALYSIS_VERSION, 0, 0], msgpack_analysis_path)
    if write_json:
        json_analysis_path = change_extension(file_path, JSON_ANALYSIS_EXTENSION)
        write_json_with_header(audio_analysis_dict, [ANALYSIS_VERSION, 0, 0], json_analysis_path)


def configure_analysis_parser(parser):
    parser.add_argument('-i', '--input', type=str, nargs='?', required=True,
                        help=f'path to the file ({",".join(AUDIO_EXTENSIONS)}) or directory')
    parser.add_argument('-s', '--start', type=float, default=0.0,
                        help='start time in seconds')
    parser.add_argument('-d', '--duration', type=float, default=None,
                        help='duration in seconds')
    parser.add_argument('-j', '--json', action='store_true',
                        help='write output to json')


def analysis_with_args(args):
    file_paths = input_to_filepaths(args.input, AUDIO_EXTENSIONS)
    for file_path in file_paths:
        run_analysis(file_path, args.start, args.duration, args.json)
    print('Done')


def analysis():
    parser = argparse.ArgumentParser(description=ANALYSIS_DESCRIPTION)
    configure_analysis_parser(parser)
    args = parser.parse_args()
    analysis_with_args(args)


if __name__ == "__main__":
    analysis()

