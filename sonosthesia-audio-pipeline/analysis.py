import argparse
import librosa.display
from array import array
import msgpack
import numpy as np

from setup import input_to_filepaths
from utils import change_extension, normalize_array_01, clip_bin_signal


def run_analysis(file_path, raw):

    hop_length = 512

    y, sr = librosa.load(file_path, sr=None)
    num_samples = y.shape[0]
    duration = num_samples / sr

    print(f'Loaded {file_path}, got {num_samples} samples at rate {sr}, estimated duration is {duration}')

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    frame_times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)

    print(f'Extracted tempo {tempo}, with {beats.shape[0]} beats')

    stft = librosa.stft(y)
    stft_mag, stft_phase = librosa.magphase(stft)
    stft_mag_db = librosa.amplitude_to_db(stft_mag, ref=np.max)
    rms = librosa.feature.rms(S=stft_mag)[0]
    rms_db = normalize_array_01(librosa.amplitude_to_db(rms, ref=np.max))
    num_frames = stft_mag_db.shape[1]
    samples_per_frame = num_samples / num_frames

    print(f'Computed stft with hop length {hop_length}, got {stft_mag_db.shape} frames, {samples_per_frame} samples per frame')

    MS = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=24, fmax=16000)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr).ravel()

    lows = normalize_array_01(clip_bin_signal(librosa.power_to_db(np.sum(MS[:1, :], axis=0))))
    mids = normalize_array_01(clip_bin_signal(librosa.power_to_db(np.sum(MS[3:4, :], axis=0))))
    highs = normalize_array_01(clip_bin_signal(librosa.power_to_db(np.sum(MS[14:22, :], axis=0))))

    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_bts = librosa.onset.onset_backtrack(onset_frames, onset_env)

    data_points = []

    for frame in range(num_frames):
        data_point = {
            'time': float(frame_times[frame]),
            'rms': float(rms_db[frame]),
            'lows': float(lows[frame]),
            'mids': float(mids[frame]),
            'highs': float(highs[frame]),
            'centroid': float(cent[frame]),
            'onset': False
        }
        data_points.append(data_point)

    for onset_bt in onset_bts:
        data_point = data_points[onset_bt]
        data_point["onset"] = True

    if raw:
        write_raw(data_points, file_path)
    else:
        write_packed(data_points, file_path)


def write_packed(data_points, file_path):
    packed_data_points = msgpack.packb(data_points, use_bin_type=True)
    output_file = change_extension(file_path, '.aad')
    print(f'Packed {len(data_points)} sonosthesia-pipeline points into {len(packed_data_points)} bytes')
    with open(output_file, 'wb') as file:
        file.write(packed_data_points)


def write_raw(data_points, file_path):
    raw_points = []
    for data_point in data_points:
        raw_points.extend([
            data_point['time'],
            data_point['rms'],
            data_point['lows'],
            data_point['mids'],
            data_point['highs'],
            data_point['centroid'],
            float(data_point['onset'])
        ])
    # note : using raw float 32-bit data is better than using msgpack to serialize an array of float
    float_data_points = array('f', raw_points)
    raw_data_points = float_data_points.tobytes()
    raw_output_file = change_extension(file_path, '.aadf')
    print(f'Packed {len(raw_points)} raw floats into {len(raw_data_points)} bytes')
    with open(raw_output_file, 'wb') as file:
        file.write(raw_data_points)


def analysis_parser():
    parser = argparse.ArgumentParser(description='Analyse an audio file')
    parser.add_argument('-i', '--input', type=str, nargs='?', default='audio/kepler.mp3',
                        help='path to the file or directory')
    parser.add_argument('-r', '--raw', action='store_true', help='output in raw format')
    return parser


def analysis():
    args = analysis_parser().parse_args()
    file_paths = input_to_filepaths(args.input)
    for audio_file in file_paths:
        run_analysis(audio_file, args.raw)
    print('Done')


if __name__ == "__main__":
    analysis()






