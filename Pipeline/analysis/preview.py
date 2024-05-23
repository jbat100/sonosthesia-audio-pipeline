import os
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from setup import parse_configuration
from utils import normalize_array, normalize_array_01


def preview(audio_file):

    hop_length = 512
    fig, ax = plt.subplots(nrows=4)

    y, sr = librosa.load(audio_file, offset=26, duration=15)
    num_samples = y.shape[0]
    duration = num_samples / sr

    print(f'Loaded {audio_file}, got {num_samples} samples at rate {sr}, estimated duration is {duration}')

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)

    librosa.display.waveshow(y=y, sr=sr, ax=ax[0], color="lightgray")
    ax[0].vlines(onset_times[beats], -1, 1, alpha=0.5, color='r', linestyle='-', label='Beats')
    ax[0].label_outer()

    print(f'Extracted tempo {tempo}, with {beats.shape[0]} beats')

    stft = librosa.stft(y)
    stft_mag, stft_phase = librosa.magphase(stft)
    stft_mag_db = librosa.amplitude_to_db(stft_mag, ref=np.max)
    rms = librosa.feature.rms(S=stft_mag)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    rms_times = librosa.times_like(rms_db)
    num_frames = stft_mag_db.shape[1]
    samples_per_frame = num_samples / num_frames

    ax[0].plot(rms_times, librosa.util.normalize(rms_db), label='RMS', color='gray')
    ax[0].legend()

    print(
        f'Computed stft with hop length {hop_length}, got {stft_mag_db.shape} frames, {samples_per_frame} samples per frame')

    # Spectral features : note spectral centroid and bandwidth seem to follow each other quite closely

    MS = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=12, fmax=8000)
    MS_dB = librosa.power_to_db(MS, ref=np.max)
    librosa.display.specshow(MS_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax[1])
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_times = librosa.times_like(cent)
    ax[1].plot(cent_times, cent.T, label='Centroid', color='lightblue')
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_bw_times = librosa.times_like(spec_bw)
    ax[1].plot(spec_bw_times, spec_bw.T, label='Bandwidth', color='lightgreen')
    ax[1].legend()

    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    librosa.display.waveshow(y=y, sr=sr, ax=ax[2], color="lightgray")

    onset_bt = librosa.onset.onset_backtrack(onset_frames, onset_env)

    ax[2].plot(onset_times, librosa.util.normalize(onset_env), label='Onset strength')
    ax[2].vlines(onset_times[onset_frames], -1, 1, color='lightcoral', alpha=0.9, linestyle='-', label='Onsets')
    ax[2].vlines(librosa.frames_to_time(onset_bt), -1, 1, label='Backtracked', color='indianred')
    ax[2].legend()

    lows = librosa.power_to_db(np.sum(MS[:4, :], axis=0))
    mids = librosa.power_to_db(np.sum(MS[4:8, :], axis=0))
    highs = librosa.power_to_db(np.sum(MS[8:12, :], axis=0))

    librosa.display.waveshow(y=y, sr=sr, ax=ax[3], color='lightgray')

    ax[3].plot(onset_times, normalize_array_01(highs), color='cornflowerblue', label='Highs')
    ax[3].plot(onset_times, normalize_array_01(mids), color='mediumseagreen', label='Mids')
    ax[3].plot(onset_times, normalize_array_01(lows), color='darksalmon', label='Lows')

    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    fig.canvas.manager.set_window_title(file_name)


if __name__ == "__main__":

    configuration = parse_configuration()

    for audio_file in configuration.file_paths:
        preview(audio_file)

    plt.show()
