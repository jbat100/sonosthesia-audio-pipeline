import collections
import matplotlib.pyplot as plt
import librosa.display
import librosa.feature
import librosa.util
import numpy as np
from scipy.signal import butter, sosfilt

Peak = collections.namedtuple('Peak', ['start', 'end', 'magnitude'])


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter(order, [lowcut, highcut], fs=fs, output='sos', btype='bandpass')
    y = sosfilt(sos, data)
    return y


def show_filtered(data, sr, fmin, fmax, original_ax, filtered_ax):
    y = butter_bandpass_filter(data, fmin, fmax, sr, order=5)
    librosa.display.waveshow(y=data, sr=sr, ax=original_ax, color="blue")
    librosa.display.waveshow(y=y, sr=sr, ax=filtered_ax, color="blue")
    return


def get_onset_envelope(y, sr, normalize):
    onset_envelope = librosa.onset.onset_strength(y=y, sr=sr)
    if normalize:
        onset_envelope = onset_envelope - np.min(onset_envelope, keepdims=True, axis=-1)
        onset_envelope /= np.max(onset_envelope, keepdims=True, axis=-1) + librosa.util.tiny(onset_envelope)
    return onset_envelope


def show_processed(data, sr, fmin, fmax, spectrum_ax, onset_ax):

    y = butter_bandpass_filter(data, fmin, fmax, sr, order=5)
    D = np.abs(librosa.stft(y))

    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis='time', y_axis='log', ax=spectrum_ax)
    spectrum_ax.set(title='Power spectrogram')
    spectrum_ax.label_outer()

    onset_envelope = get_onset_envelope(y, sr, True)

    times = librosa.times_like(onset_envelope, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_envelope, sr=sr, backtrack=False, units='frames')
    backtracked_frames = librosa.onset.onset_backtrack(onset_frames, onset_envelope)

    onset_ax.plot(times, onset_envelope, label='Onset strength')
    onset_ax.vlines(times[onset_frames], 0, onset_envelope.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
    onset_ax.vlines(times[backtracked_frames], 0, onset_envelope.max(), color='g', alpha=0.9, linestyle='--', label='Backtracked')
    onset_ax.legend()


def print_processed(data, sr, fmin, fmax, ax1, ax2):
    y = butter_bandpass_filter(data, fmin, fmax, sr, order=5)
    onset_envelope = get_onset_envelope(y, sr, True)
    times = librosa.times_like(onset_envelope, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_envelope, sr=sr, backtrack=False, units='frames')
    backtracked_frames = librosa.onset.onset_backtrack(onset_frames, onset_envelope)
    peaks = []
    for i in range(len(onset_frames)):
        start = times[backtracked_frames[i]]
        end = times[onset_frames[i]]
        magnitude = onset_envelope[onset_frames[i]]
        peaks.append(Peak(start, end, magnitude))
    print(f"Extracted {len(onset_frames)} peaks for frequency range {fmin} {fmax}")
    print(peaks)
    return peaks


def main():
    audio_file = "F:\\Sonosthesia\\music\\ES_kepler - dreem\\kepler STEMS DRUMS.mp3"

    data, sr = librosa.load(audio_file, offset=26, duration=15)

    D = np.abs(librosa.stft(data))
    D_2 = D**2

    fig, ax = plt.subplots(nrows=7, sharex=True)

    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis='time', y_axis='log', ax=ax[0])
    ax[0].set(title='Power spectrogram')
    ax[0].label_outer()

    action = print_processed

    action(data, sr, 20, 100, ax[1], ax[2])
    action(data, sr, 500, 1000, ax[3], ax[4])
    action(data, sr, 4000, 8000, ax[5], ax[6])

    plt.show()

main()