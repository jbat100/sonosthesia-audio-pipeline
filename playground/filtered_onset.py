import matplotlib.pyplot as plt
import librosa.display
import librosa.feature
import scipy.signal as signal
import numpy as np
from scipy.signal import butter, lfilter, freqz, sosfilt


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter(order, [lowcut, highcut], fs=fs, output='sos', btype='bandpass')
    y = sosfilt(sos, data)
    return y


def show_filtered(data, sr, fmin, fmax, original_ax, filtered_ax):
    y = butter_bandpass_filter(data, fmin, fmax, sr, order=5)
    librosa.display.waveshow(y=data, sr=sr, ax=original_ax, color="blue")
    librosa.display.waveshow(y=y, sr=sr, ax=filtered_ax, color="blue")
    return


def process(data, sr, fmin, fmax, spectrum_ax, onset_ax):

    y = butter_bandpass_filter(data, fmin, fmax, sr, order=5)

    librosa.onset.onset_detect(y=y, sr=sr, units='time')
    D = np.abs(librosa.stft(y))

    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis='time', y_axis='log', ax=spectrum_ax)
    spectrum_ax.set(title='Power spectrogram')
    spectrum_ax.label_outer()

    o_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(o_env, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, normalize=False, backtrack=True)

    onset_ax.plot(times, o_env, label='Onset strength')
    onset_ax.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
    onset_ax.legend()


audio_file = "F:\\Sonosthesia\\music\\ES_kepler - dreem\\kepler STEMS DRUMS.mp3"

y, sr = librosa.load(audio_file, offset=26, duration=15)

D = np.abs(librosa.stft(y))
D_2 = D**2

fig, ax = plt.subplots(nrows=7, sharex=True)

librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis='time', y_axis='log', ax=ax[0])
ax[0].set(title='Power spectrogram')
ax[0].label_outer()

if False:
    show_filtered(y, sr, 20, 100, ax[1], ax[2])
    show_filtered(y, sr, 500, 1000, ax[3], ax[4])
    show_filtered(y, sr, 4000, 8000, ax[5], ax[6])
else:
    process(y, sr, 20, 100, ax[1], ax[2])
    process(y, sr, 500, 1000, ax[3], ax[4])
    process(y, sr, 4000, 8000, ax[5], ax[6])

plt.show()