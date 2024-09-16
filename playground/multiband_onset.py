import matplotlib.pyplot as plt
import librosa.display
import librosa.feature
import numpy as np


def process_band(fmin, fmax, D, sr, ax):
    S = librosa.feature.melspectrogram(S=D, sr=sr, fmin=fmin, fmax=fmax)
    S_dB = librosa.power_to_db(S, ref=np.max)
    o_env = librosa.onset.onset_strength(sr=sr, S=S_dB, units='time')
    times = librosa.times_like(o_env, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, normalize=False, backtrack=True)
    ax.plot(times, o_env, label='Onset strength')
    ax.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
    ax.legend()


audio_file = "F:\\Sonosthesia\\music\\ES_kepler - dreem\\kepler STEMS DRUMS.mp3"

y, sr = librosa.load(audio_file, offset=26, duration=15)

D = np.abs(librosa.stft(y))
D_2 = D**2

fig, ax = plt.subplots(nrows=4, sharex=True)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis='time', y_axis='log', ax=ax[0])

ax[0].set(title='Power spectrogram')
ax[0].label_outer()

process_band(0, 40, D_2, sr, ax[1])
process_band(500, 2000, D_2, sr, ax[2])
process_band(4000, 10000, D_2, sr, ax[3])

plt.show()