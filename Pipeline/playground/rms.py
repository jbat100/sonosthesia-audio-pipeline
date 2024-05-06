import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


audio_file = "../audio/kepler STEMS BASS.mp3"

y, sr = librosa.load(audio_file, offset=19, duration=15)

S, phase = librosa.magphase(librosa.stft(y))
rms = librosa.feature.rms(S=S)

fig, ax = plt.subplots(nrows=3, sharex=True)

times = librosa.times_like(rms)
ax[0].semilogy(times, rms[0], label='RMS Energy')
ax[0].set(xticks=[])
ax[0].legend()
ax[0].label_outer()

librosa.display.waveshow(y=y, sr=sr, ax=ax[2], color="gray")

librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time', ax=ax[1])

ax[1].set(title='log Power spectrogram')

plt.show()