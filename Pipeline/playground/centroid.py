import matplotlib.pyplot as plt
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_file = "../audio/kepler STEMS BASS.mp3"

y, sr = librosa.load(audio_file, offset=26, duration=10)
cent = librosa.feature.spectral_centroid(y=y, sr=sr)
S, phase = librosa.magphase(librosa.stft(y=y))

times = librosa.times_like(cent)
fig, ax = plt.subplots()

librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time', ax=ax)

ax.plot(times, cent.T, label='Spectral centroid', color='w')
ax.legend(loc='upper right')
ax.set(title='log Power spectrogram')

plt.show()