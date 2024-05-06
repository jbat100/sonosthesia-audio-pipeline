import matplotlib.pyplot as plt
import librosa.display
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np

audio_file = "../audio/kepler STEMS DRUMS.mp3"

y, sr = librosa.load(audio_file, offset=26, duration=15)

D = np.abs(librosa.stft(y))**2
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=3, fmax=10000)

fig, ax = plt.subplots()
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')

plt.show()