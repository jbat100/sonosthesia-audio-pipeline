import matplotlib.pyplot as plt
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_file = "../audio/kepler STEMS BASS.mp3"

y, sr = librosa.load(audio_file)
fig, ax = plt.subplots(nrows=3, sharex=True)
librosa.display.waveshow(y=y, sr=sr, ax=ax[0], color="blue")
ax[0].set(title='Envelope view, mono')
ax[0].label_outer()

plt.show()