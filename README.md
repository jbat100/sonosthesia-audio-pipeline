# sonosthesia-audio-pipeline

Python (Librosa) based tooling to analyse audio files and write results to file for use in realtime visualization apps. Results can be written using Message Pack or raw binary float arrays for efficient (de)serialization. Readers are provided for the Unity timeline, to be used alongside the original audio files.

# Python Pipeline

## Analysis with librosa

Librosa is used to extract audio features which are of particular interest for driving reactive visuals, notably:

- Beats and tempo
- RMS magnitude
- Energy in low, mid and high frequency bands 
- Onsets
- Spectral centroid and bandwidth 

The analysis data is writen to a file using MessagePack which is highly efficient both in terms of size and (de)serialization performance. The serialized data is an array of dictionaries each of which represents a time step

```
{
    'time': float,
    'rms': float,
    'lows': float,
    'mids': float,
    'highs': float,
    'centroid': float,
    'onset': bool
}
```

There is a preview mode which uses matplotlib to present analysis data 

![kepler](https://github.com/jbat100/sonosthesia-audio-pipeline/assets/1318918/aa2ef61a-0c2f-409c-8e7d-be3f6c92c8ed)


## Planed

Allow source separation using frameworks such as [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch), [Demucs](https://github.com/adefossez/demucs) or [Spleeter](https://github.com/deezer/spleeter)

## Notes

- Pipeline is a PyCharm project
- Easiest to install python with [chocolatey](https://community.chocolatey.org/packages/python312) or homebrew
- Note works with Python up to 3.12, issues with 3.13 as both librosa and matplotlib do not support it 
- Exit command prompt on windows with Ctrl Z and enter
- Locate interpreter on windows with ```python -c "import os, sys; print(os.path.dirname(sys.executable))"```

# Unity Timeline 

The [com.sonosthesia.audio](https://github.com/jbat100/sonosthesia-unity-packages/tree/main/packages/com.sonosthesia.audio) package provides tooling which allows audio analysis files generated using the Python Pipeline described above to be played alongside corresponding timeline audio through sonosthesia [signals](https://github.com/jbat100/sonosthesia-unity-packages/tree/main/packages/com.sonosthesia.signal) 
