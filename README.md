# sonosthesia-audio-pipeline

Python based tooling to analyse audio files and write results to file for use in realtime visualization apps. Results can be written using Message Pack for efficient (de)serialization or JSON for human readable output. Readers are provided for the Unity timeline, to be used alongside the original audio files.


# Installation

Installation requires python (version 3.9 to 3.12 are supported). Once you have python you can run

```pip install sonosthesia-audio-pipeline --upgrade```


# Python Pipeline

## Source Separation

Currently using [Demucs](https://github.com/adefossez/demucs) because it seems to score better on overall SDR and is a lot easier to install with pip than Spleeter.

## Sound Analysis

Librosa is used to extract audio features which are of particular interest for driving reactive visuals, notably:

- Beats and tempo
- RMS magnitude
- Energy in low, mid and high frequency bands 
- Onsets
- Spectral centroid and bandwidth 

The analysis contains various kinds of data 

### Continuous

Provided for each analysis from, with 512 sample hop size

```
{
    'time': float,
    'rms': float,
    'lows': float,
    'mids': float,
    'highs': float,
    'centroid': float
}
```

### Peak

Discrete events describing a detected peak in the 

```
{
    'channel': int
    'start': float,
    'duration': float,
    'magnitude': float,
    'strength': float
}
```

- channel is 0 (main), 1 (lows), 2 (mids), 3 (highs)
- start is the peak start time in seconds
- duration is the peak start time in seconds
- magnitude is the max peak magnitude in dB
- strength is max the onset envelope (normalized)

### Planned

Look into using [Essentia](https://essentia.upf.edu/documentation.html) which seems to be good for highler level musical descriptors.


# Readers 



## Unity Timeline 

The [com.sonosthesia.audio](https://github.com/jbat100/sonosthesia-unity-packages/tree/main/packages/com.sonosthesia.audio) package provides tooling which allows audio analysis files generated using the Python Pipeline described above to be played alongside corresponding timeline audio through sonosthesia [signals](https://github.com/jbat100/sonosthesia-unity-packages/tree/main/packages/com.sonosthesia.signal) 


# Output file specification

## Binary MsgPack (.xaa)



## Human readable JSON (.json)

Primarily used for investigation and debugging purposes. JSON schema available [here](schemas/schema-version2.json) 

## Converting between .xaa and .json

