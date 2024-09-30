# sonosthesia-audio-pipeline

Python based tooling to analyse audio files and write results to file for use in realtime visualization apps. Results can be written using Message Pack for efficient (de)serialization or JSON for human readable output. Readers are provided for the Unity timeline, to be used alongside the original audio files.


# Installation

Installation requires python (version 3.9 to 3.12 are supported). Once you have python you can run

```pip install sonosthesia-audio-pipeline --upgrade```

# Quick start

Note that you might need to use the `python3` command rather than `python` depending on your setup. File path is absolute or relative to the working directory. Supported audio files are `.mp3` and `.wav`.

Note that the `--input` (shorthand `-i`) can also be a directory in which case files with appropriate extensions will be processed in order.

Use `--help` (shorthand `h`) to get argument list for each of the subcommands. For example 

```
python -m sonosthesia_audio_pipeline analysis -h
```

## Anaylysis

Will generate an `.xaa` analysis file alongside the audio file.

```
python -m sonosthesia_audio_pipeline analysis -i File.mp3
```


## Separation 

Will create seperated files in a directory with the audio file's name (without extension), nested by demucs separation model type as described [here](https://github.com/adefossez/demucs?tab=readme-ov-file#separating-tracks). The default model is `mdx_extra`.

```
python -m sonosthesia_audio_pipeline separation -i File.mp3
```

To specify another model use `--model` (shorthand `-n`)

```
python -m sonosthesia_audio_pipeline separation -i File.mp3 -n mdx
```

## Pipeline

Pipeline runs the source separation on an audio file, then runs analysis on the original audio file as well as the separated ones

```
python -m sonosthesia_audio_pipeline pipeline -i File.mp3
```


# Python Pipeline

## Source Separation

Currently using [Demucs](https://github.com/adefossez/demucs) because it seems to score better on overall SDR and is a lot easier to install with pip than Spleeter.

## Sound Analysis

This is a high level description, for output file schemas see the `Output file specification` section.

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
    "time": 0.0,
    "rms": 0.0,
    "lows": 0.0,
    "mids": 0.0,
    "highs": 0.0,
    "centroid": 0.0
}
```

### Peak

Discrete events describing a detected peak in the 

```
{
    "channel": 0
    "start": 0.0,
    "duration": 0.0,
    "magnitude": 0.0,
    "strength": 0.0
}
```

- channel is 0 (main), 1 (lows), 2 (mids), 3 (highs)
- start is the peak start time in seconds
- duration is the peak start time in seconds
- magnitude is the max peak magnitude in dB
- strength is max the onset envelope (normalized)

### Info

There is an info field which contains meta data and stats about the analysis for easy retrieval

```
{
    "info": {
      "duration": 277.0721088435374,
      "main": {
        "band": {
          "lower": 20.0,
          "upper": 8000.0
        },
        "magnitude": {
          "lower": -87.18254089355469,
          "upper": -7.182541847229004
        },
        "peaks": 699
      },
      "lows": {
        "band": {
          "lower": 30.0,
          "upper": 100.0
        },
        "magnitude": {
          "lower": -86.90585327148438,
          "upper": -6.905849456787109
        },
        "peaks": 1823
      },
      "mids": {
        "band": {
          "lower": 500.0,
          "upper": 2000.0
        },
        "magnitude": {
          "lower": -94.8998794555664,
          "upper": -14.899882316589355
        },
        "peaks": 669
      },
      "highs": {
        "band": {
          "lower": 4000.0,
          "upper": 16000.0
        },
        "magnitude": {
          "lower": -95.01151275634766,
          "upper": -15.01151180267334
        },
        "peaks": 691
      },
      "centroid": {
        "lower": 0.0,
        "upper": 8195.748929100935
      }
    }
   
}
```


### Planned

Look into using [Essentia](https://essentia.upf.edu/documentation.html) which seems to be good for highler level musical descriptors.


# Readers 


## Unity Timeline 

A Unity Timeline reader for analysis files is provided. A demo application is available [here](https://github.com/jbat100/sonosthesia-unity-audio-pipeline)


## Planned

Planning reader for Unreal Engine and Swift


# Output file specification

## Binary MsgPack (.xaa)

Header for the file is three 32 bit integers used to determine version and reserved for future use. Rest of the file is message pack data with the following JSON equivalent shema [here](https://github.com/jbat100/sonosthesia-audio-pipeline/blob/main/schemas/msgpack-version2.json)  

## Human readable JSON (.json)

Primarily used for investigation and debugging purposes. JSON schema available [here](https://github.com/jbat100/sonosthesia-audio-pipeline/blob/main/schemas/schema-version2.json). In order to write JSON analysis files, you can specify the `-j` argument



## Converting between .xaa and .json

You can convert between formats using 

```
python -m sonosthesia_audio_pipeline conversion -i File.xaa
```

```
python -m sonosthesia_audio_pipeline conversion -i File.json
```

