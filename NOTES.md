https://github.com/pypa/sampleproject/blob/main/pyproject.toml

# General python

- Can install python with [chocolatey](https://community.chocolatey.org/packages/python312) or homebrew
- Note works with Python up to 3.12, issues with 3.13 as both librosa and matplotlib do not support it 
- Exit command prompt on windows with Ctrl Z and enter
- Locate interpreter on windows with ```python -c "import os, sys; print(os.path.dirname(sys.executable))"```

# Build

## Use a virtual environment 

Create one at the root of the repo using

```
python3 -m venv venv
```

## How to build and publish

## CLI Commands

Can't get this to work

```
[project.scripts]
sonosthesia-bake = "sonosthesia-audio-pipeline:analysis_entry_point"
sonosthesia-preview = "sonosthesia-audio-pipeline:preview_entry_point"
```

To get termcolor colors in pycharm see [here](https://stackoverflow.com/questions/76764301/what-should-i-do-to-make-termcolor-work-in-pycharm)

## Dependencies

- matplotlib >=3.9
- demucs >=3.8.0
- msgpack >=3.8.0
- librosa >=3.7.0

# Analysis

Under consideration for deeper analysis 

## Essentia

### Algorithms 

https://essentia.upf.edu/reference/std_RhythmDescriptors.html

https://essentia.upf.edu/reference/std_SuperFluxExtractor.html

https://essentia.upf.edu/reference/std_TempoTap.html

https://essentia.upf.edu/reference/std_PitchContours.html

https://essentia.upf.edu/reference/std_PitchContoursMultiMelody.html

https://essentia.upf.edu/reference/std_LowLevelSpectralExtractor.html

LowLevelSpectralExtractor is very thorough

### Extractors

https://essentia.upf.edu/freesound_extractor.html

https://essentia.upf.edu/streaming_extractor_music.html#music-extractor