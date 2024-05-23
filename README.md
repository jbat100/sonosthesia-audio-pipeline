# sonosthesia-audio-pipeline

Python (Librosa) based tooling to analyse audio files and write results to file for use in realtime visualization apps. Results can be written using Message Pack or raw binary float arrays for efficient (de)serialization. Readers are provided for the Unity timeline, to be used alongside the original audio files.

## Python Pipeline



### Notes

- Pipeline is a PyCharm project
- Easiest to install python with [chocolatey](https://community.chocolatey.org/packages/python312) or homebrew
- Note works with Python up to 3.12, issues with 3.13 as both librosa and matplotlib do not support it 
- Exit command prompt on windows with Ctrl Z and enter
- Locate interpreter on windows with ```python -c "import os, sys; print(os.path.dirname(sys.executable))"```

## Unity Timeline 
