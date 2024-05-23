from collections import namedtuple

# Trying to pull out some common behaviour between analysis and preview but ends up being more of a mess

AnalysisData = namedtuple('AnalysisData', ['time', 'rms', 'lows', 'mids', 'highs', 'centroid', 'onset'])

def process_audio(y, sr):
    pass