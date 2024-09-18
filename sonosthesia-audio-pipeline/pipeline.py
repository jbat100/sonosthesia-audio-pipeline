import argparse
import collections

from separation import separation_with_args
from analysis import analysis_with_args, input_to_filepaths, AUDIO_EXTENSIONS

PIPELINE_DESCRIPTION = 'Chain source separation and analyse original audio as well as separated sources'

SeparationArgs = collections.namedtuple('SeparationArgs', ['input', 'model'])
AnalysisArgs = collections.namedtuple('AnalysisArgs', ['input', 'start', 'duration'])


def configure_pipeline_parser(parser):
    parser.add_argument('-i', '--input', type=str, nargs='?', required=True,
                        help='path to the audio file or directory')
    parser.add_argument('-n', '--model', type=str, default='mdx_extra',
                        help='demucs model used for the separation')


def pipeline_with_args(args):
    input_paths = input_to_filepaths(args.input, AUDIO_EXTENSIONS)
    separated_paths = []
    separated_paths = separation_with_args(SeparationArgs(args.input, args.model))



def pipeline():
    parser = argparse.ArgumentParser(description=PIPELINE_DESCRIPTION)
    configure_pipeline_parser(parser)
    args = parser.parse_args()
    pipeline_with_args(args)


if __name__ == "__main__":
    pipeline()