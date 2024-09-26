import argparse
import collections
import os

from colorama import just_fix_windows_console
from termcolor import colored

from sonosthesia_audio_pipeline.separation import separation_with_args, separated_output_paths
from sonosthesia_audio_pipeline.analysis import analysis_with_args, input_to_filepaths, AUDIO_EXTENSIONS

PIPELINE_DESCRIPTION = 'Chain source separation and analyse original audio as well as separated sources'

SeparationArgs = collections.namedtuple('SeparationArgs', ['input', 'model'])
AnalysisArgs = collections.namedtuple('AnalysisArgs', ['input', 'start', 'duration', 'json'])

just_fix_windows_console()


def configure_pipeline_parser(parser):
    parser.add_argument('-i', '--input', type=str, nargs='?', required=True,
                        help='path to the audio file or directory')
    parser.add_argument('-n', '--model', type=str, default='mdx_extra',
                        help='demucs model used for the separation')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='overwrite existing separated files')
    parser.add_argument('-j', '--json', action='store_true',
                        help='write output to json')


def pipeline_with_args(args):
    input_paths = input_to_filepaths(args.input, AUDIO_EXTENSIONS)
    separated_paths = []
    # Perform separation on input audio files and store separated paths
    for input_path in input_paths:
        if not args.overwrite:
            separated_check_paths = separated_output_paths(input_path, args.model)
            if all(os.path.isfile(file) for file in separated_check_paths):
                separated_paths.extend(separated_check_paths)
                description = "\n".join(separated_paths)
                print(colored(f'Using existing separated files: \n{description}', "green"))
                continue
        separated_paths.extend(separation_with_args(SeparationArgs(args.input, args.model)))
    # Perform analysis on both input and separated audio files
    audio_paths = input_paths + separated_paths
    for audio_path in audio_paths:
        analysis_with_args(AnalysisArgs(audio_path, 0.0, None, args.json))


def pipeline():
    parser = argparse.ArgumentParser(description=PIPELINE_DESCRIPTION)
    configure_pipeline_parser(parser)
    args = parser.parse_args()
    pipeline_with_args(args)


if __name__ == "__main__":
    pipeline()