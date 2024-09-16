import argparse
import os.path
import demucs.separate
import shlex
from setup import input_to_filepaths


def run_separation(audio_file):
    demucs.separate.main(shlex.split(f'--mp3 -n mdx_extra "{audio_file}"'))


def run_separation_custom(audio_file):
    directory = os.path.dirname(audio_file)
    file_name = os.path.basename(audio_file)
    name, ext = os.path.splitext(file_name)
    pattern = "{track}/{track}_{stem}.{ext}"
    command = f'--mp3 -n mdx_extra --filename {pattern} -o "{directory}" "{audio_file}"'
    print("Running separation : " + command)
    demucs.separate.main(shlex.split(command))


def separation_parser():
    parser = argparse.ArgumentParser(description='Separate tracks.')
    parser.add_argument('-i', '--input', type=str, nargs='?', required=True,
                        help='path to the file or directory')
    return parser


def separation():
    parser = separation_parser()
    args = parser.parse_args()
    file_paths = input_to_filepaths(args.input)
    for audio_file in file_paths:
        run_separation_custom(audio_file)
    print('Done')


if __name__ == "__main__":
    separation()