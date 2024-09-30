import argparse
import os.path
import demucs.separate
import shlex
import ssl

from colorama import just_fix_windows_console
from termcolor import colored

from sonosthesia_audio_pipeline.utils import input_to_filepaths, AUDIO_EXTENSIONS

SEPARATION_DESCRIPTION = 'Audio separation using demucs.'

just_fix_windows_console()


def run_separation(audio_file, model):
    # trying to address certificate error https://github.com/python-poetry/poetry/issues/5117
    ssl._create_default_https_context = ssl._create_unverified_context
    demucs.separate.main(shlex.split(f'--mp3 -n {model} "{audio_file}"'))


def get_output_directory(audio_file):
    audio_file_directory = os.path.dirname(audio_file)
    file_name = os.path.basename(audio_file)
    name, ext = os.path.splitext(file_name)
    return os.path.join(audio_file_directory, name)


def run_separation_custom(audio_file, model):
    output_directory = get_output_directory(audio_file)
    pattern = "{track}_{stem}.{ext}"
    command = f'--mp3 -n {model} --filename {pattern} -o "{output_directory}" "{audio_file}"'
    print(colored("Running separation command : " + command, "blue"))
    demucs.separate.main(shlex.split(command))


# should be a way to extract that from the command output but there's a lot of stuff there...
def separated_output_paths(audio_file, model):
    file_name = os.path.basename(audio_file)
    name, ext = os.path.splitext(file_name)
    separated_output_directory = os.path.join(str(get_output_directory(audio_file)), model)
    output_paths = []
    for stem in ['bass', 'drums', 'vocals', 'other']:
        output_name = f'{name}_{stem}{ext}'
        output_path = os.path.join(str(separated_output_directory), output_name)
        output_paths.append(output_path)
    return output_paths


def configure_separation_parser(parser):
    parser.add_argument('-i', '--input', type=str, nargs='?', required=True,
                        help='path to the audio file or directory')
    parser.add_argument('-n', '--model', type=str, default='mdx_extra',
                        help='demucs model used for the separation')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='recurse through input if directory')


def separation_with_args(args):
    file_paths = input_to_filepaths(args.input, AUDIO_EXTENSIONS, args.recursive)
    output_paths = []
    for audio_file in file_paths:
        output_paths.extend(separated_output_paths(audio_file, args.model))
        run_separation_custom(audio_file, args.model)
    print(colored('Separated files \n' + '\n'.join(output_paths), "green"))
    return output_paths


def separation():
    parser = argparse.ArgumentParser(description=SEPARATION_DESCRIPTION)
    configure_separation_parser(parser)
    args = parser.parse_args()
    separation_with_args(args)


if __name__ == "__main__":
    separation()
