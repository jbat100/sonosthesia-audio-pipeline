import argparse
import os.path
import json

from jsonschema import validate

from colorama import just_fix_windows_console
from termcolor import colored

from sonosthesia_audio_pipeline.utils import (JSON_ANALYSIS_EXTENSION, MSGPACK_ANALYSIS_EXTENSION,
                   input_to_filepaths, read_packed_with_header)

CHECK_DESCRIPTION = "Check an analysis file against a schema"

just_fix_windows_console()


def run_check(file_path, schema):
    print(colored(f'Running check against {file_path}', "blue"))
    _, extension = os.path.splitext(file_path)
    if extension == JSON_ANALYSIS_EXTENSION:
        with open(file_path, 'r') as file:
            data = json.load(file)
            validate(data, schema)
    elif extension == MSGPACK_ANALYSIS_EXTENSION:
        header, data = read_packed_with_header(file_path)
        validate(data, schema)
    print(f'Passed {file_path}')


def configure_check_parser(parser):
    parser.add_argument('-i', '--input', type=str, nargs='?', required=True,
                        help=f'path to file ({MSGPACK_ANALYSIS_EXTENSION} or {JSON_ANALYSIS_EXTENSION}) or directory')
    parser.add_argument('-s', '--schema', type=str, required=True,
                        help=f'path to a json schema file')


def check_with_args(args):
    with open(args.schema, 'r') as file:
        schema = json.load(file)
    print(colored(f'Loaded schema from {args.schema}', "green"))
    file_paths = input_to_filepaths(args.input, [MSGPACK_ANALYSIS_EXTENSION, JSON_ANALYSIS_EXTENSION])
    for file_path in file_paths:
        run_check(file_path, schema)
    print('Done')


def check():
    parser = argparse.ArgumentParser(description=CHECK_DESCRIPTION)
    configure_check_parser(parser)
    args = parser.parse_args()
    check_with_args(args)


if __name__ == "__main__":
    check()
