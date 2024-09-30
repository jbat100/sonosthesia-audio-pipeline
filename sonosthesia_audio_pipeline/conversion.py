import argparse
import os.path

from colorama import just_fix_windows_console

from sonosthesia_audio_pipeline.utils import (MSGPACK_ANALYSIS_EXTENSION, JSON_ANALYSIS_EXTENSION, input_to_filepaths, change_extension,
                   read_json_with_header, read_packed_with_header, write_json_with_header, write_packed_with_header)

CONVERSION_DESCRIPTION = "Convert between json and msgpack analysis files"

just_fix_windows_console()


def run_conversion(file_path):
    file_path_without_extension, extension = os.path.splitext(file_path)
    if extension == JSON_ANALYSIS_EXTENSION:
        header, data = read_json_with_header(file_path)
        converted_file_path = change_extension(file_path, MSGPACK_ANALYSIS_EXTENSION)
        write_packed_with_header(data, header, converted_file_path)
    elif extension == MSGPACK_ANALYSIS_EXTENSION:
        header, data = read_packed_with_header(file_path)
        converted_file_path = change_extension(file_path, JSON_ANALYSIS_EXTENSION)
        write_json_with_header(data, header, converted_file_path)


def configure_conversion_parser(parser):
    parser.add_argument('-i', '--input', type=str, nargs='?', required=True,
                        help=f'path to file ({MSGPACK_ANALYSIS_EXTENSION} or {JSON_ANALYSIS_EXTENSION}) or directory')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='recurse through input if directory')


def conversion_with_args(args):
    extensions = [MSGPACK_ANALYSIS_EXTENSION, JSON_ANALYSIS_EXTENSION]
    file_paths = input_to_filepaths(args.input, extensions, args.recursive)
    for file_path in file_paths:
        run_conversion(file_path)
    print('Done')


def conversion():
    parser = argparse.ArgumentParser(description=CONVERSION_DESCRIPTION)
    configure_conversion_parser(parser)
    args = parser.parse_args()
    conversion_with_args(args)


if __name__ == "__main__":
    conversion()
