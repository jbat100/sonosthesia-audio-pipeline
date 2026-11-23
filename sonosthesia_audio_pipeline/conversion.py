import argparse
import os.path

from colorama import just_fix_windows_console
from termcolor import colored

from sonosthesia_audio_pipeline.utils import (
    MSGPACK_ANALYSIS_EXTENSION,
    JSON_ANALYSIS_EXTENSION,
    STREAMING_ANALYSIS_EXTENSION,
    input_to_filepaths,
    change_extension,
    read_json_with_header,
    read_packed_with_header,
    write_json_with_header,
    write_packed_with_header,
)
from sonosthesia_audio_pipeline.streaming import (
    read_streaming_with_header,
    write_streaming_with_header,
)

ANALYSIS_EXTENSIONS = [
    MSGPACK_ANALYSIS_EXTENSION,
    JSON_ANALYSIS_EXTENSION,
    STREAMING_ANALYSIS_EXTENSION,
]

READERS = {
    MSGPACK_ANALYSIS_EXTENSION: read_packed_with_header,
    JSON_ANALYSIS_EXTENSION: read_json_with_header,
    STREAMING_ANALYSIS_EXTENSION: read_streaming_with_header,
}

CONVERSION_DESCRIPTION = "Convert between XAS, JSON, and MessagePack analysis files"

just_fix_windows_console()


def _default_output_extension(input_extension):
    if input_extension == MSGPACK_ANALYSIS_EXTENSION:
        return JSON_ANALYSIS_EXTENSION
    if input_extension == JSON_ANALYSIS_EXTENSION:
        return MSGPACK_ANALYSIS_EXTENSION
    return JSON_ANALYSIS_EXTENSION


def run_conversion(file_path, output_format=None, sample_rate=None):
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    reader = READERS.get(extension)
    if reader is None:
        raise ValueError(f'Unsupported analysis input extension: {extension}')

    target_extension = output_format or _default_output_extension(extension)
    if target_extension not in ANALYSIS_EXTENSIONS:
        raise ValueError(f'Unsupported analysis output extension: {target_extension}')
    if target_extension == extension:
        print(colored(f'Skipping {file_path}: source and destination formats match.', 'yellow'))
        return

    header, data = reader(file_path)
    converted_file_path = change_extension(file_path, target_extension)
    if target_extension == JSON_ANALYSIS_EXTENSION:
        write_json_with_header(data, header, converted_file_path)
    elif target_extension == MSGPACK_ANALYSIS_EXTENSION:
        write_packed_with_header(data, header, converted_file_path)
    else:
        write_streaming_with_header(
            data, header, converted_file_path, sample_rate=sample_rate
        )


def configure_conversion_parser(parser):
    parser.add_argument('-i', '--input', type=str, nargs='?', required=True,
                        help=f'path to an analysis file '
                             f'({", ".join(ANALYSIS_EXTENSIONS)}) or directory')
    parser.add_argument('-o', '--output-format',
                        choices=ANALYSIS_EXTENSIONS,
                        help='destination format; defaults preserve the legacy JSON/XAA '
                             'toggle, or convert XAS to JSON')
    parser.add_argument('-s', '--sample-rate', type=int, default=None,
                        help='audio sample rate used when writing XAS; inferred from '
                             'continuous sample timing when omitted')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='recurse through input if directory')


def conversion_with_args(args):
    file_paths = input_to_filepaths(args.input, ANALYSIS_EXTENSIONS, args.recursive)
    for file_path in file_paths:
        run_conversion(
            file_path,
            getattr(args, 'output_format', None),
            getattr(args, 'sample_rate', None),
        )
    print('Done')


def conversion():
    parser = argparse.ArgumentParser(description=CONVERSION_DESCRIPTION)
    configure_conversion_parser(parser)
    args = parser.parse_args()
    conversion_with_args(args)


if __name__ == "__main__":
    conversion()
