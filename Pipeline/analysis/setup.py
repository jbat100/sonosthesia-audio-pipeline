import argparse
import os
from collections import namedtuple

Configuration = namedtuple('Configuration', ['file_paths'])


def parse_configuration():
    """
    Parse command-line arguments using argparse.

    Returns:
        Configuration: Namedtuple containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Process files in a directory.')

    parser.add_argument('-i', '--input', type=str, nargs='?', default='../audio/kepler STEMS DRUMS.mp3',
                        help='Path to the file or directory')

    args = parser.parse_args()

    # If the specified path is a directory, get absolute file paths of all files in the directory
    if os.path.isdir(args.input):
        file_paths = [os.path.abspath(os.path.join(args.input, file)) for file in os.listdir(args.input)
                      if file.lower().endswith(('.mp3', '.wav'))]
    else:
        file_paths = [os.path.abspath(args.input)]

    return Configuration(file_paths=file_paths)
