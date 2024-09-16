import argparse
import os
from collections import namedtuple


def parse_configuration():
    """
    Parse command-line arguments using argparse.

    Returns:
        Configuration: Namedtuple containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Process files in a directory.')

    parser.add_argument('-i', '--input', type=str, nargs='?', default='audio/kepler.mp3',
                        help='Path to the file or directory')

    parser.add_argument('-r', '--raw', action='store_true', help='Output in raw format')

    args = parser.parse_args()


def input_to_filepaths(input_path):
    # If the specified path is a directory, get absolute file paths of all files in the directory
    if os.path.isdir(input_path):
        file_paths = [os.path.abspath(os.path.join(input_path, file)) for file in os.listdir(input_path)
                      if file.lower().endswith(('.mp3', '.wav'))]
    else:
        file_paths = [os.path.abspath(input_path)]
    return file_paths
