import argparse

from analysis import configure_analysis_parser
from separation import configure_separation_parser
from pipeline import configure_pipeline_parser
from inspection import configure_inspection_parser


def main():
    parser = argparse.ArgumentParser(description='Tool for baking audio analysis data')
    subparsers = parser.add_subparsers(help='sub-command help', dest='subcommand')

    configure_analysis_parser(subparsers.add_parser('analysis', help='analysis help'))
    configure_separation_parser(subparsers.add_parser('separation', help='separation help'))
    configure_pipeline_parser(subparsers.add_parser('pipeline', help='pipeline help'))
    configure_inspection_parser(subparsers.add_parser('inspection', help='inspection help'))

    args = parser.parse_args()
    print(args)


if __name__ == '__main__':
    main()
