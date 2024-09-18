import argparse

from analysis import configure_analysis_parser
from separation import configure_separation_parser
from pipeline import configure_pipeline_parser


def main():
    parser = argparse.ArgumentParser(description='Tool for baking audio analysis data')
    subparsers = parser.add_subparsers(help='sub-command help', dest='subcommand')

    parser_analysis = subparsers.add_parser('analysis', help='analysis help')
    configure_analysis_parser(parser_analysis)

    parser_separation = subparsers.add_parser('separation', help='separation help')
    configure_separation_parser(parser_separation)

    parser_pipeline = subparsers.add_parser('pipeline', help='pipeline help')
    configure_pipeline_parser(parser_pipeline)

    args = parser.parse_args()
    print(args)


if __name__ == '__main__':
    main()
