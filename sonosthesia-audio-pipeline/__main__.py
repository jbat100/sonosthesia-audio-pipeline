import argparse

from analysis import configure_analysis_parser, ANALYSIS_DESCRIPTION
from separation import configure_separation_parser, SEPARATION_DESCRIPTION
from pipeline import configure_pipeline_parser, PIPELINE_DESCRIPTION
from inspection import configure_inspection_parser, INSPECTION_DESCRIPTION
from conversion import configure_conversion_parser, CONVERSION_DESCRIPTION

def main():
    parser = argparse.ArgumentParser(description='Tool for baking audio analysis data')
    subparsers = parser.add_subparsers(help='sub-command help', dest='subcommand')

    configure_analysis_parser(subparsers.add_parser('analysis', description=ANALYSIS_DESCRIPTION))
    configure_separation_parser(subparsers.add_parser('separation', description=SEPARATION_DESCRIPTION))
    configure_pipeline_parser(subparsers.add_parser('pipeline', description=PIPELINE_DESCRIPTION))
    configure_inspection_parser(subparsers.add_parser('inspection', description=INSPECTION_DESCRIPTION))
    configure_conversion_parser(subparsers.add_parser('conversion', description=CONVERSION_DESCRIPTION))

    args = parser.parse_args()
    print(args)


if __name__ == '__main__':
    main()
