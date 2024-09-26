import argparse

from colorama import just_fix_windows_console
from termcolor import colored

from sonosthesia_audio_pipeline.analysis import configure_analysis_parser, analysis_with_args, ANALYSIS_DESCRIPTION
from sonosthesia_audio_pipeline.separation import configure_separation_parser, separation_with_args, SEPARATION_DESCRIPTION
from sonosthesia_audio_pipeline.pipeline import configure_pipeline_parser, pipeline_with_args, PIPELINE_DESCRIPTION
from sonosthesia_audio_pipeline.inspection import configure_inspection_parser, inspection_with_args, INSPECTION_DESCRIPTION
from sonosthesia_audio_pipeline.conversion import configure_conversion_parser, conversion_with_args, CONVERSION_DESCRIPTION

IMPLEMENTATIONS = {
    'analysis' : analysis_with_args,
    'separation' : separation_with_args,
    'pipeline' : pipeline_with_args,
    'inspection' : inspection_with_args,
    'conversion' : conversion_with_args
}

just_fix_windows_console()


def main():
    parser = argparse.ArgumentParser(description='Tool for baking audio analysis data')
    subparsers = parser.add_subparsers(help='sub-command help', dest='subcommand')

    configure_analysis_parser(subparsers.add_parser('analysis', description=ANALYSIS_DESCRIPTION))
    configure_separation_parser(subparsers.add_parser('separation', description=SEPARATION_DESCRIPTION))
    configure_pipeline_parser(subparsers.add_parser('pipeline', description=PIPELINE_DESCRIPTION))
    configure_inspection_parser(subparsers.add_parser('inspection', description=INSPECTION_DESCRIPTION))
    configure_conversion_parser(subparsers.add_parser('conversion', description=CONVERSION_DESCRIPTION))

    args = parser.parse_args()
    subcommand = args.subcommand

    subcommand_description = ", ".join(IMPLEMENTATIONS.keys())

    if not isinstance(subcommand, str) or not bool(subcommand.strip()):
        print(colored(f'Unspecified subcommand, expected {subcommand_description}'))
        return

    implementation = IMPLEMENTATIONS.get(subcommand)

    if implementation is None:
        print(colored(f'Unknown subcommand : {args.subcommand} expected {subcommand_description}'))
        return

    implementation(args)


if __name__ == '__main__':
    main()
