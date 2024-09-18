import argparse

from analysis2 import configure_analysis2_parser


def main():
    parser = argparse.ArgumentParser(description='Tool for baking audio analysis data')
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_analysis = subparsers.add_parser('analysis', help='analysis help')
    configure_analysis2_parser(parser_analysis)

    parser_separation = subparsers.add_parser('analysis', help='analysis help')




if __name__ == '__main__':
    main()