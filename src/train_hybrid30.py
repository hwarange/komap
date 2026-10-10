"""Dedicated entry point: process-local hooks leave all running sources intact."""
import argparse
from hybrid30 import install

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--arch')
    args, _ = parser.parse_known_args()
    install(args.arch).main()
