from __future__ import annotations

import argparse
import sys

from .validation import validate_environment


def main() -> int:
    parser = argparse.ArgumentParser(prog="openenv")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate an OpenEnv environment")
    validate_parser.add_argument("path", help="Path containing openenv.yaml")

    args = parser.parse_args()
    if args.command == "validate":
        issues = validate_environment(args.path)
        if issues:
            for issue in issues:
                print(f"FAIL: {issue}")
            return 1
        print("PASS: environment is OpenEnv compliant")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
