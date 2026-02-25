#!/usr/bin/env python3
import argparse
import subprocess
import sys


COMMANDS = {
    "predict": {
        "module": "deephrd.DeepHRD_predict",
        "help": "Run prediction pipeline.",
    },
    "test": {
        "module": "deephrd.DeepHRD_predict",
        "help": "Alias for predict.",
    },
    "train": {
        "module": "deephrd.DeepHRD_train",
        "help": "Run training pipeline.",
    },
    "generate_metadata": {
        "module": "deephrd.generate_metadata",
        "help": "Generate metadata TSV from slides.",
    },
}


def run_module(module: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *args]
    result = subprocess.run(cmd)
    raise SystemExit(result.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deephrd",
        description="DeepHRD command-line interface.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    for name, meta in COMMANDS.items():
        subparser = subparsers.add_parser(
            name,
            help=meta["help"],
            add_help=False,
        )
        subparser.set_defaults(module=meta["module"])

    return parser


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] in COMMANDS:
        if any(flag in sys.argv[2:] for flag in ("-h", "--help")):
            run_module(COMMANDS[sys.argv[1]]["module"], ["-h"])
            return

    parser = build_parser()
    args, unknown = parser.parse_known_args()

    if not args.command:
        if unknown:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        parser.print_help()
        raise SystemExit(0)

    run_module(args.module, unknown)


if __name__ == "__main__":
    main()
