#!/usr/bin/env python3
# Example:
# python run_deephrd.py --model /path/to/models --projectPath /path/to/projects \
#   --project MyProject --output /path/to/output --metadata /path/to/metadata.csv \
#   --preprocess --stainNorm --generateDataSets --predict5x --pullROIs \
#   --predict20x --reportVerbose
import argparse
import os
import subprocess
import sys

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepHRD prediction pipeline with configurable steps."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to pretrained BRCA FFPE models directory.",
    )
    parser.add_argument(
        "--projectPath",
        required=True,
        help="Base path containing the project folder.",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Project folder name; slides live in PROJECT_PATH / PROJECT.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for DeepHRD results.",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Metadata CSV path used by DeepHRD.",
    )
    parser.add_argument(
        "--generateMetadata",
        action="store_true",
        help="Generate metadata TSV before running prediction.",
    )
    parser.add_argument(
        "--slideDir",
        help="Slide directory used when generating metadata.",
    )
    parser.add_argument(
        "--metadataLabel",
        type=float,
        default=0.0,
        help="Metadata placeholder label value (default: 0.0).",
    )
    parser.add_argument(
        "--metadataSoftLabel",
        type=float,
        default=0.0,
        help="Metadata placeholder softLabel value (default: 0.0).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Worker processes to use (default: 32).",
    )
    parser.add_argument(
        "--BN_reps",
        type=int,
        default=10,
        help="BatchNorm repetitions; increase for final runs (default: 10).",
    )
    parser.add_argument(
        "--max_gpu",
        type=int,
        default=0,
        help="Max GPUs for DeepHRD (default: 0 - uses all available).",
    )
    parser.add_argument(
        "--max_cpu",
        type=int,
        default=0,
        help="Max CPUs for DeepHRD (default: 0 - uses all available).",
    )

    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run preprocessing.",
    )
    parser.add_argument(
        "--stainNorm",
        action="store_true",
        help="Run stain normalization.",
    )
    parser.add_argument(
        "--generateDataSets",
        action="store_true",
        help="Generate datasets.",
    )
    parser.add_argument(
        "--predict5x",
        action="store_true",
        help="Run 5x prediction.",
    )
    parser.add_argument(
        "--pullROIs",
        action="store_true",
        help="Pull ROIs (enable after first successful run).",
    )
    parser.add_argument(
        "--predict20x",
        action="store_true",
        help="Run 20x prediction (enable after ROIs work).",
    )
    parser.add_argument(
        "--predictionMasks",
        action="store_true",
        help="Generate prediction masks.",
    )
    parser.add_argument(
        "--reportVerbose",
        action="store_true",
        help="Enable verbose reporting from DeepHRD.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # =========================
    # SAFETY CHECKS
    # =========================

    required_paths = {
        "DEEPHRD_MODELS": args.model,
        "PROJECT_PATH": args.projectPath,
    }

    for name, path in required_paths.items():
        if not os.path.isdir(path):
            sys.exit(f"[ERROR] {name} does not exist or is not a directory: {path}")

    if not os.path.isdir(os.path.join(args.projectPath, args.project)):
        sys.exit(
            f"[ERROR] Slide directory not found: "
            f"{os.path.join(args.projectPath, args.project)}"
        )

    if args.generateMetadata:
        if not args.slideDir:
            sys.exit("[ERROR] --slideDir is required with --generateMetadata.")
        if not os.path.isdir(args.slideDir):
            sys.exit(f"[ERROR] Slide directory not found: {args.slideDir}")
        metadata_parent = os.path.dirname(args.metadata)
        if metadata_parent:
            os.makedirs(metadata_parent, exist_ok=True)

        metadata_cmd = [
            sys.executable,
            "-m",
            "deephrd.generate_metadata",
            "--slideDir", args.slideDir,
            "--output", args.metadata,
            "--label", str(args.metadataLabel),
            "--softLabel", str(args.metadataSoftLabel),
        ]
        print("\n[DeepHRD] Generating metadata:\n")
        print(" ".join(metadata_cmd))
        print("\n----------------------------------------\n")
        result = subprocess.run(metadata_cmd, env=os.environ.copy())
        if result.returncode != 0:
            sys.exit(f"[ERROR] Metadata generation failed with code {result.returncode}")

    # =========================
    # BUILD COMMAND
    # =========================

    cmd = [
        sys.executable,
        "-m",
        "deephrd.DeepHRD_predict",
        "--projectPath", args.projectPath,
        "--project", args.project,
        "--metadata", args.metadata,
        "--output", args.output,
        "--model", args.model,
        "--workers", str(args.workers),
        "--BN_reps", str(args.BN_reps),
        "--max_gpu", str(args.max_gpu),
        "--max_cpu", str(args.max_cpu),
    ]

    if args.reportVerbose:
        cmd.append("--reportVerbose")

    if args.preprocess:
        cmd.append("--preprocess")

    if args.stainNorm:
        cmd.append("--stainNorm")

    if args.generateDataSets:
        cmd.append("--generateDataSets")

    if args.predict5x:
        cmd.append("--predict5x")

    if args.pullROIs:
        cmd.append("--pullROIs")

    if args.predict20x:
        cmd.append("--predict20x")

    if args.predictionMasks:
        cmd.append("--predictionMasks")

    # =========================
    # EXECUTION
    # =========================

    print("\n[DeepHRD] Running command:\n")
    print(" ".join(cmd))
    print("\n----------------------------------------\n")

    env = os.environ.copy()

    # HPC safety: avoid thread oversubscription
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        sys.exit(f"[ERROR] DeepHRD exited with code {result.returncode}")

    print("\n[DeepHRD] Completed successfully.")


if __name__ == "__main__":
    main()
