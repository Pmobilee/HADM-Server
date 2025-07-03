#!/usr/bin/env python3
"""
Simple HADM Demo Runner
Uses the existing demo.py to run HADM inference on images
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run HADM demo inference")
    parser.add_argument("--input", required=True,
                        help="Input image path or directory")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--model", choices=["local", "global"], default="local",
                        help="Model to use (local=HADM-L, global=HADM-G)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold")

    args = parser.parse_args()

    # Check if we're in the right directory
    if not Path("demo").exists():
        print("Error: demo directory not found. Please run this script from the root of the HADM project.")
        sys.exit(1)

    # Check if hadm_demo.py exists
    if not Path("hadm_demo.py").exists():
        print("Error: hadm_demo.py not found. Please run this script from the root of the HADM project.")
        sys.exit(1)

    # Build command
    cmd = [
        "python3", "hadm_demo.py",
        "--input", args.input,
        "--model", args.model,
        "--confidence", str(args.confidence)
    ]

    if args.output:
        cmd.extend(["--output", args.output])

    print(f"Running HADM-{'L' if args.model == 'local' else 'G'} demo...")
    print(f"Input: {args.input}")
    if args.output:
        print(f"Output: {args.output}")
    print("=" * 50)

    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print("Demo completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Demo failed with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
