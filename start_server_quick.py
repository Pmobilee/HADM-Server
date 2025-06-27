#!/usr/bin/env python3
"""
HADM Server Quick Launcher
Skips all dependency checks and starts the FastAPI server directly.
"""

import os
import sys
import time
import argparse


def log_time(message):
    """Log message with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def start_server(lazy_mode=False):
    """Start the FastAPI server"""
    log_time("Starting FastAPI server...")
    print("\n" + "=" * 50)
    print("Starting HADM FastAPI Server...")
    if lazy_mode:
        print("LAZY MODE: Models and heavy imports will load on first request")
    print("Server will be available at: http://localhost:8080")
    print("Web Interface: http://localhost:8080/interface")
    print("API documentation: http://localhost:8080/docs")
    print("=" * 50)

    try:
        # Start the server with lazy flag if specified
        if lazy_mode:
            log_time("Executing: python api.py --lazy")
            os.system("python api.py --lazy")
        else:
            log_time("Executing: python api.py")
            os.system("python api.py")
    except KeyboardInterrupt:
        log_time("Server stopped by user.")
        print("\nServer stopped by user.")
    except Exception as e:
        log_time(f"Error starting server: {e}")
        print(f"Error starting server: {e}")


def main():
    parser = argparse.ArgumentParser(description="HADM Server Quick Launcher")
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Enable lazy loading - models and heavy imports load on first request only",
    )
    args = parser.parse_args()

    log_time("HADM Server Quick Launcher started")
    if args.lazy:
        log_time("Lazy mode enabled - faster startup, slower first request")
    start_server(lazy_mode=args.lazy)


if __name__ == "__main__":
    main()
