#!/usr/bin/env python3
"""
HADM Server Launcher
Checks dependencies and starts the FastAPI server
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def log_time(message):
    """Log message with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def check_dependencies():
    """Check if all required dependencies are available"""
    log_time("Starting dependency check...")
    start_time = time.time()

    # Check if we're in virtual environment
    if not hasattr(sys, "real_prefix") and not (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        log_time("WARNING: Not running in a virtual environment!")
        print("Consider running: source venv/bin/activate")

    # Check required packages
    required_packages = ["torch", "detectron2", "fastapi", "uvicorn", "PIL", "numpy"]

    missing_packages = []
    for i, package in enumerate(required_packages):
        log_time(f"Checking package {i+1}/{len(required_packages)}: {package}")
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)

    duration = time.time() - start_time
    log_time(f"Dependency check completed in {duration:.2f}s")

    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install missing packages and try again.")
        return False

    return True


def check_models():
    """Check if model files exist"""
    log_time("Starting model file check...")
    start_time = time.time()

    model_dir = Path("pretrained_models")
    required_models = [
        "HADM-L_0249999.pth",
        "HADM-G_0249999.pth",
        "eva02_L_coco_det_sys_o365.pth",
    ]

    missing_models = []
    for i, model in enumerate(required_models):
        log_time(f"Checking model {i+1}/{len(required_models)}: {model}")
        model_path = model_dir / model
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✓ {model} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {model} - NOT FOUND")
            missing_models.append(model)

    duration = time.time() - start_time
    log_time(f"Model file check completed in {duration:.2f}s")

    if missing_models:
        print(f"\nMissing models: {', '.join(missing_models)}")
        print("Please run setup_environment.py to download models.")
        return False

    return True


def check_gpu():
    """Check GPU availability"""
    log_time("Starting GPU check...")
    start_time = time.time()

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✓ GPU available: {gpu_name}")
            print(f"✓ GPU memory: {gpu_memory:.1f} GB")
            duration = time.time() - start_time
            log_time(f"GPU check completed in {duration:.2f}s")
            return True
        else:
            print("✗ No GPU available - will use CPU (much slower)")
            duration = time.time() - start_time
            log_time(f"GPU check completed in {duration:.2f}s")
            return True  # Still allow CPU usage
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        duration = time.time() - start_time
        log_time(f"GPU check completed in {duration:.2f}s")
        return True


def start_server():
    """Start the FastAPI server"""
    log_time("Starting FastAPI server...")
    print("\n" + "=" * 50)
    print("Starting HADM FastAPI Server...")
    print("Server will be available at: http://localhost:8080")
    print("API documentation: http://localhost:8080/docs")
    print("=" * 50)

    try:
        # Start the server
        log_time("Executing: python api.py")
        os.system("python api.py")
    except KeyboardInterrupt:
        log_time("Server stopped by user.")
        print("\nServer stopped by user.")
    except Exception as e:
        log_time(f"Error starting server: {e}")
        print(f"Error starting server: {e}")


def main():
    overall_start = time.time()
    log_time("HADM Server Launcher started")
    print("HADM Server Launcher")
    print("=" * 30)

    # Check all prerequisites
    if not check_dependencies():
        log_time("Dependency check failed - exiting")
        sys.exit(1)

    if not check_models():
        log_time("Model check failed - exiting")
        sys.exit(1)

    check_gpu()

    overall_duration = time.time() - overall_start
    log_time(f"All checks completed in {overall_duration:.2f}s")

    # Start server
    start_server()


if __name__ == "__main__":
    main()
