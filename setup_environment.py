#!/usr/bin/env python3
"""
HADM Environment Setup Script
Optimizes the environment for maximum performance on GPU systems
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def log_step(message):
    """Log setup steps with formatting"""
    print(f"[SETUP] {message}")


def set_performance_environment_variables():
    """Set environment variables for optimal performance"""
    log_step("Setting performance environment variables...")

    # PyTorch optimizations
    env_vars = {
        # Cache directories on fast storage
        "TORCH_HOME": "/tmp/torch_cache",
        "HF_HOME": "/tmp/hf_cache",
        "TRANSFORMERS_CACHE": "/tmp/transformers_cache",
        "DETECTRON2_CACHE": "/tmp/detectron2_cache",
        # CUDA optimizations
        "CUDA_LAUNCH_BLOCKING": "0",  # Async CUDA operations
        "CUDA_CACHE_DISABLE": "0",  # Enable CUDA caching
        "TORCH_CUDNN_V8_API_ENABLED": "1",  # Use optimized cuDNN
        # PyTorch compilation
        "TORCH_COMPILE_DEBUG": "0",  # Disable debug for speed
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/torch_inductor_cache",
        # Memory optimizations
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True",
        # Disable unnecessary features for speed
        "TOKENIZERS_PARALLELISM": "false",  # Avoid threading conflicts
        "OMP_NUM_THREADS": "1",  # Single-threaded for inference
        "MKL_NUM_THREADS": "1",
        # Network optimizations
        "CURL_CA_BUNDLE": "",  # Skip SSL verification for model downloads
        "REQUESTS_CA_BUNDLE": "",
    }

    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  {key}={value}")

    # Create cache directories
    cache_dirs = [
        "/tmp/torch_cache",
        "/tmp/hf_cache",
        "/tmp/transformers_cache",
        "/tmp/detectron2_cache",
        "/tmp/torch_inductor_cache",
    ]

    for cache_dir in cache_dirs:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"  Created cache directory: {cache_dir}")


def optimize_python_path():
    """Optimize Python import paths"""
    log_step("Optimizing Python import paths...")

    # Add current directory to Python path for faster local imports
    current_dir = str(Path.cwd())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"  Added to Python path: {current_dir}")


def check_system_requirements():
    """Check system requirements and provide recommendations"""
    log_step("Checking system requirements...")

    # Check GPU
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✓ NVIDIA GPU detected")
            # Extract GPU info
            lines = result.stdout.split("\n")
            for line in lines:
                if (
                    "RTX" in line
                    or "GeForce" in line
                    or "Tesla" in line
                    or "Quadro" in line
                ):
                    print(f"  GPU: {line.strip()}")
                    break
        else:
            print("  ⚠ NVIDIA GPU not detected or nvidia-smi not available")
    except FileNotFoundError:
        print("  ⚠ nvidia-smi not found")

    # Check memory
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_kb = int(line.split()[1])
                    mem_gb = mem_kb / 1024 / 1024
                    print(f"  System RAM: {mem_gb:.1f} GB")
                    if mem_gb < 16:
                        print("  ⚠ Less than 16GB RAM - may cause performance issues")
                    break
    except:
        print("  ⚠ Could not read memory information")

    # Check storage
    current_path = Path.cwd()
    try:
        stat = shutil.disk_usage(current_path)
        free_gb = stat.free / (1024**3)
        print(f"  Available disk space: {free_gb:.1f} GB")
        if free_gb < 10:
            print("  ⚠ Less than 10GB free space - may cause issues")
    except:
        print("  ⚠ Could not check disk space")


def optimize_configs():
    """Fix configuration files for better performance"""
    log_step("Optimizing configuration files...")

    # Check for problematic config paths
    config_files = [
        "projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py",
        "projects/ViTDet/configs/eva2_o365_to_coco/demo_global.py",
    ]

    for config_file in config_files:
        config_path = Path(config_file)
        if config_path.exists():
            print(f"  Checking {config_file}...")

            # Read the config file
            try:
                with open(config_path, "r") as f:
                    content = f.read()

                # Check for network paths that could cause delays
                if "/net/" in content or "mnt/data" in content:
                    print(f"    ⚠ Found network paths in {config_file}")
                    print(
                        "    This may cause slow loading. Consider using local paths."
                    )

                # Check for large batch sizes
                if "total_batch_size = " in content:
                    for line in content.split("\n"):
                        if (
                            "total_batch_size = " in line
                            and not line.strip().startswith("#")
                        ):
                            batch_size = line.split("=")[1].strip()
                            print(f"    Batch size: {batch_size}")
                            if int(batch_size) > 8:
                                print("    ⚠ Large batch size may cause memory issues")

            except Exception as e:
                print(f"    ⚠ Could not read {config_file}: {e}")
        else:
            print(f"  ⚠ Config file not found: {config_file}")


def create_optimized_launcher():
    """Create an optimized launcher script"""
    log_step("Creating optimized launcher script...")

    launcher_content = """#!/bin/bash
# HADM Optimized Launcher
# This script sets up the optimal environment and starts the server

echo "=== HADM Optimized Launcher ==="

# Set performance environment variables
export TORCH_HOME="/tmp/torch_cache"
export HF_HOME="/tmp/hf_cache"
export TRANSFORMERS_CACHE="/tmp/transformers_cache"
export DETECTRON2_CACHE="/tmp/detectron2_cache"
export CUDA_LAUNCH_BLOCKING="0"
export CUDA_CACHE_DISABLE="0"
export TORCH_CUDNN_V8_API_ENABLED="1"
export TORCH_COMPILE_DEBUG="0"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_inductor_cache"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"

# Create cache directories
mkdir -p /tmp/torch_cache /tmp/hf_cache /tmp/transformers_cache /tmp/detectron2_cache /tmp/torch_inductor_cache

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠ Virtual environment not found at ./venv"
fi

# Check GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || echo "⚠ Could not query GPU"

# Start the server
echo "Starting HADM FastAPI Server..."
if [ "$1" = "--lazy" ]; then
    echo "Using lazy loading mode for fastest startup..."
    python start_server_quick.py --lazy
else
    echo "Using normal mode..."
    python start_server_quick.py
fi
"""

    launcher_path = Path("start_optimized.sh")
    with open(launcher_path, "w") as f:
        f.write(launcher_content)

    # Make executable
    launcher_path.chmod(0o755)
    print(f"  Created optimized launcher: {launcher_path}")
    print("  Usage: ./start_optimized.sh [--lazy]")


def main():
    """Main setup function"""
    print("=" * 60)
    print("HADM Environment Optimization Setup")
    print("=" * 60)

    set_performance_environment_variables()
    optimize_python_path()
    check_system_requirements()
    optimize_configs()
    create_optimized_launcher()

    print("\n" + "=" * 60)
    print("Setup completed!")
    print("=" * 60)
    print("Recommendations:")
    print("1. Use lazy loading for fastest startup: ./start_optimized.sh --lazy")
    print("2. Move virtual environment to faster storage if possible")
    print("3. Monitor GPU memory usage during operation")
    print("4. Use batch processing for multiple images")
    print("=" * 60)


if __name__ == "__main__":
    main()
