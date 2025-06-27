#!/usr/bin/env python3
"""
Debug script to test individual import times
"""
import time
import sys
import os

def test_import(module_name, import_statement):
    print(f"Testing: {module_name}")
    start = time.time()
    try:
        exec(import_statement)
        duration = time.time() - start
        print(f"  ✓ {module_name}: {duration:.2f}s")
        return duration
    except Exception as e:
        duration = time.time() - start
        print(f"  ✗ {module_name}: {duration:.2f}s - ERROR: {e}")
        return duration

def main():
    print("=== Import Speed Test ===")
    total_start = time.time()
    
    # Test basic imports
    test_import("os", "import os")
    test_import("sys", "import sys") 
    test_import("time", "import time")
    test_import("pathlib", "from pathlib import Path")
    
    # Test heavy imports
    torch_time = test_import("torch", "import torch")
    test_import("numpy", "import numpy as np")
    test_import("PIL", "from PIL import Image")
    
    # Test web framework imports
    fastapi_time = test_import("fastapi", "from fastapi import FastAPI")
    uvicorn_time = test_import("uvicorn", "import uvicorn")
    
    # Test detectron2 imports
    detectron2_time = test_import("detectron2.config", "from detectron2.config import get_cfg")
    
    total_duration = time.time() - total_start
    print(f"\n=== Summary ===")
    print(f"Total import time: {total_duration:.2f}s")
    print(f"PyTorch: {torch_time:.2f}s")
    print(f"FastAPI: {fastapi_time:.2f}s") 
    print(f"Uvicorn: {uvicorn_time:.2f}s")
    print(f"Detectron2: {detectron2_time:.2f}s")
    
    # Check for network activity
    print(f"\n=== Environment Check ===")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"TORCH_HOME: {os.environ.get('TORCH_HOME', 'Not set')}")
    print(f"HF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
    print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', 'Not set')}")

if __name__ == "__main__":
    main() 