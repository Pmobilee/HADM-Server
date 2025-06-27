# HADM Server Performance Optimization Guide

## Overview

This guide addresses the significant performance issues identified in the HADM FastAPI server, particularly the slow import times (52-100+ seconds) that are far from normal for a high-end GPU system.

## Performance Issues Identified

### 1. Import Performance Problems
- **PyTorch import**: 52.58s (should be <2s)
- **Detectron2 import**: 100.01s (should be <5s)
- **FastAPI import**: 10.52s (should be <1s)

### 2. Root Causes
- **Virtual environment on slow storage**: Network-mounted or slow filesystem
- **Cold start overhead**: First imports dramatically slower than subsequent ones
- **Config file network dependencies**: References to `/net/` paths causing delays
- **CUDA initialization during import**: GPU setup happening too early
- **Development package installations**: `.egg-link` files causing filesystem overhead

## Solutions Implemented

### 1. Lazy Loading System
The server now supports lazy loading mode that defers heavy imports until first request:

```bash
# Fast startup (lazy mode)
./start_optimized.sh --lazy

# Or using the quick launcher
python start_server_quick.py --lazy
```

**Benefits:**
- Server starts in <5 seconds
- Heavy imports (torch, detectron2) load only on first request
- Ideal for development and testing

### 2. Environment Optimization
Run the environment setup script to optimize your system:

```bash
python setup_environment.py
```

This script:
- Sets optimal environment variables
- Creates fast cache directories in `/tmp`
- Checks system requirements
- Creates optimized launcher scripts

### 3. Performance Monitoring
Monitor and diagnose performance issues:

```bash
# Analyze import performance
python monitor_performance.py --mode imports

# Monitor server startup
python monitor_performance.py --mode startup --lazy
```

## Optimization Strategies

### 1. Environment Variables
Set these for optimal performance:

```bash
# Cache directories on fast storage
export TORCH_HOME="/tmp/torch_cache"
export HF_HOME="/tmp/hf_cache"
export TRANSFORMERS_CACHE="/tmp/transformers_cache"
export DETECTRON2_CACHE="/tmp/detectron2_cache"

# CUDA optimizations
export CUDA_LAUNCH_BLOCKING="0"
export CUDA_CACHE_DISABLE="0"
export TORCH_CUDNN_V8_API_ENABLED="1"

# Memory optimizations
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"

# Single-threaded for inference
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
```

### 2. Virtual Environment Optimization
Move your virtual environment to faster storage:

```bash
# If using network storage, move to local/tmp
mv venv /tmp/hadm_venv
ln -s /tmp/hadm_venv venv

# Or create new venv on fast storage
python3 -m venv /tmp/hadm_venv
source /tmp/hadm_venv/bin/activate
pip install -r requirements.txt
```

### 3. Configuration File Fixes
Check and fix config files that reference network paths:

```python
# In projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py
# Change this:
train.init_checkpoint = "/net/ivcfs5/mnt/data/kwang/adobe/detectron2/pretrained_models/eva02_L_coco_det_sys_o365.pth"

# To this:
train.init_checkpoint = "pretrained_models/eva02_L_coco_det_sys_o365.pth"
```

### 4. GPU Memory Optimization
Pre-allocate GPU memory to avoid fragmentation:

```python
# In api.py, add after device setup:
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
```

## Performance Benchmarks

### Expected Performance (Optimized)
- **Server startup (lazy)**: <5 seconds
- **Server startup (normal)**: <30 seconds
- **First inference**: <10 seconds
- **Subsequent inference**: <2 seconds
- **PyTorch import**: <2 seconds
- **Detectron2 import**: <5 seconds

### Current Performance (Unoptimized)
- **Server startup**: 2+ minutes
- **PyTorch import**: 52+ seconds
- **Detectron2 import**: 100+ seconds

## Usage Recommendations

### 1. Development
Use lazy mode for fastest iteration:
```bash
./start_optimized.sh --lazy
```

### 2. Production
Use normal mode with background loading:
```bash
./start_optimized.sh
```

### 3. Debugging Performance
Monitor performance during startup:
```bash
python monitor_performance.py --mode startup --script api.py --lazy
```

## Troubleshooting

### 1. Still Slow After Optimization?
Check these common issues:

```bash
# Check if venv is on fast storage
df -h venv/

# Check for network mounts
mount | grep workspace

# Check GPU accessibility
nvidia-smi

# Check cache directories
ls -la /tmp/torch_cache /tmp/hf_cache
```

### 2. Import Errors in Lazy Mode
Ensure all imports are properly handled:
```python
# In functions that use heavy imports
ensure_heavy_imports()  # This loads torch, detectron2, etc.
```

### 3. Memory Issues
Monitor GPU memory usage:
```bash
watch -n 1 nvidia-smi
```

### 4. Config File Issues
Check for problematic paths:
```bash
grep -r "/net/" projects/ViTDet/configs/
grep -r "mnt/data" projects/ViTDet/configs/
```

## System Requirements

### Minimum
- **GPU**: NVIDIA RTX 3080 or better
- **RAM**: 16GB system RAM
- **Storage**: 20GB free space on fast storage (SSD/NVMe)
- **CUDA**: 11.6 or later

### Recommended
- **GPU**: NVIDIA RTX 4090/5090
- **RAM**: 32GB+ system RAM
- **Storage**: NVMe SSD with 50GB+ free space
- **Network**: Fast internet for initial model downloads

## Additional Optimizations

### 1. Model Parallel Loading
Load HADM-L and HADM-G models simultaneously:
```python
# Future enhancement: parallel model loading
with ThreadPoolExecutor(max_workers=2) as executor:
    future_l = executor.submit(load_hadm_model, hadm_l_config, hadm_l_path, "HADM-L")
    future_g = executor.submit(load_hadm_model, hadm_g_config, hadm_g_path, "HADM-G")
    hadm_l_model = future_l.result()
    hadm_g_model = future_g.result()
```

### 2. Model Quantization
Consider using quantized models for faster inference:
```python
# Future enhancement: model quantization
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

### 3. Batch Processing
Process multiple images simultaneously:
```python
# Use the /detect/batch endpoint for multiple images
# More efficient than individual requests
```

## Support

If you continue experiencing performance issues after following this guide:

1. Run the performance monitor and save the report
2. Check system specifications against requirements
3. Verify virtual environment location and package installations
4. Consider using a different container/environment setup

The lazy loading system should provide immediate relief for development workflows, while the environment optimizations address the underlying storage and configuration issues. 