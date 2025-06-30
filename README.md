# HADM Server - Human Artifact Detection Models

> **Credits**: This server implementation is based on the original HADM (Human Artifact Detection Models) work by Wang et al. All model architectures, training methodologies, and detection algorithms are credited to the original authors.
>
> **Original Repository**: https://github.com/wangkaihong/HADM  
> **Paper**: Wang, Kaihong, Zhang, Lingzhi, and Zhang, Jianming. "Detecting Human Artifacts from Text-to-Image Models." arXiv preprint arXiv:2411.13842 (2024). https://arxiv.org/abs/2411.13842
>
> This implementation provides a production-ready FastAPI server wrapper around the original HADM models with web dashboard and REST API endpoints.

## Overview

A unified FastAPI server that provides both web dashboard and REST API endpoints for detecting AI-generated artifacts in images using HADM-L (Local) and HADM-G (Global) models.

### Screenshots

| Feature | Screenshot |
|---------|------------|
| **Control Dashboard** | ![HADM Control Center](HADM_control_center.png) |
| **Detection Results** | ![Detection Results](detection_results.png) |
| **API Documentation** | Available at `/docs` (Swagger UI style) |

## Features

- **Unified Architecture**: Single server process with web dashboard and API endpoints
- **Lazy Loading**: Models load on-demand to avoid CUDA multiprocessing issues
- **Web Dashboard**: Interactive interface for model management and image analysis
- **REST API**: Programmatic access with both Basic Auth and API key authentication
- **Real-time Monitoring**: VRAM usage, model status, and system diagnostics
- **Flexible Authentication**: Environment-based credentials and API keys

## Quick Start

### 1. Environment Setup

Copy the example environment file and configure your credentials:

```bash
cp .env_example .env
```

Edit `.env` with your preferred credentials (default: admin/password):

```bash
# Authentication credentials
ADMIN_USERNAME=admin
ADMIN_PASSWORD=password

# API Key for secure API access (generate a new one for production)
API_KEY=hadm_7k9m2n4p8q1r5s3t6v9w2x5z8a1b4c7e
```

**⚠️ Important**: Change the default credentials and API key before deploying to production!

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Models

Download the pre-trained models from the original HADM repository and place them in the `pretrained_models/` directory:

- **HADM-L**: `HADM-L_0249999.pth` - Local artifact detection model
- **HADM-G**: `HADM-G_0249999.pth` - Global artifact detection model  
- **EVA-02 Backbone**: `eva02_L_coco_det_sys_o365.pth` - Backbone model

> **Note**: Model download links are available in the original HADM repository: https://github.com/wangkaihong/HADM

### 4. Start Server

```bash
# Start in lazy mode (recommended)
./start_unified.sh --lazy

# Or start with heavy imports pre-loaded
./start_unified.sh start
```

### 5. Access the Application

- **Web Dashboard**: http://localhost:8080/dashboard
- **Simple Interface**: http://localhost:8080/interface  
- **API Documentation**: http://localhost:8080/docs (Swagger UI style testing)
- **Login**: Use credentials from your `.env` file (default: admin/password)

## Usage

### Web Interface

1. Navigate to http://localhost:8080
2. Login with your credentials
3. Choose between Dashboard (advanced) or Interface (simple)
4. Load models using the control buttons
5. Upload images for artifact detection

### REST API

#### Basic Authentication

```bash
curl -u admin:password \
  -X POST \
  -F "file=@image.jpg" \
  -F "mode=both" \
  http://localhost:8080/api/detect
```

#### API Key Authentication

```bash
curl -X POST \
  -F "file=@image.jpg" \
  -F "mode=both" \
  "http://localhost:8080/api/v1/detect?api_key=your_api_key_here"
```

### Model Management

#### Load Models
```bash
# Load HADM-L model
curl -u admin:password -X POST http://localhost:8080/api/control/load_l

# Load HADM-G model  
curl -u admin:password -X POST http://localhost:8080/api/control/load_g
```

#### Check Status
```bash
curl -u admin:password http://localhost:8080/api/diagnostics
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ADMIN_USERNAME` | admin | Web dashboard username |
| `ADMIN_PASSWORD` | password | Web dashboard password |
| `API_KEY` | (random) | API key for programmatic access |
| `SERVER_HOST` | 0.0.0.0 | Server bind address |
| `SERVER_PORT` | 8080 | Server port |
| `HADM_L_MODEL_PATH` | pretrained_models/HADM-L_0249999.pth | Local model path |
| `HADM_G_MODEL_PATH` | pretrained_models/HADM-G_0249999.pth | Global model path |
| `EVA02_BACKBONE_PATH` | pretrained_models/eva02_L_coco_det_sys_o365.pth | Backbone path |
| `LOG_LEVEL` | INFO | Logging level |

### Detection Modes

Based on the original HADM paper methodology:

- **`local`**: Detect local artifacts (body parts: face, torso, arm, leg, hand, feet)
- **`global`**: Detect global artifacts (missing/extra body parts)
- **`both`**: Run both local and global detection (recommended)

## Server Management

### Startup Options

```bash
# Start with lazy loading (recommended)
./start_unified.sh --lazy

# Start with pre-loaded imports
./start_unified.sh start

# Restart server
./start_unified.sh restart

# Restart in lazy mode
./start_unified.sh restart-lazy

# Stop server
./start_unified.sh stop

# Check status
./start_unified.sh status

# View logs
./start_unified.sh logs [lines]
```

### Model Operations

- **Load Models**: Use dashboard buttons or API endpoints
- **Unload Models**: Individual or "Unload All Models" button
- **Monitor VRAM**: Real-time usage via nvidia-smi integration
- **Check Status**: System diagnostics and model states

## API Endpoints

### Authentication Endpoints
- `GET /` - Redirect to interface (if authenticated) or login
- `GET /login` - Login page
- `POST /login` - Handle login
- `GET /dashboard` - Advanced dashboard (requires auth)
- `GET /interface` - Simple interface (requires auth)

### Detection Endpoints
- `POST /api/detect` - Detect artifacts (Basic Auth)
- `POST /api/v1/detect` - Detect artifacts (API Key)
- `POST /interface/detect` - Web form detection

### Management Endpoints
- `GET /api/diagnostics` - System status and VRAM usage
- `POST /api/control/{command}` - Model control (load_l, unload_l, load_g, unload_g)
- `GET /api/logs` - System logs
- `GET /models/status` - Model loading status
- `GET /health` - Health check

### API Documentation
- `GET /docs` - Interactive Swagger UI for API testing
- `GET /redoc` - Alternative API documentation

## Architecture

### Unified Design
- **Single Process**: No separate workers, eliminates CUDA multiprocessing issues
- **Lazy Loading**: Heavy ML imports and models load on-demand
- **Thread Safe**: Model loading/unloading with proper locking
- **Real-time Monitoring**: Live VRAM and status updates

### Model Pipeline
1. **Image Preprocessing**: RGB→BGR conversion, resizing to 1024x1024
2. **HADM-L Detection**: Local artifact detection (body parts)
3. **HADM-G Detection**: Global artifact detection (missing/extra parts)
4. **Post-processing**: Bounding box drawing, confidence scoring
5. **Results**: JSON response with detections and metadata

## Troubleshooting

### Common Issues

1. **Models not loading**: Check file paths in `.env` and ensure models exist
2. **CUDA errors**: Use lazy mode: `./start_unified.sh --lazy`
3. **Authentication failed**: Verify credentials in `.env` file
4. **Port conflicts**: Change `SERVER_PORT` in `.env`
5. **VRAM issues**: Use "Unload All Models" to free memory

### Logs and Debugging

```bash
# View recent logs
./start_unified.sh logs 50

# Check server status
./start_unified.sh status

# Test health endpoint
curl http://localhost:8080/health
```

## Security Notes

- **Change default credentials** in `.env` before production use
- **Generate new API key** for production deployments  
- **Use HTTPS** in production environments
- **Restrict network access** to authorized users only
- **Regularly rotate** API keys and passwords

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- Detectron2
- FastAPI and dependencies
- NVIDIA GPU (recommended)
- 8GB+ VRAM for both models

## Citation

If you use this server implementation, please cite the original HADM paper:

```bibtex
@article{Wang2024HADM,
  title={Detecting Human Artifacts from Text-to-Image Models},
  author={Wang, Kaihong and Zhang, Lingzhi and Zhang, Jianming},
  journal={arXiv preprint arXiv:2411.13842},
  year={2024}
}
```

## Acknowledgments

- **Original HADM Models**: Wang, Kaihong, Zhang, Lingzhi, and Zhang, Jianming
- **Original Repository**: https://github.com/wangkaihong/HADM
- **Paper**: https://arxiv.org/abs/2411.13842
- **Base Framework**: Built on Detectron2 and EVA-02

## License

This server implementation is provided as-is. Please refer to the original HADM model licenses and terms of use from the original repository: https://github.com/wangkaihong/HADM
