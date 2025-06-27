# HADM FastAPI Server

A FastAPI server for Human Artifact Detection Models (HADM) that provides real-time inference capabilities for detecting artifacts in AI-generated images. This server loads pretrained HADM-L (Local) and HADM-G (Global) models into VRAM and provides REST API endpoints for artifact detection.

## Features

- **Real-time Inference**: Fast artifact detection using pretrained HADM models
- **Queue Management**: Request queuing system for handling multiple concurrent requests  
- **Model Persistence**: Models remain loaded in VRAM for optimal performance
- **REST API**: Clean FastAPI endpoints with automatic documentation
- **Comprehensive Detection**: Supports both local and global artifact detection modes

## Setup

### Environment Setup

Run the automated setup script to create a virtual environment and install all dependencies:

```bash
python setup_environment.py
```

This script will:
- Create a Python virtual environment (`venv`)
- Install PyTorch with CUDA support
- Install all required dependencies from `requirements.txt`
- Install FastAPI and server components
- Download pretrained models
- Create necessary directories

To manually activate the environment:

```bash
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### Model Downloads

The setup script will automatically download the following pretrained models to the `pretrained_models` directory:

- **EVA-02-L**: Base backbone model from EVA-02-det
- **HADM-L**: Local Human Artifact Detection Model  
- **HADM-G**: Global Human Artifact Detection Model

Models will be downloaded from:
- [EVA-02-L](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_coco_det_sys_o365.pth)
- [HADM-L](https://www.dropbox.com/scl/fi/zwasvod906x1akzinnj3i/HADM-L_0249999.pth?rlkey=bqz5517tm8yt8l6ngzne4xejx&st=k1a1gzph&dl=0)
- [HADM-G](https://www.dropbox.com/scl/fi/bzj1m8p4cvm2vg4mai6uj/HADM-G_0249999.pth?rlkey=813x6wraigivc6qx02aut9p2r&st=n8rnb47r&dl=0)

## Usage

### Quick Start

After setting up the environment, start the FastAPI server:

```bash
source venv/bin/activate  # Activate the environment
python start_server.py    # Check dependencies and start server
# OR
python api.py             # Start server directly
```

The server will:
1. Start FastAPI with Uvicorn on port 8080 immediately
2. Load HADM-L and HADM-G models into VRAM in the background
3. Provide API endpoints for artifact detection once models are loaded

**Note**: The server starts quickly but models load in the background. Use `/models/status` endpoint to monitor loading progress.

### API Endpoints

Once the server is running, you can access:

- **API Documentation**: `http://localhost:8080/docs` (Swagger UI)
- **Alternative Docs**: `http://localhost:8080/redoc`
- **Health Check**: `http://localhost:8080/health` - Server and model status
- **Model Status**: `http://localhost:8080/models/status` - Detailed model loading information

### Inference Modes

The server supports two detection modes:

- **Local Detection (HADM-L)**: Detects localized artifacts in specific body parts (face, torso, arm, leg, hand, feet)
- **Global Detection (HADM-G)**: Detects global human-level artifacts (missing/extra body parts)
- **Combined Detection**: Returns results from both models

#### HADM-L Classes
- `face` - Facial artifacts
- `torso` - Torso/body artifacts  
- `arm` - Arm artifacts
- `leg` - Leg artifacts
- `hand` - Hand artifacts
- `feet` - Feet artifacts

#### HADM-G Classes
- `human missing [body_part]` - Missing body parts
- `human with extra [body_part]` - Extra body parts

### Input Requirements

- **Image Format**: JPEG images only
- **API**: Submit images via REST API endpoints
- **Response**: JSON with detection results including bounding boxes, confidence scores, and artifact classifications

## Test Images

Sample test images are available in the `test_images/` directory for testing the API endpoints.

## Architecture

The server architecture includes:
- FastAPI application with automatic documentation
- Model loading and VRAM management
- Request queuing system
- Comprehensive error handling
- Structured JSON responses with all available detection information

## Citation

Based on the research paper:
```bibtex
@article{Wang2024HADM,
  title={Detecting Human Artifacts from Text-to-Image Models},
  author={Wang, Kaihong and Zhang, Lingzhi and Zhang, Jianming},
  journal={arXiv preprint arXiv:2411.13842},
  year={2024}
}
```

## Acknowledgments

This implementation builds upon [EVA-02-det](https://github.com/baaivision/EVA/tree/master/EVA-02/det) and [Detectron2](https://github.com/facebookresearch/detectron2).
