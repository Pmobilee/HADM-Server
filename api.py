#!/usr/bin/env python3
"""
HADM FastAPI Server
Provides REST API endpoints for Human Artifact Detection using HADM-L and HADM-G models
"""

import os
import io
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from queue import Queue
from threading import Thread
import base64

import torch
import numpy as np
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Detectron2 imports
from detectron2.config import get_cfg, LazyConfig
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
hadm_l_model = None
hadm_g_model = None
device = None
request_queue = Queue()
is_processing = False


# Response models
class DetectionResult(BaseModel):
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_name: str
    artifacts: Dict[str, Any]


class InferenceResponse(BaseModel):
    success: bool
    message: str
    results: Optional[Dict[str, Any]] = None
    local_detections: Optional[List[DetectionResult]] = None
    global_detections: Optional[List[DetectionResult]] = None
    processing_time: Optional[float] = None


# FastAPI app
app = FastAPI(
    title="HADM FastAPI Server",
    description="Human Artifact Detection Models API for detecting artifacts in AI-generated images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_hadm_model(config_path: str, model_path: str, model_name: str):
    """Load a HADM model using detectron2 LazyConfig"""
    try:
        logger.info(f"Loading {model_name} model...")

        # Load config
        cfg = LazyConfig.load_config(config_path)

        # Set the model checkpoint path
        cfg.train.init_checkpoint = model_path

        # Create predictor
        predictor = DefaultPredictor(LazyConfig.instantiate(cfg))

        logger.info(f"{model_name} model loaded successfully")
        return predictor

    except Exception as e:
        logger.error(f"Failed to load {model_name} model: {e}")
        return None


def load_models():
    """Load HADM-L and HADM-G models into VRAM"""
    global hadm_l_model, hadm_g_model, device

    logger.info("Loading HADM models into VRAM...")

    # Setup detectron2 logger
    setup_logger()

    # Check for CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU")

    # Check for model files
    model_dir = Path("pretrained_models")
    hadm_l_path = model_dir / "HADM-L_0249999.pth"
    hadm_g_path = model_dir / "HADM-G_0249999.pth"
    eva_path = model_dir / "eva02_L_coco_det_sys_o365.pth"

    if not hadm_l_path.exists():
        logger.error(f"HADM-L model not found at {hadm_l_path}")
        return False

    if not hadm_g_path.exists():
        logger.error(f"HADM-G model not found at {hadm_g_path}")
        return False

    if not eva_path.exists():
        logger.error(f"EVA-02-L model not found at {eva_path}")
        return False

    try:
        # Load HADM-L model
        hadm_l_config = "projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py"
        hadm_l_model = load_hadm_model(hadm_l_config, str(hadm_l_path), "HADM-L")

        # Load HADM-G model
        hadm_g_config = "projects/ViTDet/configs/eva2_o365_to_coco/demo_global.py"
        hadm_g_model = load_hadm_model(hadm_g_config, str(hadm_g_path), "HADM-G")

        if hadm_l_model is None or hadm_g_model is None:
            return False

        logger.info("Models loaded successfully into VRAM")
        return True

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return False


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for detectron2 inference"""
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert to numpy array in BGR format (detectron2 expects BGR)
    image_array = np.array(image)
    image_bgr = image_array[:, :, ::-1]  # RGB to BGR

    return image_bgr


def run_hadm_l_inference(image_array: np.ndarray) -> List[DetectionResult]:
    """Run HADM-L (Local) inference"""
    global hadm_l_model

    if hadm_l_model is None:
        logger.error("HADM-L model not loaded")
        return []

    try:
        # Run inference
        predictions = hadm_l_model(image_array)

        # Convert predictions to DetectionResult format
        results = []
        if "instances" in predictions:
            instances = predictions["instances"]
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            classes = instances.pred_classes.cpu().numpy()

            for i in range(len(boxes)):
                bbox = boxes[i].tolist()  # [x1, y1, x2, y2]
                confidence = float(scores[i])
                class_id = int(classes[i])

                results.append(
                    DetectionResult(
                        bbox=bbox,
                        confidence=confidence,
                        class_name=f"local_artifact_class_{class_id}",
                        artifacts={
                            "class_id": class_id,
                            "type": "local_artifact",
                            "detection_method": "HADM-L",
                        },
                    )
                )

        return results

    except Exception as e:
        logger.error(f"HADM-L inference failed: {e}")
        return []


def run_hadm_g_inference(image_array: np.ndarray) -> List[DetectionResult]:
    """Run HADM-G (Global) inference"""
    global hadm_g_model

    if hadm_g_model is None:
        logger.error("HADM-G model not loaded")
        return []

    try:
        # Run inference
        predictions = hadm_g_model(image_array)

        # Convert predictions to DetectionResult format
        results = []
        if "instances" in predictions:
            instances = predictions["instances"]
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            classes = instances.pred_classes.cpu().numpy()

            for i in range(len(boxes)):
                bbox = boxes[i].tolist()  # [x1, y1, x2, y2]
                confidence = float(scores[i])
                class_id = int(classes[i])

                results.append(
                    DetectionResult(
                        bbox=bbox,
                        confidence=confidence,
                        class_name=f"global_artifact_class_{class_id}",
                        artifacts={
                            "class_id": class_id,
                            "type": "global_artifact",
                            "detection_method": "HADM-G",
                        },
                    )
                )

        return results

    except Exception as e:
        logger.error(f"HADM-G inference failed: {e}")
        return []


def process_inference_request(image: Image.Image, mode: str) -> InferenceResponse:
    """Process a single inference request"""
    import time

    start_time = time.time()

    try:
        # Preprocess image
        image_array = preprocess_image(image)

        local_detections = []
        global_detections = []

        # Run inference based on mode
        if mode in ["local", "both"]:
            local_detections = run_hadm_l_inference(image_array)

        if mode in ["global", "both"]:
            global_detections = run_hadm_g_inference(image_array)

        processing_time = time.time() - start_time

        return InferenceResponse(
            success=True,
            message="Inference completed successfully",
            local_detections=local_detections,
            global_detections=global_detections,
            processing_time=processing_time,
            results={
                "image_size": image.size,
                "mode": mode,
                "total_local_detections": len(local_detections),
                "total_global_detections": len(global_detections),
            },
        )

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return InferenceResponse(success=False, message=f"Inference failed: {str(e)}")


# API Endpoints


@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "HADM FastAPI Server",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": hadm_l_model is not None and hadm_g_model is not None,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": hadm_l_model is not None and hadm_g_model is not None,
        "gpu_available": torch.cuda.is_available(),
        "device": str(device) if device else "not_set",
    }


@app.post("/detect", response_model=InferenceResponse)
async def detect_artifacts(
    file: UploadFile = File(...), mode: str = "both"  # local, global, or both
):
    """
    Detect artifacts in uploaded image

    Args:
        file: Image file (JPEG format preferred)
        mode: Detection mode - 'local', 'global', or 'both'

    Returns:
        Detection results with bounding boxes and artifact information
    """

    # Validate mode
    if mode not in ["local", "global", "both"]:
        raise HTTPException(
            status_code=400, detail="Mode must be 'local', 'global', or 'both'"
        )

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Process inference
        result = process_inference_request(image, mode)

        return result

    except Exception as e:
        logger.error(f"Detection endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/detect/batch")
async def detect_artifacts_batch(
    files: List[UploadFile] = File(...), mode: str = "both"
):
    """
    Batch detection for multiple images
    """
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")

    results = []
    for file in files:
        try:
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            result = process_inference_request(image, mode)
            results.append({"filename": file.filename, "result": result})
        except Exception as e:
            results.append(
                {
                    "filename": file.filename,
                    "result": InferenceResponse(
                        success=False,
                        message=f"Failed to process {file.filename}: {str(e)}",
                    ),
                }
            )

    return {"batch_results": results}


@app.get("/models/status")
async def models_status():
    """Get model loading status and information"""
    return {
        "hadm_l_loaded": hadm_l_model is not None,
        "hadm_g_loaded": hadm_g_model is not None,
        "device": str(device) if device else "not_set",
        "gpu_memory": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
    }


@app.post("/models/reload")
async def reload_models():
    """Reload models (useful for updates)"""
    success = load_models()
    return {
        "success": success,
        "message": (
            "Models reloaded successfully" if success else "Failed to reload models"
        ),
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting HADM FastAPI Server...")

    # Load models
    success = load_models()
    if not success:
        logger.error("Failed to load models on startup")
    else:
        logger.info("HADM FastAPI Server started successfully")


if __name__ == "__main__":
    # Run the server
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=False, log_level="info")
