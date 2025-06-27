#!/usr/bin/env python3
"""
HADM FastAPI Server
Provides REST API endpoints for Human Artifact Detection using HADM-L and HADM-G models
"""

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
import uvicorn
import contextlib
import base64
import threading
from queue import Queue
from pathlib import Path
import logging
import asyncio
import io
import os
import time
import sys
import argparse
from typing import Optional, Dict, Any, List, Union

# Check for lazy mode before any heavy imports
lazy_mode = "--lazy" in sys.argv


def log_import_time(name):
    """Log import timing"""
    timestamp = time.strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
    print(f"[{timestamp}] Importing {name}...")


print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] Starting api.py imports...")
if lazy_mode:
    print(
        f"[{time.strftime('%H:%M:%S.%f')[:-3]}] LAZY MODE: Heavy imports will be deferred until first request"
    )

log_import_time("os")

log_import_time("io")

log_import_time("asyncio")

log_import_time("logging")

log_import_time("pathlib.Path")

log_import_time("queue.Queue")

log_import_time("threading")

log_import_time("base64")

log_import_time("contextlib")

# Type hints for lazy imports
torch: Optional[Any] = None
np: Optional[Any] = None
Image: Optional[Any] = None
ImageDraw: Optional[Any] = None
ImageFont: Optional[Any] = None
LazyConfig: Optional[Any] = None
instantiate: Optional[Any] = None
setup_logger: Optional[Any] = None
DetectionCheckpointer: Optional[Any] = None
T: Optional[Any] = None
ListConfig: Optional[Any] = None
DictConfig: Optional[Any] = None
OmegaConf: Optional[Any] = None
xops: Optional[Any] = None

# Conditional heavy imports
if not lazy_mode:
    print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] About to import torch...")
    start_torch = time.time()
    log_import_time("torch")
    import torch

    torch_duration = time.time() - start_torch
    print(
        f"[{time.strftime('%H:%M:%S.%f')[:-3]}] torch import took {torch_duration:.2f}s"
    )

    log_import_time("numpy")
    import numpy as np

    log_import_time("PIL.Image")
    from PIL import Image, ImageDraw, ImageFont

    # Detectron2 imports
    print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] About to import detectron2...")
    start_detectron2 = time.time()

    log_import_time("detectron2.config")
    from detectron2.config import get_cfg, LazyConfig, instantiate

    log_import_time("detectron2.engine.defaults")
    from detectron2.engine.defaults import DefaultPredictor

    log_import_time("detectron2.utils.logger")
    from detectron2.utils.logger import setup_logger

    log_import_time("detectron2.checkpoint")
    from detectron2.checkpoint import DetectionCheckpointer

    log_import_time("detectron2.data.transforms")
    from detectron2.data import transforms as T

    log_import_time("detectron2.data.detection_utils")
    from detectron2.data.detection_utils import convert_PIL_to_numpy

    log_import_time("omegaconf")
    from omegaconf import ListConfig, DictConfig, OmegaConf

    # Register OmegaConf classes as safe globals for torch.load
    import torch.serialization

    torch.serialization.add_safe_globals([ListConfig, DictConfig, OmegaConf])

    detectron2_duration = time.time() - start_detectron2
    print(
        f"[{time.strftime('%H:%M:%S.%f')[:-3]}] detectron2 imports took {detectron2_duration:.2f}s"
    )

    try:
        log_import_time("xformers.ops")
        import xformers.ops as xops
    except ImportError:
        print("xformers not found. Some models may not work.")
        xops = None

print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] About to import uvicorn...")
start_uvicorn = time.time()
log_import_time("uvicorn")

uvicorn_duration = time.time() - start_uvicorn
print(
    f"[{time.strftime('%H:%M:%S.%f')[:-3]}] uvicorn import took {uvicorn_duration:.2f}s"
)

print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] About to import fastapi...")
start_fastapi = time.time()
log_import_time("fastapi")

log_import_time("fastapi.responses")

log_import_time("fastapi.middleware.cors")

log_import_time("pydantic")

log_import_time("fastapi.templating")

log_import_time("fastapi.staticfiles")

fastapi_duration = time.time() - start_fastapi
print(
    f"[{time.strftime('%H:%M:%S.%f')[:-3]}] fastapi imports took {fastapi_duration:.2f}s"
)

print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] All imports completed!")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] About to create FastAPI app...")
start_app_creation = time.time()

# Global variables for models
hadm_l_model: Optional[Any] = None
hadm_g_model: Optional[Any] = None
device: Optional[Any] = None
request_queue = Queue()
is_processing = False
models_loading = False
heavy_imports_loaded = not lazy_mode


def ensure_heavy_imports():
    """Ensure heavy imports are loaded - decorator-friendly version"""
    if lazy_mode and not heavy_imports_loaded:
        load_heavy_imports()


def load_heavy_imports():
    """Load heavy imports when in lazy mode"""
    global torch, np, Image, ImageDraw, ImageFont, LazyConfig, instantiate
    global setup_logger, DetectionCheckpointer, T, ListConfig, DictConfig, OmegaConf, xops
    global heavy_imports_loaded

    if heavy_imports_loaded:
        return

    logger.info("Loading heavy imports (torch, detectron2, etc.)...")
    start_heavy = time.time()

    try:
        import torch as torch_module
        import numpy as numpy_module
        from PIL import (
            Image as PIL_Image,
            ImageDraw as PIL_ImageDraw,
            ImageFont as PIL_ImageFont,
        )
        from detectron2.config import (
            LazyConfig as D2_LazyConfig,
            instantiate as D2_instantiate,
        )
        from detectron2.utils.logger import setup_logger as D2_setup_logger
        from detectron2.checkpoint import (
            DetectionCheckpointer as D2_DetectionCheckpointer,
        )
        from detectron2.data import transforms as D2_T
        from omegaconf import (
            ListConfig as OC_ListConfig,
            DictConfig as OC_DictConfig,
            OmegaConf as OC_OmegaConf,
        )

        # Assign to global variables
        torch = torch_module
        np = numpy_module
        Image = PIL_Image
        ImageDraw = PIL_ImageDraw
        ImageFont = PIL_ImageFont
        LazyConfig = D2_LazyConfig
        instantiate = D2_instantiate
        setup_logger = D2_setup_logger
        DetectionCheckpointer = D2_DetectionCheckpointer
        T = D2_T
        ListConfig = OC_ListConfig
        DictConfig = OC_DictConfig
        OmegaConf = OC_OmegaConf

        # Register OmegaConf classes as safe globals for torch.load
        import torch.serialization

        torch.serialization.add_safe_globals(
            [ListConfig, DictConfig, OmegaConf])

        try:
            import xformers.ops as xformers_ops

            xops = xformers_ops
        except ImportError:
            logger.warning("xformers not found. Some models may not work.")
            xops = None

        heavy_imports_loaded = True
        duration = time.time() - start_heavy
        logger.info(f"Heavy imports loaded in {duration:.2f}s")

    except Exception as e:
        logger.error(f"Failed to load heavy imports: {e}")
        raise RuntimeError(f"Critical imports failed: {e}")


def trigger_lazy_load():
    """If in lazy mode and models are not loaded, load them. This is blocking."""
    with threading.Lock():
        if lazy_mode and (hadm_l_model is None or hadm_g_model is None):
            logger.info(
                "Lazy loading triggered by first request. This will take a moment..."
            )
            load_models()


# Utility context manager for timing steps with logging
@contextlib.contextmanager
def log_time(step: str):
    start = time.time()
    logger.info(f"[TIMING] {step} - started")
    try:
        yield
    finally:
        duration = time.time() - start
        logger.info(f"[TIMING] {step} - completed in {duration:.2f}s")


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
app.state.lazy_mode = lazy_mode

app_creation_duration = time.time() - start_app_creation
print(
    f"[{time.strftime('%H:%M:%S.%f')[:-3]}] FastAPI app creation took {app_creation_duration:.2f}s"
)

print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] Adding CORS middleware...")
start_cors = time.time()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cors_duration = time.time() - start_cors
print(
    f"[{time.strftime('%H:%M:%S.%f')[:-3]}] CORS middleware took {cors_duration:.2f}s"
)

# Setup templates and static files
print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] Setting up templates and static files...")
templates = Jinja2Templates(directory="templates")

# Create static and templates directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] Defining functions...")


def load_hadm_model(config_path: str, model_path: str, model_name: str):
    """Load a HADM model using LazyConfig"""
    try:
        # Ensure heavy imports are loaded
        ensure_heavy_imports()

        logger.info(f"Loading {model_name} model from {model_path}...")

        # Check if files exist
        if not Path(config_path).exists():
            logger.error(f"Config file not found: {config_path}")
            return None

        if not Path(model_path).exists():
            logger.error(f"Model file not found: {model_path}")
            return None

        # Load LazyConfig
        with log_time(f"{model_name} -> Load LazyConfig"):
            cfg = LazyConfig.load(config_path)
        logger.info(f"Config loaded successfully for {model_name}")

        # Update model checkpoint path
        if hasattr(cfg, "train") and hasattr(cfg.train, "init_checkpoint"):
            cfg.train.init_checkpoint = str(
                Path("pretrained_models/eva02_L_coco_det_sys_o365.pth")
            )
            logger.info(f"Updated init_checkpoint for {model_name}")

        # Clean up any device references in model config to avoid conflicts
        if hasattr(cfg.model, "device"):
            delattr(cfg.model, "device")

        # Set device for training config
        if hasattr(cfg, "train"):
            cfg.train.device = str(device)
            logger.info(f"Set device to {device} for {model_name}")

        logger.info(f"About to instantiate model for {model_name}...")

        # Instantiate model
        with log_time(f"{model_name} -> Instantiate model"):
            model = instantiate(cfg.model)
            model.to(device)
            model.eval()

        # Load the HADM checkpoint directly, bypassing fvcore's rigid checkpointer
        with log_time(f"{model_name} -> Load checkpoint"):
            logger.info(
                f"Loading checkpoint from {model_path} with torch.load...")
            checkpoint_data = torch.load(
                model_path, map_location=device, weights_only=False
            )

            # The actual model weights are in the 'model' key
            if "model" in checkpoint_data:
                # Use DetectionCheckpointer just for its weight-mapping logic
                checkpointer = DetectionCheckpointer(model)
                # The _load_model method is what handles the state dict loading
                checkpointer._load_model(checkpoint_data)
            else:
                logger.error(
                    f"Checkpoint for {model_name} does not contain a 'model' key!"
                )
                return None

        logger.info(f"{model_name} model loaded successfully")
        logger.info(
            f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )

        # Create predictor-like wrapper
        class HADMPredictor:
            def __init__(self, model, cfg):
                self.model = model
                # OmegaConf doesn't have clone(), use copy() instead
                self.cfg = OmegaConf.create(cfg)
                self.model.eval()

                # Use proper preprocessing for ViTDet models (1024x1024 square)
                # Based on projects/ViTDet/configs/common/coco_loader_lsj_1024.py
                self.aug = T.ResizeShortestEdge(
                    short_edge_length=1024, max_size=1024)
                self.input_format = "BGR"  # detectron2 default

            def __call__(self, image_bgr):
                with torch.no_grad():
                    # Apply pre-processing to image.
                    height, width = image_bgr.shape[:2]

                    # Apply resizing augmentation to make it 1024x1024
                    image = self.aug.get_transform(
                        image_bgr).apply_image(image_bgr)

                    # Convert to tensor
                    image = torch.as_tensor(
                        image.astype("float32").transpose(2, 0, 1))

                    inputs = {"image": image, "height": height, "width": width}
                    predictions = self.model([inputs])
                    return predictions[0]

        predictor = HADMPredictor(model, cfg)
        return predictor

    except Exception as e:
        logger.error(f"Failed to load {model_name} model: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def load_models():
    """Load HADM-L and HADM-G models into VRAM, skipping already loaded ones."""
    global hadm_l_model, hadm_g_model, device, models_loading

    if models_loading:
        logger.info("Models are already being loaded...")
        return False

    models_loading = True
    success = True
    try:
        ensure_heavy_imports()
        logger.info("Checking and loading HADM models into VRAM...")
        setup_logger()

        if not device:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                logger.warning("CUDA not available, using CPU")

        model_dir = Path("pretrained_models")
        hadm_l_path = model_dir / "HADM-L_0249999.pth"
        hadm_g_path = model_dir / "HADM-G_0249999.pth"

        # Load HADM-L if not already loaded
        if hadm_l_model is None:
            if hadm_l_path.exists():
                hadm_l_config = "projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py"
                with log_time("Total HADM-L load time"):
                    hadm_l_model = load_hadm_model(
                        hadm_l_config, str(hadm_l_path), "HADM-L")
                if hadm_l_model is None:
                    logger.error("HADM-L model failed to load.")
                    success = False
            else:
                logger.error(f"HADM-L model not found at {hadm_l_path}")
                success = False
        else:
            logger.info("HADM-L model is already loaded.")

        # Load HADM-G if not already loaded
        if hadm_g_model is None:
            if hadm_g_path.exists():
                hadm_g_config = "projects/ViTDet/configs/eva2_o365_to_coco/demo_global.py"
                with log_time("Total HADM-G load time"):
                    hadm_g_model = load_hadm_model(
                        hadm_g_config, str(hadm_g_path), "HADM-G")
                if hadm_g_model is None:
                    logger.error("HADM-G model failed to load.")
                    success = False
            else:
                logger.error(f"HADM-G model not found at {hadm_g_path}")
                success = False
        else:
            logger.info("HADM-G model is already loaded.")

        if success:
            logger.info(
                "All required models are loaded successfully into VRAM.")
        else:
            logger.warning("One or more models failed to load.")

        return success

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        models_loading = False


def preprocess_image(image):
    """Preprocess image for detectron2 inference"""
    # Ensure heavy imports are loaded
    ensure_heavy_imports()

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert to numpy array in BGR format (detectron2 expects BGR)
    image_array = np.array(image)
    image_bgr = image_array[:, :, ::-1]  # RGB to BGR

    return image_bgr


def run_hadm_l_inference(image_array) -> List[DetectionResult]:
    """Run HADM-L (Local) inference on image"""
    # Ensure heavy imports are loaded
    ensure_heavy_imports()

    if hadm_l_model is None:
        logger.error("HADM-L model not loaded")
        return []

    try:
        # Run inference
        predictions = hadm_l_model(image_array)

        results = []
        if "instances" in predictions and len(predictions["instances"]) > 0:
            instances = predictions["instances"]
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            classes = instances.pred_classes.cpu().numpy()

            for i in range(len(boxes)):
                bbox = boxes[i].tolist()  # [x1, y1, x2, y2]
                confidence = float(scores[i])
                class_id = int(classes[i])

                # HADM-L class mapping (based on LOCAL_HUMAN_ARTIFACT_CATEGORIES)
                class_names = {
                    0: "face",
                    1: "torso",
                    2: "arm",
                    3: "leg",
                    4: "hand",
                    5: "feet",
                }

                results.append(
                    DetectionResult(
                        bbox=bbox,
                        confidence=confidence,
                        class_name=class_names.get(
                            class_id, f"local_artifact_class_{class_id}"
                        ),
                        artifacts={
                            "class_id": class_id,
                            "type": "local_artifact",
                            "detection_method": "HADM-L",
                            "bbox_area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                        },
                    )
                )

        return results

    except Exception as e:
        logger.error(f"HADM-L inference failed: {e}")
        return []


def run_hadm_g_inference(image_array) -> List[DetectionResult]:
    """Run HADM-G (Global) inference on image"""
    # Ensure heavy imports are loaded
    ensure_heavy_imports()

    if hadm_g_model is None:
        logger.error("HADM-G model not loaded")
        return []

    try:
        # Run inference
        predictions = hadm_g_model(image_array)

        results = []
        if "instances" in predictions and len(predictions["instances"]) > 0:
            instances = predictions["instances"]
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            classes = instances.pred_classes.cpu().numpy()

            for i in range(len(boxes)):
                bbox = boxes[i].tolist()  # [x1, y1, x2, y2]
                confidence = float(scores[i])
                class_id = int(classes[i])

                # HADM-G class mapping (based on GLOBAL_HUMAN_ARTIFACT_CATEGORIES)
                class_names = {
                    0: "human missing arm",
                    1: "human missing face",
                    2: "human missing feet",
                    3: "human missing hand",
                    4: "human missing leg",
                    5: "human missing torso",
                    6: "human with extra arm",
                    7: "human with extra face",
                    8: "human with extra feet",
                    9: "human with extra hand",
                    10: "human with extra leg",
                    11: "human with extra torso",
                }

                results.append(
                    DetectionResult(
                        bbox=bbox,
                        confidence=confidence,
                        class_name=class_names.get(
                            class_id, f"global_artifact_class_{class_id}"
                        ),
                        artifacts={
                            "class_id": class_id,
                            "type": "global_artifact",
                            "detection_method": "HADM-G",
                            "bbox_area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                        },
                    )
                )

        return results

    except Exception as e:
        logger.error(f"HADM-G inference failed: {e}")
        return []


def draw_bounding_boxes(
    image,
    local_detections: List[DetectionResult],
    global_detections: List[DetectionResult],
):
    """Draw bounding boxes and labels on the image"""
    # Ensure heavy imports are loaded
    ensure_heavy_imports()

    # Create a copy of the image to draw on
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)

    # Try to use a better font, fall back to default if not available
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
        small_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
        )
    except (OSError, IOError):
        try:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        except:
            font = None
            small_font = None

    # Draw local detections in red
    for detection in local_detections:
        bbox = detection.bbox
        x1, y1, x2, y2 = bbox

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Prepare label text
        label = f"LOCAL: {detection.class_name}"
        confidence_text = f"Conf: {detection.confidence:.2f}"

        # Draw label background
        if font:
            label_bbox = draw.textbbox((x1, y1 - 30), label, font=font)
            conf_bbox = draw.textbbox(
                (x1, y1 - 15), confidence_text, font=small_font)
        else:
            label_bbox = draw.textbbox((x1, y1 - 30), label)
            conf_bbox = draw.textbbox((x1, y1 - 15), confidence_text)

        draw.rectangle(
            [
                label_bbox[0] - 2,
                label_bbox[1] - 2,
                max(label_bbox[2], conf_bbox[2]) + 2,
                conf_bbox[3] + 2,
            ],
            fill="red",
        )

        # Draw label text
        if font:
            draw.text((x1, y1 - 30), label, fill="white", font=font)
            draw.text((x1, y1 - 15), confidence_text,
                      fill="white", font=small_font)
        else:
            draw.text((x1, y1 - 30), label, fill="white")
            draw.text((x1, y1 - 15), confidence_text, fill="white")

    # Draw global detections in blue
    for detection in global_detections:
        bbox = detection.bbox
        x1, y1, x2, y2 = bbox

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)

        # Prepare label text
        label = f"GLOBAL: {detection.class_name}"
        confidence_text = f"Conf: {detection.confidence:.2f}"

        # Draw label background
        if font:
            label_bbox = draw.textbbox((x1, y1 - 30), label, font=font)
            conf_bbox = draw.textbbox(
                (x1, y1 - 15), confidence_text, font=small_font)
        else:
            label_bbox = draw.textbbox((x1, y1 - 30), label)
            conf_bbox = draw.textbbox((x1, y1 - 15), confidence_text)

        draw.rectangle(
            [
                label_bbox[0] - 2,
                label_bbox[1] - 2,
                max(label_bbox[2], conf_bbox[2]) + 2,
                conf_bbox[3] + 2,
            ],
            fill="blue",
        )

        # Draw label text
        if font:
            draw.text((x1, y1 - 30), label, fill="white", font=font)
            draw.text((x1, y1 - 15), confidence_text,
                      fill="white", font=small_font)
        else:
            draw.text((x1, y1 - 30), label, fill="white")
            draw.text((x1, y1 - 15), confidence_text, fill="white")

    return img_with_boxes


def process_inference_request(image, mode: str) -> InferenceResponse:
    """Process a single inference request"""
    import time

    # Ensure heavy imports are loaded
    ensure_heavy_imports()

    with log_time(f"Inference request ({mode})"):
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

        logger.info(
            f"[TIMING] Inference request ({mode}) - completed in {processing_time:.2f}s"
        )

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
        "lazy_mode": lazy_mode,
        "heavy_imports_loaded": heavy_imports_loaded,
        "models_loaded": hadm_l_model is not None and hadm_g_model is not None,
        "models_loading": models_loading,
        "gpu_available": (
            torch.cuda.is_available() if heavy_imports_loaded else "unknown"
        ),
        "device": str(device) if device else "not_set",
        "gpu_memory_gb": (
            torch.cuda.memory_allocated() / 1024**3
            if heavy_imports_loaded and torch.cuda.is_available()
            else 0
        ),
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
    trigger_lazy_load()

    # Check if models are still loading
    if models_loading:
        raise HTTPException(
            status_code=503,
            detail="Models are still loading. Please try again in a few moments.",
        )

    # Check if required models are loaded
    if mode in ["local", "both"] and hadm_l_model is None:
        raise HTTPException(
            status_code=503,
            detail="HADM-L model not loaded. Please check model status.",
        )

    if mode in ["global", "both"] and hadm_g_model is None:
        raise HTTPException(
            status_code=503,
            detail="HADM-G model not loaded. Please check model status.",
        )

    # Validate mode
    if mode not in ["local", "global", "both"]:
        raise HTTPException(
            status_code=400, detail="Mode must be 'local', 'global', or 'both'"
        )

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Ensure heavy imports are loaded
        ensure_heavy_imports()

        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Process inference
        result = process_inference_request(image, mode)

        return result

    except Exception as e:
        logger.error(f"Detection endpoint failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/detect/batch")
async def detect_artifacts_batch(
    files: List[UploadFile] = File(...), mode: str = "both"
):
    """
    Batch detection for multiple images
    """
    if len(files) > 10:  # Limit batch size
        raise HTTPException(
            status_code=400, detail="Maximum 10 images per batch")

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
        "models_loading": models_loading,
        "device": str(device) if device else "not_set",
        "gpu_memory_gb": (
            torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        ),
        "gpu_memory_total_gb": (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
            if torch.cuda.is_available()
            else 0
        ),
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


@app.get("/interface")
async def web_interface(request: Request):
    """Web interface for uploading images and viewing detection results"""
    return templates.TemplateResponse("interface.html", {"request": request})


@app.post("/interface/detect")
async def web_detect_artifacts(
    request: Request, file: UploadFile = File(...), mode: str = "both"
):
    """
    Web interface endpoint for artifact detection with visual results
    """
    trigger_lazy_load()

    # Check if models are still loading
    if models_loading:
        return templates.TemplateResponse(
            "interface.html",
            {
                "request": request,
                "error": "Models are still loading. Please try again in a few moments.",
            },
        )

    # Check if required models are loaded
    if mode in ["local", "both"] and hadm_l_model is None:
        return templates.TemplateResponse(
            "interface.html",
            {
                "request": request,
                "error": "HADM-L model not loaded. Please check model status.",
            },
        )

    if mode in ["global", "both"] and hadm_g_model is None:
        return templates.TemplateResponse(
            "interface.html",
            {
                "request": request,
                "error": "HADM-G model not loaded. Please check model status.",
            },
        )

    # Validate mode
    if mode not in ["local", "global", "both"]:
        return templates.TemplateResponse(
            "interface.html",
            {"request": request, "error": "Mode must be 'local', 'global', or 'both'"},
        )

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        return templates.TemplateResponse(
            "interface.html", {"request": request,
                               "error": "File must be an image"}
        )

    try:
        # Ensure heavy imports are loaded
        ensure_heavy_imports()

        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Process inference
        result = process_inference_request(image, mode)

        if result.success:
            # Draw bounding boxes on the image
            img_with_boxes = draw_bounding_boxes(
                image, result.local_detections or [], result.global_detections or []
            )

            # Save the image with bounding boxes
            img_buffer = io.BytesIO()
            img_with_boxes.save(img_buffer, format="PNG")
            img_buffer.seek(0)

            # Convert to base64 for display
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

            return templates.TemplateResponse(
                "interface.html",
                {
                    "request": request,
                    "result": result,
                    "image_base64": img_base64,
                    "original_filename": file.filename,
                    "mode": mode,
                },
            )
        else:
            return templates.TemplateResponse(
                "interface.html", {"request": request, "error": result.message}
            )

    except Exception as e:
        logger.error(f"Web detection endpoint failed: {e}")
        return templates.TemplateResponse(
            "interface.html",
            {"request": request, "error": f"Detection failed: {str(e)}"},
        )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting HADM FastAPI Server...")

    if lazy_mode:
        logger.info(
            "LAZY MODE: Server is ready. Models will load on first request.")
        logger.info("HADM FastAPI Server started successfully in lazy mode")
    else:
        logger.info(
            "Server is ready to accept requests. Models will load in background..."
        )

        # Load models in background thread to avoid blocking startup
        def load_models_background():
            success = load_models()
            if not success:
                logger.error("Failed to load models in background")
            else:
                logger.info("HADM models loaded successfully in background")

        # Start model loading in background
        model_thread = threading.Thread(
            target=load_models_background, daemon=True)
        model_thread.start()

        logger.info("HADM FastAPI Server started successfully")


if __name__ == "__main__":
    print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] About to start uvicorn server...")
    start_uvicorn_run = time.time()

    # Run the server
    uvicorn.run("api:app", host="0.0.0.0", port=8080,
                reload=False, log_level="info")
