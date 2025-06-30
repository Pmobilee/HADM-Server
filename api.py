#!/usr/bin/env python3
"""
HADM Unified FastAPI Server
Provides REST API endpoints and web dashboard for Human Artifact Detection using HADM-L and HADM-G models
Includes lazy loading to avoid CUDA multiprocessing issues
"""

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request, Depends, Form, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
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
from datetime import datetime
import json
import aiohttp

# Check for lazy mode before any heavy imports
lazy_mode = "--lazy" in sys.argv


def log_import_time(name):
    """Log import timing"""
    timestamp = time.strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
    print(f"[{timestamp}] Importing {name}...")


print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] Starting unified api.py...")

# Load environment variables
def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path(".env")
    if not env_file.exists():
        print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] ERROR: .env file not found!")
        print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] Please copy .env_example to .env and configure your settings")
        sys.exit(1)
    
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] ERROR: Failed to load .env file: {e}")
        sys.exit(1)

# Load environment variables first
load_env_file()
print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] Environment variables loaded from .env")

if lazy_mode:
    print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] LAZY MODE: Heavy imports will be deferred until model loading")

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
    print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] torch import took {torch_duration:.2f}s")

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
    print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] detectron2 imports took {detectron2_duration:.2f}s")

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
print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] uvicorn import took {uvicorn_duration:.2f}s")

print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] About to import fastapi...")
start_fastapi = time.time()
log_import_time("fastapi")
log_import_time("fastapi.responses")
log_import_time("fastapi.middleware.cors")
log_import_time("pydantic")
log_import_time("fastapi.templating")
log_import_time("fastapi.staticfiles")
fastapi_duration = time.time() - start_fastapi
print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] fastapi imports took {fastapi_duration:.2f}s")

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
models_loading = False
heavy_imports_loaded = not lazy_mode
model_load_lock = threading.Lock()

# Authentication
security = HTTPBasic()
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
API_KEY = os.getenv("API_KEY")

# Validate required environment variables
if not ADMIN_USERNAME or not ADMIN_PASSWORD or not API_KEY:
    logger.error("Missing required environment variables. Please check your .env file.")
    logger.error("Required: ADMIN_USERNAME, ADMIN_PASSWORD, API_KEY")
    sys.exit(1)

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
        torch.serialization.add_safe_globals([ListConfig, DictConfig, OmegaConf])

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

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify HTTP Basic Auth credentials"""
    if credentials.username != ADMIN_USERNAME or credentials.password != ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

def verify_api_key(request: Request, api_key: str = None):
    """Verify API key from header or query parameter"""
    # Try to get API key from multiple sources
    provided_key = api_key
    
    # Check query parameter first
    if not provided_key:
        provided_key = request.query_params.get("api_key")
    
    # Check headers
    if not provided_key:
        provided_key = request.headers.get("X-API-Key")
    
    # Check Authorization header
    if not provided_key:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            provided_key = auth_header[7:]  # Remove "Bearer " prefix
    
    if not provided_key or provided_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide via 'api_key' query parameter, 'X-API-Key' header, or 'Authorization: Bearer <key>' header."
        )
    return True

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

# Request models for different input types
class ImageUrlRequest(BaseModel):
    image_url: str = Field(..., description="URL of the image to analyze")
    mode: str = Field(default="both", description="Detection mode: 'local', 'global', or 'both'")

class ImageBase64Request(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image data (without data:image/... prefix)")
    mode: str = Field(default="both", description="Detection mode: 'local', 'global', or 'both'")

# FastAPI app
app = FastAPI(
    title="HADM Unified Server",
    description="Human Artifact Detection Models API with Web Dashboard",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.state.lazy_mode = lazy_mode

app_creation_duration = time.time() - start_app_creation
print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] FastAPI app creation took {app_creation_duration:.2f}s")

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
print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] CORS middleware took {cors_duration:.2f}s")

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
            eva02_path = os.getenv("EVA02_BACKBONE_PATH", "pretrained_models/eva02_L_coco_det_sys_o365.pth")
            cfg.train.init_checkpoint = str(Path(eva02_path))
            logger.info(f"Updated init_checkpoint for {model_name} to {eva02_path}")

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
            logger.info(f"Loading checkpoint from {model_path} with torch.load...")
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
                logger.error(f"Checkpoint for {model_name} does not contain a 'model' key!")
                return None

        logger.info(f"{model_name} model loaded successfully")
        if torch.cuda.is_available():
            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

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
                    image = self.aug.get_transform(image_bgr).apply_image(image_bgr)

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

def load_model_by_type(model_type: str):
    """Load a specific model type (hadm_l or hadm_g)"""
    global hadm_l_model, hadm_g_model, device, models_loading

    with model_load_lock:
        if models_loading:
            return {"success": False, "message": "Models are already being loaded"}

        models_loading = True
        try:
            ensure_heavy_imports()
            logger.info(f"Loading {model_type.upper()} model...")
            setup_logger()

            if not device:
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                else:
                    device = torch.device("cpu")
                    logger.warning("CUDA not available, using CPU")

            if model_type == "hadm_l":
                if hadm_l_model is not None:
                    return {"success": True, "message": "HADM-L model is already loaded"}
                
                hadm_l_path = Path(os.getenv("HADM_L_MODEL_PATH", "pretrained_models/HADM-L_0249999.pth"))
                if hadm_l_path.exists():
                    hadm_l_config = "projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py"
                    with log_time("Total HADM-L load time"):
                        hadm_l_model = load_hadm_model(hadm_l_config, str(hadm_l_path), "HADM-L")
                    if hadm_l_model is None:
                        return {"success": False, "message": "HADM-L model failed to load"}
                    return {"success": True, "message": "HADM-L model loaded successfully"}
                else:
                    return {"success": False, "message": f"HADM-L model not found at {hadm_l_path}"}
            
            elif model_type == "hadm_g":
                if hadm_g_model is not None:
                    return {"success": True, "message": "HADM-G model is already loaded"}
                
                hadm_g_path = Path(os.getenv("HADM_G_MODEL_PATH", "pretrained_models/HADM-G_0249999.pth"))
                if hadm_g_path.exists():
                    hadm_g_config = "projects/ViTDet/configs/eva2_o365_to_coco/demo_global.py"
                    with log_time("Total HADM-G load time"):
                        hadm_g_model = load_hadm_model(hadm_g_config, str(hadm_g_path), "HADM-G")
                    if hadm_g_model is None:
                        return {"success": False, "message": "HADM-G model failed to load"}
                    return {"success": True, "message": "HADM-G model loaded successfully"}
                else:
                    return {"success": False, "message": f"HADM-G model not found at {hadm_g_path}"}
            
            else:
                return {"success": False, "message": f"Unknown model type: {model_type}"}

        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"success": False, "message": f"Failed to load {model_type} model: {str(e)}"}
        finally:
            models_loading = False

def unload_model_by_type(model_type: str):
    """Unload a specific model type"""
    global hadm_l_model, hadm_g_model

    try:
        if model_type == "hadm_l":
            if hadm_l_model is not None:
                del hadm_l_model
                hadm_l_model = None
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("HADM-L model unloaded")
                return {"success": True, "message": "HADM-L model unloaded successfully"}
            else:
                return {"success": True, "message": "HADM-L model was not loaded"}
        
        elif model_type == "hadm_g":
            if hadm_g_model is not None:
                del hadm_g_model
                hadm_g_model = None
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("HADM-G model unloaded")
                return {"success": True, "message": "HADM-G model unloaded successfully"}
            else:
                return {"success": True, "message": "HADM-G model was not loaded"}
        
        else:
            return {"success": False, "message": f"Unknown model type: {model_type}"}

    except Exception as e:
        logger.error(f"Failed to unload {model_type} model: {e}")
        return {"success": False, "message": f"Failed to unload {model_type} model: {str(e)}"}

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

async def download_image_from_url(url: str):
    """Download image from URL and return PIL Image"""
    try:
        timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Failed to download image from URL. HTTP {response.status}"
                    )
                
                # Check content type
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    raise HTTPException(
                        status_code=400, 
                        detail=f"URL does not point to an image. Content-Type: {content_type}"
                    )
                
                # Read image data
                image_data = await response.read()
                
                # Check file size (limit to 50MB)
                if len(image_data) > 50 * 1024 * 1024:
                    raise HTTPException(
                        status_code=400, 
                        detail="Image file too large. Maximum size is 50MB."
                    )
                
                # Ensure heavy imports are loaded
                ensure_heavy_imports()
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(image_data))
                return image
                
    except aiohttp.ClientError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to download image from URL: {str(e)}"
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=400, 
            detail=f"Error processing image from URL: {str(e)}"
        )

def decode_base64_image(base64_data: str):
    """Decode base64 image data and return PIL Image"""
    try:
        # Ensure heavy imports are loaded
        ensure_heavy_imports()
        
        # Remove data URL prefix if present
        if base64_data.startswith('data:image/'):
            base64_data = base64_data.split(',', 1)[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_data)
        
        # Check file size (limit to 50MB)
        if len(image_data) > 50 * 1024 * 1024:
            raise HTTPException(
                status_code=400, 
                detail="Image file too large. Maximum size is 50MB."
            )
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        return image
        
    except Exception as decode_error:
        if "Invalid base64" in str(decode_error) or "binascii" in str(decode_error):
            raise HTTPException(
                status_code=400, 
                detail="Invalid base64 image data"
            )
        raise decode_error
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=400, 
            detail=f"Error processing base64 image: {str(e)}"
        )

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
            conf_bbox = draw.textbbox((x1, y1 - 15), confidence_text, font=small_font)
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
            draw.text((x1, y1 - 15), confidence_text, fill="white", font=small_font)
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
            conf_bbox = draw.textbbox((x1, y1 - 15), confidence_text, font=small_font)
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
            draw.text((x1, y1 - 15), confidence_text, fill="white", font=small_font)
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

        logger.info(f"[TIMING] Inference request ({mode}) - completed in {processing_time:.2f}s")

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
async def root(request: Request):
    """Redirect to interface if authenticated, otherwise to login"""
    auth_cookie = request.cookies.get("authenticated")
    if auth_cookie == "true":
        return RedirectResponse(url="/interface")
    return RedirectResponse(url="/login")

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve login page or redirect to interface if already authenticated"""
    auth_cookie = request.cookies.get("authenticated")
    if auth_cookie == "true":
        return RedirectResponse(url="/interface")
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login form submission"""
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        response = RedirectResponse(url="/interface", status_code=status.HTTP_302_FOUND)
        # In a real app, you'd set a secure session cookie here
        response.set_cookie(key="authenticated", value="true", httponly=True)
        return response
    else:
        return templates.TemplateResponse(
            "login.html", 
            {"request": request, "error": "Invalid credentials"}
        )

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve dashboard page"""
    # Simple cookie-based auth check
    auth_cookie = request.cookies.get("authenticated")
    if auth_cookie != "true":
        return RedirectResponse(url="/login")
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request, 
        "api_key": API_KEY
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Safely check GPU info only if torch is available
    gpu_available = "unknown"
    gpu_memory_gb = 0

    if heavy_imports_loaded and torch is not None:
        gpu_available = torch.cuda.is_available()
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.memory_allocated() / 1024**3

    return {
        "status": "healthy",
        "lazy_mode": lazy_mode,
        "heavy_imports_loaded": heavy_imports_loaded,
        "models_loaded": hadm_l_model is not None and hadm_g_model is not None,
        "models_loading": models_loading,
        "gpu_available": gpu_available,
        "device": str(device) if device else "not_set",
        "gpu_memory_gb": gpu_memory_gb,
    }

@app.get("/api/diagnostics")
async def get_diagnostics(request: Request, api_key: str = None):
    """Get system diagnostics"""
    verify_api_key(request, api_key)
    try:
        # Get current status
        gpu_memory_gb = 0
        gpu_memory_total_gb = 0
        device_name = "Unknown"

        if heavy_imports_loaded and torch is not None:
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                
                # Use nvidia-smi for accurate VRAM measurement
                try:
                    import subprocess
                    result = subprocess.run([
                        'nvidia-smi', '--query-gpu=memory.used,memory.total', 
                        '--format=csv,noheader,nounits'
                    ], capture_output=True, text=True, timeout=5)
                    
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if lines and lines[0]:
                            # Take the first GPU (index 0)
                            used_mb, total_mb = lines[0].split(', ')
                            gpu_memory_gb = float(used_mb) / 1024  # Convert MB to GB
                            gpu_memory_total_gb = float(total_mb) / 1024
                    else:
                        # Fallback to torch method
                        gpu_memory_gb = torch.cuda.memory_allocated() / 1024**3
                        gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
                    logger.warning(f"nvidia-smi not available, using torch fallback: {e}")
                    # Fallback to torch method
                    gpu_memory_gb = torch.cuda.memory_allocated() / 1024**3
                    gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

        return {
            "success": True,
            "worker_status": "RUNNING",
            "hadm_l_loaded": hadm_l_model is not None,
            "hadm_g_loaded": hadm_g_model is not None,
            "device": device_name,
            "vram_info": {
                "allocated_memory": gpu_memory_gb,
                "total_memory": gpu_memory_total_gb
            }
        }
    
    except Exception as e:
        logger.error(f"Error getting diagnostics: {str(e)}")
        return {
            "success": False,
            "message": f"Error: {str(e)}",
            "worker_status": "ERROR"
        }

@app.post("/api/control/{command}")
async def control_command(command: str, request: Request, api_key: str = None):
    """Send control command to load/unload models"""
    verify_api_key(request, api_key)
    try:
        valid_commands = ['load_l', 'unload_l', 'load_g', 'unload_g']
        if command not in valid_commands:
            raise HTTPException(status_code=400, detail=f"Invalid command. Valid commands: {valid_commands}")
        
        if command == 'load_l':
            result = load_model_by_type('hadm_l')
        elif command == 'unload_l':
            result = unload_model_by_type('hadm_l')
        elif command == 'load_g':
            result = load_model_by_type('hadm_g')
        elif command == 'unload_g':
            result = unload_model_by_type('hadm_g')
        
        return result
    
    except Exception as e:
        logger.error(f"Error handling control command: {str(e)}")
        return {"success": False, "message": f"Command failed: {str(e)}"}

@app.post("/api/detect", response_model=InferenceResponse)
async def detect_artifacts(
    request: Request,
    file: UploadFile = File(...), 
    mode: str = "both",
    api_key: str = None
):
    """
    Detect artifacts in uploaded image

    Args:
        file: Image file (JPEG format preferred)
        mode: Detection mode - 'local', 'global', or 'both'

    Returns:
        Detection results with bounding boxes and artifact information
    """
    verify_api_key(request, api_key)
    
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
            detail="HADM-L model not loaded. Please load the model first.",
        )

    if mode in ["global", "both"] and hadm_g_model is None:
        raise HTTPException(
            status_code=503,
            detail="HADM-G model not loaded. Please load the model first.",
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
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/api/v1/detect", response_model=InferenceResponse)
async def detect_artifacts_api_key(
    request: Request,
    file: UploadFile = File(...), 
    mode: str = "both",
    api_key: str = None
):
    """
    Detect artifacts in uploaded image using API key authentication
    
    Args:
        file: Image file (JPEG format preferred)
        mode: Detection mode - 'local', 'global', or 'both'
        api_key: API key for authentication (can be passed as query param or header)
    
    Returns:
        Detection results with bounding boxes and artifact information
    """
    # Verify API key
    verify_api_key(request, api_key)
    
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
            detail="HADM-L model not loaded. Please load the model first.",
        )

    if mode in ["global", "both"] and hadm_g_model is None:
        raise HTTPException(
            status_code=503,
            detail="HADM-G model not loaded. Please load the model first.",
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
        logger.error(f"API detection endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/api/v1/detect-url", response_model=InferenceResponse)
async def detect_artifacts_from_url(
    request: Request,
    image_request: ImageUrlRequest,
    api_key: str = None
):
    """
    Detect artifacts in image from URL using API key authentication
    
    Args:
        image_request: Request containing image URL and detection mode
        api_key: API key for authentication (can be passed as query param or header)
    
    Returns:
        Detection results with bounding boxes and artifact information
    """
    # Verify API key
    verify_api_key(request, api_key)
    
    # Check if models are still loading
    if models_loading:
        raise HTTPException(
            status_code=503,
            detail="Models are still loading. Please try again in a few moments.",
        )

    # Check if required models are loaded
    if image_request.mode in ["local", "both"] and hadm_l_model is None:
        raise HTTPException(
            status_code=503,
            detail="HADM-L model not loaded. Please load the model first.",
        )

    if image_request.mode in ["global", "both"] and hadm_g_model is None:
        raise HTTPException(
            status_code=503,
            detail="HADM-G model not loaded. Please load the model first.",
        )

    # Validate mode
    if image_request.mode not in ["local", "global", "both"]:
        raise HTTPException(
            status_code=400, detail="Mode must be 'local', 'global', or 'both'"
        )

    try:
        # Download image from URL
        image = await download_image_from_url(image_request.image_url)

        # Process inference
        result = process_inference_request(image, image_request.mode)

        return result

    except Exception as e:
        logger.error(f"URL detection endpoint failed: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/api/v1/detect-base64", response_model=InferenceResponse)
async def detect_artifacts_from_base64(
    request: Request,
    image_request: ImageBase64Request,
    api_key: str = None
):
    """
    Detect artifacts in base64 encoded image using API key authentication
    
    Args:
        image_request: Request containing base64 image data and detection mode
        api_key: API key for authentication (can be passed as query param or header)
    
    Returns:
        Detection results with bounding boxes and artifact information
    """
    # Verify API key
    verify_api_key(request, api_key)
    
    # Check if models are still loading
    if models_loading:
        raise HTTPException(
            status_code=503,
            detail="Models are still loading. Please try again in a few moments.",
        )

    # Check if required models are loaded
    if image_request.mode in ["local", "both"] and hadm_l_model is None:
        raise HTTPException(
            status_code=503,
            detail="HADM-L model not loaded. Please load the model first.",
        )

    if image_request.mode in ["global", "both"] and hadm_g_model is None:
        raise HTTPException(
            status_code=503,
            detail="HADM-G model not loaded. Please load the model first.",
        )

    # Validate mode
    if image_request.mode not in ["local", "global", "both"]:
        raise HTTPException(
            status_code=400, detail="Mode must be 'local', 'global', or 'both'"
        )

    try:
        # Decode base64 image
        image = decode_base64_image(image_request.image_base64)

        # Process inference
        result = process_inference_request(image, image_request.mode)

        return result

    except Exception as e:
        logger.error(f"Base64 detection endpoint failed: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/models/status")
async def models_status(request: Request, api_key: str = None):
    """Get model loading status and information"""
    verify_api_key(request, api_key)
    
    # Safely check GPU memory only if torch is available
    gpu_memory_gb = 0
    gpu_memory_total_gb = 0

    if heavy_imports_loaded and torch is not None and torch.cuda.is_available():
        # Use nvidia-smi for accurate VRAM measurement
        try:
            import subprocess
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=memory.used,memory.total', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    # Take the first GPU (index 0)
                    used_mb, total_mb = lines[0].split(', ')
                    gpu_memory_gb = float(used_mb) / 1024  # Convert MB to GB
                    gpu_memory_total_gb = float(total_mb) / 1024
            else:
                # Fallback to torch method
                gpu_memory_gb = torch.cuda.memory_allocated() / 1024**3
                gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to torch method
            gpu_memory_gb = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    return {
        "hadm_l_loaded": hadm_l_model is not None,
        "hadm_g_loaded": hadm_g_model is not None,
        "models_loading": models_loading,
        "device": str(device) if device else "not_set",
        "heavy_imports_loaded": heavy_imports_loaded,
        "torch_available": torch is not None,
        "cuda_available": (torch.cuda.is_available() if torch is not None else False),
        "gpu_memory_gb": gpu_memory_gb,
        "gpu_memory_total_gb": gpu_memory_total_gb,
    }

@app.get("/interface")
async def web_interface(request: Request):
    """Web interface for uploading images and viewing detection results"""
    # Check authentication
    auth_cookie = request.cookies.get("authenticated")
    if auth_cookie != "true":
        return RedirectResponse(url="/login")
    
    return templates.TemplateResponse("interface.html", {"request": request})

@app.post("/interface/detect")
async def web_detect_artifacts(
    request: Request, file: UploadFile = File(...), mode: str = "both"
):
    """
    Web interface endpoint for artifact detection with visual results
    """
    # Check authentication
    auth_cookie = request.cookies.get("authenticated")
    if auth_cookie != "true":
        return RedirectResponse(url="/login")
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
                "error": "HADM-L model not loaded. Please load the model first.",
            },
        )

    if mode in ["global", "both"] and hadm_g_model is None:
        return templates.TemplateResponse(
            "interface.html",
            {
                "request": request,
                "error": "HADM-G model not loaded. Please load the model first.",
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
            "interface.html", {"request": request, "error": "File must be an image"}
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

@app.get("/api/logs")
async def get_logs(request: Request, api_key: str = None):
    """Get system logs from log files"""
    verify_api_key(request, api_key)
    
    import re
    from datetime import datetime
    
    def parse_log_line(line: str) -> Dict[str, str]:
        """Parse a log line and extract timestamp, level, and message"""
        # Handle different log formats
        # Format 1: INFO:api:message
        # Format 2: INFO:     ip - "request" status
        # Format 3: [timestamp] message
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        level = "INFO"
        message = line.strip()
        
        # Try to extract timestamp from [HH:MM:SS] format
        timestamp_match = re.search(r'\[(\d{2}:\d{2}:\d{2})\]', line)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            message = re.sub(r'\[\d{2}:\d{2}:\d{2}\]\s*', '', line).strip()
        
        # Try to extract log level
        level_match = re.search(r'(DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL)', line)
        if level_match:
            level = level_match.group(1)
            if level == "WARNING":
                level = "WARN"
        
        # Clean up the message
        message = re.sub(r'^(DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL):\s*', '', message)
        message = re.sub(r'^[^:]*:\s*', '', message)  # Remove logger name
        
        return {
            "timestamp": timestamp,
            "level": level,
            "message": message
        }
    
    def read_log_file(path: Path, max_lines: int = 100) -> List[Dict[str, str]]:
        """Read and parse log file"""
        if not path.exists():
            return []
        
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            
            # Get last max_lines
            lines = lines[-max_lines:] if len(lines) > max_lines else lines
            
            parsed_logs = []
            for line in lines:
                if line.strip():  # Skip empty lines
                    parsed_logs.append(parse_log_line(line))
            
            return parsed_logs
        except Exception as e:
            logger.error(f"Error reading log file {path}: {e}")
            return []
    
    try:
        log_dir = Path("logs")
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Read API logs (acts as both uvicorn and general logs)
        api_log_file = log_dir / f"api-{date_str}.log"
        api_logs = read_log_file(api_log_file)
        
        # Filter logs for different categories
        uvicorn_logs = []
        hadm_l_logs = []
        hadm_g_logs = []
        
        for log_entry in api_logs:
            message = log_entry["message"].lower()
            
            # Categorize logs
            if any(keyword in message for keyword in ["http", "request", "response", "uvicorn", "started server"]):
                uvicorn_logs.append(log_entry)
            elif any(keyword in message for keyword in ["hadm-l", "hadm_l", "local"]):
                hadm_l_logs.append(log_entry)
            elif any(keyword in message for keyword in ["hadm-g", "hadm_g", "global"]):
                hadm_g_logs.append(log_entry)
            else:
                # General logs go to uvicorn category
                uvicorn_logs.append(log_entry)
        
        return {
            "success": True,
            "logs": {
                "uvicorn": uvicorn_logs[-50:],  # Last 50 entries
                "hadm-l": hadm_l_logs[-50:],
                "hadm-g": hadm_g_logs[-50:]
            }
        }
    
    except Exception as e:
        logger.error(f"Error getting logs: {str(e)}")
        return {
            "success": False,
            "message": f"Error reading logs: {str(e)}",
            "logs": {
                "uvicorn": [],
                "hadm-l": [],
                "hadm-g": []
            }
        }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize server on startup"""
    logger.info("Starting HADM Unified Server...")

    if lazy_mode:
        logger.info("LAZY MODE: Server is ready. Models will load when requested.")
        logger.info("HADM Unified Server started successfully in lazy mode")
    else:
        logger.info("Server is ready. Heavy imports loaded at startup.")
        logger.info("HADM Unified Server started successfully")

if __name__ == "__main__":
    print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] About to start uvicorn server...")
    start_uvicorn_run = time.time()

    # Get server configuration from environment
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8080"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    # Run the server
    uvicorn.run("api:app", host=host, port=port, reload=False, log_level=log_level)
