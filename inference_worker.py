#!/usr/bin/env python3
"""
HADM Inference Worker
Background worker process that handles model loading and inference tasks via Redis queue.
This process loads and maintains HADM-L and HADM-G models in VRAM.
"""

import os
import sys
import time
import json
import base64
import logging
import traceback
from io import BytesIO
from typing import Optional, Dict, Any, List
import redis
from rq import Worker, Queue, Connection, get_current_job
import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for models
hadm_l_model: Optional[Any] = None
hadm_g_model: Optional[Any] = None
device: Optional[Any] = None

# Heavy imports - loaded when worker starts
torch = None
LazyConfig = None
instantiate = None
setup_logger = None
DetectionCheckpointer = None
T = None
ListConfig = None
DictConfig = None
OmegaConf = None


def load_heavy_imports():
    """Load all heavy ML libraries"""
    global torch, LazyConfig, instantiate, setup_logger, DetectionCheckpointer
    global T, ListConfig, DictConfig, OmegaConf

    logger.info("Loading heavy imports (torch, detectron2, etc.)...")
    start_time = time.time()

    import torch as torch_module
    torch = torch_module

    import numpy as np_module

    from detectron2.config import LazyConfig as LazyConfig_module
    from detectron2.config import instantiate as instantiate_module
    from detectron2.utils.logger import setup_logger as setup_logger_module
    from detectron2.checkpoint import DetectionCheckpointer as DetectionCheckpointer_module
    from detectron2.data import transforms as T_module
    from omegaconf import ListConfig as ListConfig_module, DictConfig as DictConfig_module, OmegaConf as OmegaConf_module

    LazyConfig = LazyConfig_module
    instantiate = instantiate_module
    setup_logger = setup_logger_module
    DetectionCheckpointer = DetectionCheckpointer_module
    T = T_module
    ListConfig = ListConfig_module
    DictConfig = DictConfig_module
    OmegaConf = OmegaConf_module

    # Register OmegaConf classes as safe globals for torch.load
    import torch.serialization
    torch.serialization.add_safe_globals([ListConfig, DictConfig, OmegaConf])

    # Try to import xformers
    try:
        import xformers.ops as xops
    except ImportError:
        logger.warning(
            "xformers not found. Some models may not work optimally.")

    duration = time.time() - start_time
    logger.info(f"Heavy imports loaded in {duration:.2f} seconds")


def get_device():
    """Get the appropriate device for inference"""
    global device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
    return device


def load_hadm_model(config_path: str, model_path: str, model_name: str):
    """Load a HADM model from config and checkpoint files"""
    logger.info(f"Loading {model_name} model...")
    start_time = time.time()

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load configuration
    cfg = LazyConfig.load_file(config_path)

    # Set device
    device = get_device()

    # Build model
    model = instantiate(cfg.model)
    model.to(device)
    model.eval()

    # Load checkpoint
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(model_path)

    # Create custom predictor class
    class HADMPredictor:
        def __init__(self, model, cfg):
            self.model = model
            self.cfg = cfg
            self.device = device

        def __call__(self, image_bgr):
            # Convert BGR to RGB
            image_rgb = image_bgr[:, :, ::-1]

            # Apply transforms
            height, width = image_rgb.shape[:2]
            transform = T.ResizeShortestEdge([800], 1333)
            image_tensor = transform.get_transform(
                image_rgb).apply_image(image_rgb)
            image_tensor = torch.as_tensor(
                image_tensor.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image_tensor.to(
                self.device), "height": height, "width": width}

            with torch.no_grad():
                predictions = self.model([inputs])

            return predictions[0]

    predictor = HADMPredictor(model, cfg)

    duration = time.time() - start_time
    logger.info(
        f"{model_name} model loaded successfully in {duration:.2f} seconds")

    return predictor


def load_models():
    """Load both HADM-L and HADM-G models"""
    global hadm_l_model, hadm_g_model

    logger.info("Loading HADM models...")

    # Model paths
    base_path = "./pretrained_models"
    hadm_l_config = os.path.join(base_path, "HADM-L", "config.yaml")
    hadm_l_weights = os.path.join(base_path, "HADM-L", "model_final.pth")
    hadm_g_config = os.path.join(base_path, "HADM-G", "config.yaml")
    hadm_g_weights = os.path.join(base_path, "HADM-G", "model_final.pth")

    try:
        # Load HADM-L
        if os.path.exists(hadm_l_config) and os.path.exists(hadm_l_weights):
            hadm_l_model = load_hadm_model(
                hadm_l_config, hadm_l_weights, "HADM-L")
        else:
            logger.warning("HADM-L model files not found")

        # Load HADM-G
        if os.path.exists(hadm_g_config) and os.path.exists(hadm_g_weights):
            hadm_g_model = load_hadm_model(
                hadm_g_config, hadm_g_weights, "HADM-G")
        else:
            logger.warning("HADM-G model files not found")

        logger.info("Model loading completed")

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


def run_hadm_l_inference(image_array) -> List[Dict]:
    """Run HADM-L inference on image array"""
    if hadm_l_model is None:
        raise ValueError("HADM-L model not loaded")

    logger.info("Running HADM-L inference...")
    start_time = time.time()

    try:
        outputs = hadm_l_model(image_array)

        # Process outputs
        instances = outputs["instances"]
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()

        # Class names for HADM-L (local artifacts)
        class_names = ["local_artifact"]

        detections = []
        for i in range(len(boxes)):
            if scores[i] > 0.5:  # Confidence threshold
                detection = {
                    "bbox": boxes[i].tolist(),
                    "confidence": float(scores[i]),
                    "class_name": class_names[0],
                    "artifacts": {"type": "local", "confidence": float(scores[i])}
                }
                detections.append(detection)

        duration = time.time() - start_time
        logger.info(
            f"HADM-L inference completed in {duration:.2f} seconds, found {len(detections)} detections")

        return detections

    except Exception as e:
        logger.error(f"Error in HADM-L inference: {str(e)}")
        raise


def run_hadm_g_inference(image_array) -> List[Dict]:
    """Run HADM-G inference on image array"""
    if hadm_g_model is None:
        raise ValueError("HADM-G model not loaded")

    logger.info("Running HADM-G inference...")
    start_time = time.time()

    try:
        outputs = hadm_g_model(image_array)

        # Process outputs
        instances = outputs["instances"]
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()

        # Class names for HADM-G (global artifacts)
        class_names = ["global_artifact"]

        detections = []
        for i in range(len(boxes)):
            if scores[i] > 0.5:  # Confidence threshold
                detection = {
                    "bbox": boxes[i].tolist(),
                    "confidence": float(scores[i]),
                    "class_name": class_names[0],
                    "artifacts": {"type": "global", "confidence": float(scores[i])}
                }
                detections.append(detection)

        duration = time.time() - start_time
        logger.info(
            f"HADM-G inference completed in {duration:.2f} seconds, found {len(detections)} detections")

        return detections

    except Exception as e:
        logger.error(f"Error in HADM-G inference: {str(e)}")
        raise


def execute_inference(job_data):
    """Execute inference job"""
    try:
        # Decode image data
        image_data = base64.b64decode(job_data['image_data'])
        image = Image.open(BytesIO(image_data))
        image_array = np.array(image)

        # Convert RGB to BGR for OpenCV format
        if len(image_array.shape) == 3:
            image_array = image_array[:, :, ::-1]

        mode = job_data.get('mode', 'both')

        local_detections = []
        global_detections = []

        start_time = time.time()

        if mode in ['local', 'both']:
            if hadm_l_model is not None:
                local_detections = run_hadm_l_inference(image_array)
            else:
                logger.warning(
                    "HADM-L model not loaded, skipping local inference")

        if mode in ['global', 'both']:
            if hadm_g_model is not None:
                global_detections = run_hadm_g_inference(image_array)
            else:
                logger.warning(
                    "HADM-G model not loaded, skipping global inference")

        processing_time = time.time() - start_time

        result = {
            "success": True,
            "message": "Inference completed successfully",
            "local_detections": local_detections,
            "global_detections": global_detections,
            "processing_time": processing_time
        }

        logger.info(
            f"Inference job completed in {processing_time:.2f} seconds")
        return result

    except Exception as e:
        logger.error(f"Error in inference job: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "message": f"Inference failed: {str(e)}",
            "local_detections": [],
            "global_detections": [],
            "processing_time": 0
        }


def handle_control_command(command_data):
    """Handle control commands"""
    global hadm_l_model, hadm_g_model

    try:
        command = command_data.get('command')
        logger.info(f"Handling control command: {command}")

        if command == 'load_l':
            if hadm_l_model is None:
                base_path = "./pretrained_models"
                hadm_l_config = os.path.join(
                    base_path, "HADM-L", "config.yaml")
                hadm_l_weights = os.path.join(
                    base_path, "HADM-L", "model_final.pth")
                hadm_l_model = load_hadm_model(
                    hadm_l_config, hadm_l_weights, "HADM-L")
                return {"success": True, "message": "HADM-L model loaded successfully"}
            else:
                return {"success": True, "message": "HADM-L model already loaded"}

        elif command == 'unload_l':
            if hadm_l_model is not None:
                del hadm_l_model
                hadm_l_model = None
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return {"success": True, "message": "HADM-L model unloaded successfully"}
            else:
                return {"success": True, "message": "HADM-L model not loaded"}

        elif command == 'load_g':
            if hadm_g_model is None:
                base_path = "./pretrained_models"
                hadm_g_config = os.path.join(
                    base_path, "HADM-G", "config.yaml")
                hadm_g_weights = os.path.join(
                    base_path, "HADM-G", "model_final.pth")
                hadm_g_model = load_hadm_model(
                    hadm_g_config, hadm_g_weights, "HADM-G")
                return {"success": True, "message": "HADM-G model loaded successfully"}
            else:
                return {"success": True, "message": "HADM-G model already loaded"}

        elif command == 'unload_g':
            if hadm_g_model is not None:
                del hadm_g_model
                hadm_g_model = None
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return {"success": True, "message": "HADM-G model unloaded successfully"}
            else:
                return {"success": True, "message": "HADM-G model not loaded"}

        elif command == 'status':
            vram_info = {}
            if torch and torch.cuda.is_available():
                vram_info = {
                    "total_memory": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    "allocated_memory": torch.cuda.memory_allocated() / (1024**3),
                    "cached_memory": torch.cuda.memory_reserved() / (1024**3)
                }

            return {
                "success": True,
                "message": "Status retrieved successfully",
                "status": {
                    "hadm_l_loaded": hadm_l_model is not None,
                    "hadm_g_loaded": hadm_g_model is not None,
                    "device": str(device) if device else "Not set",
                    "vram_info": vram_info
                }
            }

        else:
            return {"success": False, "message": f"Unknown command: {command}"}

    except Exception as e:
        logger.error(f"Error handling control command: {str(e)}")
        logger.error(traceback.format_exc())
        return {"success": False, "message": f"Command failed: {str(e)}"}

# Initialize worker globally


def initialize_worker():
    """Initialize worker with heavy imports and models"""
    logger.info("Initializing HADM Inference Worker...")

    # Load heavy imports
    load_heavy_imports()

    # Initialize device
    get_device()

    # Load models on startup
    try:
        load_models()
    except Exception as e:
        logger.error(f"Failed to load models on startup: {str(e)}")
        logger.info("Worker will continue without models loaded")


def main():
    """Main worker function"""
    logger.info("Starting HADM Inference Worker...")

    # Initialize worker
    initialize_worker()

    # Connect to Redis
    redis_conn = redis.Redis(host='localhost', port=6379, db=0)

    logger.info("Worker ready, listening for jobs...")

    # Start worker - RQ will handle the job dispatching
    with Connection(redis_conn):
        worker = Worker(['inference_q', 'control_q'], connection=redis_conn)
        worker.work()


if __name__ == '__main__':
    main()
