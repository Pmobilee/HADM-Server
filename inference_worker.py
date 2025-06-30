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
from rq import Worker, Queue, get_current_job
import numpy as np
from PIL import Image
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define project base directory
BASE_DIR = Path(__file__).resolve().parent

# Global variables for models
hadm_l_model: Optional[Any] = None
hadm_g_model: Optional[Any] = None
device: Optional[Any] = None
worker_initialized: bool = False

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

    # Skip if already loaded
    if torch is not None:
        return

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
        # Ensure heavy imports are loaded before using torch
        if torch is None:
            load_heavy_imports()
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Avoid calling get_device_name() which triggers CUDA initialization
            logger.info("Using CUDA device")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
    return device


def load_hadm_model(config_path: str, model_path: str, model_name: str):
    """Load a HADM model using LazyConfig (same approach as api.py)"""
    logger.info(f"[LOAD_START] Loading {model_name} model from {model_path}...")
    start_time = time.time()

    # Check if files exist
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = get_device()

    # Load LazyConfig
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
    model = instantiate(cfg.model)
    model.to(device)
    model.eval()

    # Load the HADM checkpoint directly, bypassing fvcore's rigid checkpointer
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
        raise ValueError(f"Checkpoint for {model_name} does not contain a 'model' key!")

    logger.info(f"[LOAD_COMPLETE] {model_name} model loaded successfully")
    # Only log GPU memory if CUDA is available and initialized
    if torch.cuda.is_available() and torch.cuda.is_initialized():
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
                image = self.aug.get_transform(image_bgr).apply_image(image_bgr)

                # Convert to tensor
                image = torch.as_tensor(
                    image.astype("float32").transpose(2, 0, 1))

                inputs = {"image": image, "height": height, "width": width}
                predictions = self.model([inputs])
                return predictions[0]

    predictor = HADMPredictor(model, cfg)

    duration = time.time() - start_time
    logger.info(f"[LOAD_FINAL] {model_name} model loaded successfully in {duration:.2f} seconds")

    return predictor


def load_models(force_reload: bool = False):
    """Load both HADM-L and HADM-G models"""
    global hadm_l_model, hadm_g_model

    # Ensure heavy imports are loaded first (like api.py does)
    load_heavy_imports()
    
    logger.info("Loading HADM models...")

    # Model paths - using correct safetensors format
    base_path = BASE_DIR / "pretrained_models"
    hadm_l_weights = base_path / "HADM-L_0249999.pth"
    hadm_g_weights = base_path / "HADM-G_0249999.pth"

    try:
        # Load HADM-L
        if hadm_l_weights.exists():
            hadm_l_config = "projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py"
            hadm_l_model = load_hadm_model(hadm_l_config, str(hadm_l_weights), "HADM-L")
        else:
            logger.warning(f"HADM-L model file not found: {hadm_l_weights}")

        # Load HADM-G
        if hadm_g_weights.exists():
            hadm_g_config = "projects/ViTDet/configs/eva2_o365_to_coco/demo_global.py"
            hadm_g_model = load_hadm_model(hadm_g_config, str(hadm_g_weights), "HADM-G")
        else:
            logger.warning(f"HADM-G model file not found: {hadm_g_weights}")

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

        # HADM-L class mapping (based on LOCAL_HUMAN_ARTIFACT_CATEGORIES)
        class_names = {
            0: "face",
            1: "torso",
            2: "arm",
            3: "leg",
            4: "hand",
            5: "feet",
        }

        detections = []
        for i in range(len(boxes)):
            if scores[i] > 0.5:  # Confidence threshold
                bbox = boxes[i].tolist()
                confidence = float(scores[i])
                class_id = int(classes[i])
                
                detection = {
                    "bbox": bbox,
                    "confidence": confidence,
                    "class_name": class_names.get(class_id, f"local_artifact_class_{class_id}"),
                    "artifacts": {
                        "class_id": class_id,
                        "type": "local_artifact",
                        "detection_method": "HADM-L",
                        "bbox_area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    }
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

        detections = []
        for i in range(len(boxes)):
            if scores[i] > 0.5:  # Confidence threshold
                bbox = boxes[i].tolist()
                confidence = float(scores[i])
                class_id = int(classes[i])
                
                detection = {
                    "bbox": bbox,
                    "confidence": confidence,
                    "class_name": class_names.get(class_id, f"global_artifact_class_{class_id}"),
                    "artifacts": {
                        "class_id": class_id,
                        "type": "global_artifact",
                        "detection_method": "HADM-G",
                        "bbox_area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    }
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
    # Ensure worker is initialized in this process (lazy init after fork)
    ensure_worker_initialized()
    
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

    # Ensure worker is initialized in this process (lazy init after fork)
    ensure_worker_initialized()

    try:
        command = command_data.get('command')
        # Suppress noisy status logs
        if command != 'status':
            logger.info(f"Handling control command: {command}")

        if command == 'load_l':
            if hadm_l_model is None:
                logger.info("[CONTROL] Loading HADM-L model on explicit request...")
                base_path = BASE_DIR / "pretrained_models"
                hadm_l_weights = base_path / "HADM-L_0249999.pth"
                hadm_l_config = "projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py"
                hadm_l_model = load_hadm_model(hadm_l_config, str(hadm_l_weights), "HADM-L")
                logger.info("[CONTROL] HADM-L model loading command completed")
                return {"success": True, "message": "HADM-L model loaded successfully"}
            else:
                logger.info("[CONTROL] HADM-L model already loaded, skipping")
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
                logger.info("[CONTROL] Loading HADM-G model on explicit request...")
                base_path = BASE_DIR / "pretrained_models"
                hadm_g_weights = base_path / "HADM-G_0249999.pth"
                hadm_g_config = "projects/ViTDet/configs/eva2_o365_to_coco/demo_global.py"
                hadm_g_model = load_hadm_model(hadm_g_config, str(hadm_g_weights), "HADM-G")
                logger.info("[CONTROL] HADM-G model loading command completed")
                return {"success": True, "message": "HADM-G model loaded successfully"}
            else:
                logger.info("[CONTROL] HADM-G model already loaded, skipping")
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

def ensure_worker_initialized():
    """Ensure worker is initialized in the worker process (lazy initialization)"""
    global worker_initialized
    
    if not worker_initialized:
        logger.info("First job in worker process, initializing worker environment...")
        
        # Initialize device (but don't load models automatically)
        get_device()
        
        # Don't auto-load models here - let explicit commands handle model loading
        # This prevents race conditions and duplicate loading attempts
        logger.info("Worker initialized. Models will be loaded on explicit commands.")
        
        worker_initialized = True


# Initialize worker globally


def initialize_worker(load_on_startup: bool = True):
    """Initialize worker with heavy imports and models"""
    logger.info("Initializing HADM Inference Worker...")

    # Load heavy imports (but don't initialize CUDA yet)
    load_heavy_imports()

    # Don't initialize device or load models in the parent process
    # This will be done in the worker process after forking
    if load_on_startup:
        logger.info("Models will be loaded after worker fork to avoid CUDA multiprocessing issues")
    else:
        logger.info("Skipping model loading on startup as requested.")


def main():
    """Main worker function"""
    import multiprocessing
    
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # This must be done before any CUDA operations
    try:
        multiprocessing.set_start_method('spawn', force=True)
        logger.info("Set multiprocessing start method to 'spawn' for CUDA compatibility")
    except RuntimeError as e:
        # If already set, that's fine
        logger.info(f"Multiprocessing start method already set: {e}")
    
    logger.info("Starting HADM Inference Worker...")

    # Check for --unload argument
    load_on_startup = '--unload' not in sys.argv
    
    # Initialize worker
    initialize_worker(load_on_startup=load_on_startup)

    # Connect to Redis
    redis_conn = redis.Redis(host='localhost', port=6379, db=0)

    logger.info("Worker ready, listening for jobs...")

    # Start worker - RQ will handle the job dispatching
    worker = Worker(['inference_q', 'control_q'], connection=redis_conn)
    worker.work()


if __name__ == '__main__':
    main()
