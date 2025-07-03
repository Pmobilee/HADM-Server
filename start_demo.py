#!/usr/bin/env python3
"""
HADM Visualization Demo Web Server
Launches a web-based visualization demo for HADM-L and HADM-G models on port 8188
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import base64
import io
from typing import Optional, List
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if we're in the correct directory
if not Path("demo").exists():
    logger.error(
        "demo directory not found. Please run this script from the root of the HADM project.")
    sys.exit(1)

# Add the current directory to Python path so we can import from demo
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "demo"))

# Import detectron2 and related modules
try:
    import torch
    import numpy as np
    from PIL import Image
    import cv2
    from detectron2.config import LazyConfig
    from detectron2.engine.defaults import DefaultPredictor
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import instantiate
    from omegaconf import OmegaConf

    # Import the demo predictor directly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "predictor", "demo/predictor.py")
    predictor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor_module)
    VisualizationDemo = predictor_module.VisualizationDemo

    logger.info("Successfully imported all required modules")

except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error(
        "Make sure you have detectron2 and all dependencies installed")
    sys.exit(1)

# Global variables for models
hadm_l_demo: Optional[VisualizationDemo] = None
hadm_g_demo: Optional[VisualizationDemo] = None
device = None

# FastAPI app
app = FastAPI(
    title="HADM Visualization Demo",
    description="Web-based visualization demo for HADM-L and HADM-G models",
    version="1.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


def load_hadm_model(config_path: str, model_path: str, model_name: str):
    """Load a HADM model using LazyConfig"""
    try:
        logger.info(f"Loading {model_name} model from {model_path}...")

        # Check if files exist
        if not Path(config_path).exists():
            logger.error(f"Config file not found: {config_path}")
            return None

        if not Path(model_path).exists():
            logger.error(f"Model file not found: {model_path}")
            return None

        # Load LazyConfig
        cfg = LazyConfig.load(config_path)
        logger.info(f"Config loaded successfully for {model_name}")

        # Create demo wrapper using the config
        demo = VisualizationDemo(
            cfg, instance_mode=ColorMode.IMAGE, parallel=False)

        # Load the model checkpoint
        logger.info(f"Loading checkpoint from {model_path}...")
        checkpoint_data = torch.load(
            model_path, map_location=device, weights_only=False)

        # Update the predictor's model with the checkpoint
        if "model" in checkpoint_data:
            demo.predictor.model.load_state_dict(checkpoint_data["model"])
        else:
            logger.error(
                f"Checkpoint for {model_name} does not contain a 'model' key!")
            return None

        demo.predictor.model.to(device)
        demo.predictor.model.eval()

        logger.info(f"{model_name} model loaded successfully")
        return demo

    except Exception as e:
        logger.error(f"Failed to load {model_name} model: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def initialize_models():
    """Initialize HADM models"""
    global hadm_l_demo, hadm_g_demo, device

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU")

    # Setup detectron2 logger
    setup_logger()

    # Load HADM-L model
    hadm_l_path = Path(os.getenv("HADM_L_MODEL_PATH",
                       "pretrained_models/HADM-L_0249999.pth"))
    if hadm_l_path.exists():
        hadm_l_config = "projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py"
        hadm_l_demo = load_hadm_model(
            hadm_l_config, str(hadm_l_path), "HADM-L")
    else:
        logger.warning(f"HADM-L model not found at {hadm_l_path}")

    # Load HADM-G model
    hadm_g_path = Path(os.getenv("HADM_G_MODEL_PATH",
                       "pretrained_models/HADM-G_0249999.pth"))
    if hadm_g_path.exists():
        hadm_g_config = "projects/ViTDet/configs/eva2_o365_to_coco/demo_global.py"
        hadm_g_demo = load_hadm_model(
            hadm_g_config, str(hadm_g_path), "HADM-G")
    else:
        logger.warning(f"HADM-G model not found at {hadm_g_path}")

    if hadm_l_demo is None and hadm_g_demo is None:
        logger.error(
            "No models could be loaded. Please check your model paths.")
        sys.exit(1)


def process_image(image_data: bytes, model_type: str = "both"):
    """Process image with specified model(s)"""
    try:
        # Load image
        image = Image.open(io.BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to numpy array in BGR format (OpenCV format)
        image_array = np.array(image)
        image_bgr = image_array[:, :, ::-1]  # RGB to BGR

        results = {}

        # Process with HADM-L
        if model_type in ["local", "both"] and hadm_l_demo is not None:
            try:
                predictions_l, vis_output_l = hadm_l_demo.run_on_image(
                    image_bgr)

                # Convert visualization to base64
                vis_image_l = vis_output_l.get_image()
                vis_pil_l = Image.fromarray(vis_image_l)
                buffer_l = io.BytesIO()
                vis_pil_l.save(buffer_l, format="PNG")
                vis_base64_l = base64.b64encode(buffer_l.getvalue()).decode()

                results["local"] = {
                    "image": vis_base64_l,
                    "detections": len(predictions_l["instances"]) if "instances" in predictions_l else 0
                }

            except Exception as e:
                logger.error(f"HADM-L processing failed: {e}")
                results["local"] = {"error": str(e)}

        # Process with HADM-G
        if model_type in ["global", "both"] and hadm_g_demo is not None:
            try:
                predictions_g, vis_output_g = hadm_g_demo.run_on_image(
                    image_bgr)

                # Convert visualization to base64
                vis_image_g = vis_output_g.get_image()
                vis_pil_g = Image.fromarray(vis_image_g)
                buffer_g = io.BytesIO()
                vis_pil_g.save(buffer_g, format="PNG")
                vis_base64_g = base64.b64encode(buffer_g.getvalue()).decode()

                results["global"] = {
                    "image": vis_base64_g,
                    "detections": len(predictions_g["instances"]) if "instances" in predictions_g else 0
                }

            except Exception as e:
                logger.error(f"HADM-G processing failed: {e}")
                results["global"] = {"error": str(e)}

        return results

    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        return {"error": str(e)}

# Routes


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main demo page"""
    return templates.TemplateResponse("demo.html", {
        "request": request,
        "hadm_l_available": hadm_l_demo is not None,
        "hadm_g_available": hadm_g_demo is not None
    })


@app.post("/detect")
async def detect_artifacts(
    file: UploadFile = File(...),
    model_type: str = Form(default="both")
):
    """Process uploaded image and return detection results"""

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Validate model type
    if model_type not in ["local", "global", "both"]:
        raise HTTPException(
            status_code=400, detail="Model type must be 'local', 'global', or 'both'")

    # Check if requested models are available
    if model_type in ["local", "both"] and hadm_l_demo is None:
        raise HTTPException(
            status_code=503, detail="HADM-L model not available")

    if model_type in ["global", "both"] and hadm_g_demo is None:
        raise HTTPException(
            status_code=503, detail="HADM-G model not available")

    try:
        # Read image data
        image_data = await file.read()

        # Process image
        results = process_image(image_data, model_type)

        return JSONResponse(content=results)

    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Detection failed: {str(e)}")


@app.get("/status")
async def get_status():
    """Get model status"""
    return {
        "hadm_l_loaded": hadm_l_demo is not None,
        "hadm_g_loaded": hadm_g_demo is not None,
        "device": str(device) if device else "not_set",
        "cuda_available": torch.cuda.is_available()
    }

# Create HTML template if it doesn't exist


def create_demo_template():
    """Create the demo HTML template"""
    template_path = Path("templates/demo.html")
    if not template_path.exists():
        html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>HADM Visualization Demo</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .upload-section { margin: 20px 0; padding: 20px; border: 2px dashed #ccc; border-radius: 10px; text-align: center; }
        .upload-section.dragover { border-color: #007bff; background-color: #f0f8ff; }
        .model-selection { margin: 20px 0; }
        .model-selection label { margin-right: 15px; }
        .results { margin: 20px 0; }
        .result-item { margin: 10px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .result-image { max-width: 100%; height: auto; border-radius: 5px; }
        .error { color: red; }
        .success { color: green; }
        .loading { color: blue; }
        .status { margin: 20px 0; padding: 10px; background: #e9ecef; border-radius: 5px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        button:disabled { background: #ccc; cursor: not-allowed; }
    </style>
</head>
<body>
    <div class="container">
        <h1>HADM Visualization Demo</h1>
        <p>Upload an image to detect human artifacts using HADM-L (local) and HADM-G (global) models.</p>
        
        <div class="status">
            <h3>Model Status:</h3>
            <p>HADM-L (Local): {% if hadm_l_available %}<span class="success">✓ Available</span>{% else %}<span class="error">✗ Not Available</span>{% endif %}</p>
            <p>HADM-G (Global): {% if hadm_g_available %}<span class="success">✓ Available</span>{% else %}<span class="error">✗ Not Available</span>{% endif %}</p>
        </div>
        
        <div class="upload-section" id="uploadSection">
            <p>Drag and drop an image here or click to select</p>
            <input type="file" id="fileInput" accept="image/*" style="display: none;">
            <button onclick="document.getElementById('fileInput').click()">Select Image</button>
        </div>
        
        <div class="model-selection">
            <h3>Select Model:</h3>
            <label><input type="radio" name="model" value="both" checked> Both Models</label>
            {% if hadm_l_available %}<label><input type="radio" name="model" value="local"> HADM-L (Local) Only</label>{% endif %}
            {% if hadm_g_available %}<label><input type="radio" name="model" value="global"> HADM-G (Global) Only</label>{% endif %}
        </div>
        
        <button id="processBtn" onclick="processImage()" disabled>Process Image</button>
        
        <div id="results" class="results"></div>
    </div>
    
    <script>
        let selectedFile = null;
        
        // File input handling
        document.getElementById('fileInput').addEventListener('change', function(e) {
            selectedFile = e.target.files[0];
            updateUI();
        });
        
        // Drag and drop handling
        const uploadSection = document.getElementById('uploadSection');
        uploadSection.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });
        
        uploadSection.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
        });
        
        uploadSection.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            selectedFile = e.dataTransfer.files[0];
            updateUI();
        });
        
        function updateUI() {
            const processBtn = document.getElementById('processBtn');
            if (selectedFile) {
                processBtn.disabled = false;
                uploadSection.innerHTML = '<p>Selected: ' + selectedFile.name + '</p><button onclick="document.getElementById(\\'fileInput\\').click()">Change Image</button>';
            } else {
                processBtn.disabled = true;
            }
        }
        
        async function processImage() {
            if (!selectedFile) return;
            
            const modelType = document.querySelector('input[name="model"]:checked').value;
            const resultsDiv = document.getElementById('results');
            
            // Show loading
            resultsDiv.innerHTML = '<div class="loading">Processing image...</div>';
            
            // Prepare form data
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('model_type', modelType);
            
            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Detection failed');
                }
                
                const results = await response.json();
                displayResults(results);
                
            } catch (error) {
                resultsDiv.innerHTML = '<div class="error">Error: ' + error.message + '</div>';
            }
        }
        
        function displayResults(results) {
            const resultsDiv = document.getElementById('results');
            let html = '<h3>Detection Results:</h3>';
            
            if (results.local) {
                html += '<div class="result-item">';
                html += '<h4>HADM-L (Local) Results:</h4>';
                if (results.local.error) {
                    html += '<div class="error">Error: ' + results.local.error + '</div>';
                } else {
                    html += '<p>Detections: ' + results.local.detections + '</p>';
                    html += '<img src="data:image/png;base64,' + results.local.image + '" class="result-image">';
                }
                html += '</div>';
            }
            
            if (results.global) {
                html += '<div class="result-item">';
                html += '<h4>HADM-G (Global) Results:</h4>';
                if (results.global.error) {
                    html += '<div class="error">Error: ' + results.global.error + '</div>';
                } else {
                    html += '<p>Detections: ' + results.global.detections + '</p>';
                    html += '<img src="data:image/png;base64,' + results.global.image + '" class="result-image">';
                }
                html += '</div>';
            }
            
            if (results.error) {
                html += '<div class="error">Error: ' + results.error + '</div>';
            }
            
            resultsDiv.innerHTML = html;
        }
    </script>
</body>
</html>'''

        with open(template_path, 'w') as f:
            f.write(html_content)

        logger.info("Created demo HTML template")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="HADM Visualization Demo Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8188,
                        help="Port to bind to")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload")
    args = parser.parse_args()

    logger.info("Starting HADM Visualization Demo...")

    # Create template if it doesn't exist
    create_demo_template()

    # Initialize models
    logger.info("Initializing models...")
    initialize_models()

    logger.info(f"Starting web server on {args.host}:{args.port}")
    logger.info(f"Open your browser to http://localhost:{args.port}")

    # Run the server
    uvicorn.run(
        "start_demo:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
