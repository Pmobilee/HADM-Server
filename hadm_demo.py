#!/usr/bin/env python3
"""
HADM Demo Script
Runs HADM-L and HADM-G inference on images using LazyConfig
"""

from omegaconf import ListConfig, DictConfig
import torch.serialization
from omegaconf import OmegaConf
from detectron2.data.detection_utils import read_image
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.logger import setup_logger
from detectron2.config import LazyConfig, instantiate
import os
import sys
import argparse
import time
import glob
from pathlib import Path
import torch
import numpy as np
import cv2
from PIL import Image

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

# Import detectron2 modules

# Register OmegaConf classes as safe globals for torch.load
torch.serialization.add_safe_globals([ListConfig, DictConfig, OmegaConf])


def load_hadm_model(config_path: str, model_path: str, model_name: str, device):
    """Load a HADM model using LazyConfig"""
    print(f"Loading {model_name} model...")
    print(f"Config: {config_path}")
    print(f"Model: {model_path}")

    # Load LazyConfig
    cfg = LazyConfig.load(config_path)

    # Clean up any device references in model config
    if hasattr(cfg.model, "device"):
        delattr(cfg.model, "device")

    # Instantiate model
    model = instantiate(cfg.model)
    model.to(device)
    model.eval()

    # Load checkpoint
    print(f"Loading checkpoint from {model_path}...")
    checkpoint_data = torch.load(
        model_path, map_location=device, weights_only=False)

    if "model" in checkpoint_data:
        checkpointer = DetectionCheckpointer(model)
        checkpointer._load_model(checkpoint_data)
    else:
        print(f"ERROR: Checkpoint does not contain 'model' key!")
        return None

    print(f"{model_name} model loaded successfully")
    return model, cfg


def run_inference(model, cfg, image_path, model_name, device):
    """Run inference on an image"""
    print(f"Processing {image_path} with {model_name}...")

    # Read image
    img = read_image(image_path, format="BGR")

    # Prepare input
    height, width = img.shape[:2]

    # Simple preprocessing - resize to 1024x1024 as expected by ViTDet
    from detectron2.data import transforms as T
    aug = T.ResizeShortestEdge(short_edge_length=1024, max_size=1024)
    image = aug.get_transform(img).apply_image(img)

    # Convert to tensor
    image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = {"image": image_tensor, "height": height, "width": width}

    # Run inference
    with torch.no_grad():
        predictions = model([inputs])

    # Visualize results
    img_rgb = img[:, :, ::-1]  # BGR to RGB

    # Get metadata for visualization
    if model_name == "HADM-L":
        # Local artifact classes
        class_names = ["face", "torso", "arm", "leg", "hand", "feet"]
    else:
        # Global artifact classes
        class_names = [
            "human missing arm", "human missing face", "human missing feet",
            "human missing hand", "human missing leg", "human missing torso",
            "human with extra arm", "human with extra face", "human with extra feet",
            "human with extra hand", "human with extra leg", "human with extra torso"
        ]

    # Create custom metadata
    from detectron2.data.catalog import Metadata
    metadata = Metadata()
    metadata.set(thing_classes=class_names)

    visualizer = Visualizer(img_rgb, metadata=metadata,
                            instance_mode=ColorMode.IMAGE)

    if "instances" in predictions[0]:
        instances = predictions[0]["instances"].to("cpu")
        vis_output = visualizer.draw_instance_predictions(
            predictions=instances)

        # Print detection results
        num_instances = len(instances)
        print(f"Found {num_instances} detections")

        if num_instances > 0:
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()

            for i in range(num_instances):
                class_name = class_names[classes[i]] if classes[i] < len(
                    class_names) else f"class_{classes[i]}"
                print(f"  {i+1}: {class_name} (confidence: {scores[i]:.3f})")
    else:
        vis_output = visualizer.output
        print("No detections found")

    return vis_output, predictions[0]


def main():
    parser = argparse.ArgumentParser(description="HADM Demo")
    parser.add_argument("--input", required=True,
                        help="Input image path or directory")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--model", choices=["local", "global", "both"], default="both",
                        help="Model to use")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold")

    args = parser.parse_args()

    # Setup logging
    setup_logger()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model paths
    hadm_l_path = os.getenv("HADM_L_MODEL_PATH",
                            "pretrained_models/HADM-L_0249999.pth")
    hadm_g_path = os.getenv("HADM_G_MODEL_PATH",
                            "pretrained_models/HADM-G_0249999.pth")

    # Config paths
    hadm_l_config = "projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py"
    hadm_g_config = "projects/ViTDet/configs/eva2_o365_to_coco/demo_global.py"

    # Load models
    models = {}

    if args.model in ["local", "both"]:
        if Path(hadm_l_path).exists() and Path(hadm_l_config).exists():
            result = load_hadm_model(
                hadm_l_config, hadm_l_path, "HADM-L", device)
            if result:
                models["HADM-L"] = result
        else:
            print(f"Warning: HADM-L model or config not found")

    if args.model in ["global", "both"]:
        if Path(hadm_g_path).exists() and Path(hadm_g_config).exists():
            result = load_hadm_model(
                hadm_g_config, hadm_g_path, "HADM-G", device)
            if result:
                models["HADM-G"] = result
        else:
            print(f"Warning: HADM-G model or config not found")

    if not models:
        print("ERROR: No models could be loaded!")
        return

    # Get input files
    input_path = Path(args.input)
    if input_path.is_file():
        input_files = [str(input_path)]
    elif input_path.is_dir():
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        input_files = []
        for ext in extensions:
            input_files.extend(glob.glob(str(input_path / ext)))
            input_files.extend(glob.glob(str(input_path / ext.upper())))
    else:
        print(f"ERROR: Input path {args.input} not found")
        return

    if not input_files:
        print("ERROR: No image files found")
        return

    print(f"Found {len(input_files)} image(s) to process")

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    for image_path in input_files:
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")

        for model_name, (model, cfg) in models.items():
            try:
                vis_output, predictions = run_inference(
                    model, cfg, image_path, model_name, device)

                # Save result if output directory specified
                if args.output:
                    base_name = Path(image_path).stem
                    output_path = output_dir / \
                        f"{base_name}_{model_name.lower()}_result.jpg"
                    vis_output.save(str(output_path))
                    print(f"Saved result to: {output_path}")
                else:
                    # Save to default output directory if no output specified
                    default_output = Path("demo_output")
                    default_output.mkdir(exist_ok=True)
                    base_name = Path(image_path).stem
                    output_path = default_output / \
                        f"{base_name}_{model_name.lower()}_result.jpg"
                    vis_output.save(str(output_path))
                    print(f"Saved result to: {output_path}")
                    print("(Use --output to specify a different output directory)")

            except Exception as e:
                print(f"ERROR processing {image_path} with {model_name}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*60}")
    print("Processing complete!")


if __name__ == "__main__":
    main()
