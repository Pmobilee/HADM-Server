#!/usr/bin/env python3
"""
Environment setup script for HADM FastAPI Server
Creates venv, installs dependencies, and downloads models
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command and return the result"""
    print(f"Running: {cmd}")
    result = subprocess.run(
        cmd, shell=True, check=check, capture_output=True, text=True
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result


def create_venv():
    """Create Python virtual environment"""
    print("Creating Python virtual environment...")

    if os.path.exists("venv"):
        print("Virtual environment already exists")
        return

    run_command("python3 -m venv venv")
    print("Virtual environment created successfully")


def install_dependencies():
    """Install all required dependencies"""
    print("Installing dependencies...")

    # Activate venv and install PyTorch first
    pip_cmd = "./venv/bin/pip" if os.name != "nt" else "venv\\Scripts\\pip"

    # Install PyTorch with CUDA support
    print("Installing PyTorch with CUDA support...")
    run_command(
        f"{pip_cmd} install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116"
    )

    # Install other dependencies
    print("Installing other dependencies...")
    run_command(f"{pip_cmd} install cryptography")
    run_command(f"{pip_cmd} install -r requirements.txt")

    # Install xformers
    print("Installing xformers...")
    run_command(
        f"{pip_cmd} install -v -U git+https://github.com/facebookresearch/xformers.git@v0.0.18#egg=xformers"
    )

    # Install mmcv
    print("Installing mmcv...")
    run_command(f"{pip_cmd} install mmcv==1.7.1 openmim")
    run_command(
        f"./venv/bin/mim install mmcv-full"
        if os.name != "nt"
        else "venv\\Scripts\\mim install mmcv-full"
    )

    # Install FastAPI and related packages
    print("Installing FastAPI and server dependencies...")
    run_command(f"{pip_cmd} install fastapi uvicorn python-multipart aiofiles")

    # Install detectron2
    print("Installing detectron2...")
    python_cmd = "./venv/bin/python" if os.name != "nt" else "venv\\Scripts\\python"
    run_command(f"{python_cmd} -m pip install -e .")


def create_directories():
    """Create necessary directories"""
    print("Creating directories...")

    directories = ["pretrained_models", "test_images", "outputs", "cache"]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")


def download_models():
    """Download pretrained models"""
    print("Downloading pretrained models...")

    models = [
        {
            "name": "EVA-02-L",
            "url": "https://huggingface.co/Yuxin-CV/EVA-02/resolve/main/eva02/det/eva02_L_coco_det_sys_o365.pth",
            "filename": "eva02_L_coco_det_sys_o365.pth",
        },
        {
            "name": "HADM-L",
            "url": "https://www.dropbox.com/scl/fi/zwasvod906x1akzinnj3i/HADM-L_0249999.pth?rlkey=bqz5517tm8yt8l6ngzne4xejx&st=k1a1gzph&dl=1",
            "filename": "HADM-L_0249999.pth",
        },
        {
            "name": "HADM-G",
            "url": "https://www.dropbox.com/scl/fi/bzj1m8p4cvm2vg4mai6uj/HADM-G_0249999.pth?rlkey=813x6wraigivc6qx02aut9p2r&st=n8rnb47r&dl=1",
            "filename": "HADM-G_0249999.pth",
        },
    ]

    for model in models:
        filepath = Path("pretrained_models") / model["filename"]
        if filepath.exists():
            print(f"{model['name']} already exists, skipping download")
            continue

        print(f"Downloading {model['name']}...")
        try:
            urllib.request.urlretrieve(model["url"], filepath)
            print(f"Successfully downloaded {model['name']}")
        except Exception as e:
            print(f"Failed to download {model['name']}: {e}")
            print("You may need to download this model manually")


def main():
    """Main setup function"""
    print("Setting up HADM FastAPI Server environment...")

    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)

    try:
        create_venv()
        install_dependencies()
        create_directories()
        download_models()

        print("\n" + "=" * 50)
        print("Environment setup completed successfully!")
        print("=" * 50)
        print("\nTo activate the environment, run:")
        if os.name != "nt":
            print("source venv/bin/activate")
        else:
            print("venv\\Scripts\\activate")
        print("\nTo start the API server, run:")
        print("python api.py")

    except Exception as e:
        print(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
