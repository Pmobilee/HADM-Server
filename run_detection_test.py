#!/usr/bin/env python3
"""
HADM Detection Test Script
Sends a local image to the running HADM server for artifact detection.
"""

import requests
import argparse
import time

# --- Configuration ---
# You can change these default values
DEFAULT_IMAGE_PATH = "test_images/woman_bathroom_high.png"
SERVER_URL = "http://localhost:8080/detect"
# --- End Configuration ---

def run_detection(image_path: str, mode: str):
    """
    Sends an image to the HADM server for detection and prints the results.

    Args:
        image_path (str): The local path to the image file.
        mode (str): The detection mode ('local', 'global', 'both').
    """
    print(f"▶️  Starting detection for: {image_path}")
    print(f"    Mode: {mode}")
    print(f"    Server: {SERVER_URL}")

    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path, f, "image/png")}
            params = {"mode": mode}
            
            start_time = time.time()
            response = requests.post(SERVER_URL, files=files, params=params)
            end_time = time.time()

            duration = end_time - start_time
            print(f"✅ Request completed in {duration:.2f} seconds.")

            if response.status_code == 200:
                print("✔️  Successfully received response from server.")
                results = response.json()
                
                print("\n" + "="*20 + " RESULTS " + "="*20)
                if results.get("success"):
                    local_dets = results.get("local_detections", [])
                    global_dets = results.get("global_detections", [])
                    
                    print(f"Found {len(local_dets)} local artifact(s).")
                    for i, det in enumerate(local_dets):
                        print(f"  - Local Artifact {i+1}:")
                        print(f"    Class: {det['class_name']}")
                        print(f"    Confidence: {det['confidence']:.2f}")
                        print(f"    BBox: {det['bbox']}")

                    print(f"\nFound {len(global_dets)} global artifact(s).")
                    for i, det in enumerate(global_dets):
                        print(f"  - Global Artifact {i+1}:")
                        print(f"    Class: {det['class_name']}")
                        print(f"    Confidence: {det['confidence']:.2f}")
                        print(f"    BBox: {det['bbox']}")
                else:
                    print(f"❗️ Server reported failure: {results.get('message')}")
                print("="*49 + "\n")

            else:
                print(f"❌ Error: Server returned status code {response.status_code}")
                try:
                    # Try to print the error detail from the server's JSON response
                    error_detail = response.json().get("detail", response.text)
                    print(f"   Detail: {error_detail}")
                except requests.exceptions.JSONDecodeError:
                    print(f"   Response: {response.text}")

    except FileNotFoundError:
        print(f"❌ Error: Image file not found at '{image_path}'")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: Could not connect to the server at {SERVER_URL}.")
        print(f"   Please make sure the server is running.")
        print(f"   Error details: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HADM Detection Test Client")
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        default=DEFAULT_IMAGE_PATH,
        help=f"Path to the image file. Defaults to {DEFAULT_IMAGE_PATH}",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["local", "global", "both"],
        help="Detection mode to use.",
    )
    args = parser.parse_args()
    
    run_detection(args.image_path, args.mode) 