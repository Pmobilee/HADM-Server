#!/usr/bin/env python3
"""
Standalone image artifact checker

This script allows you to check any image file for artifacts using the same
artifact detection API that the main application uses.

Usage:
    python check_image.py <image_path>
    python check_image.py image.jpg
    python check_image.py /full/path/to/image.png

The script will:
1. Load the image file
2. Convert it to base64
3. Send it to the artifact detection API
4. Display all detection results with confidence scores
"""

import os
import sys
import base64
import json
from PIL import Image
from io import BytesIO
import argparse

# Import from the main API module
from api import detect_artifacts, has_artifacts_above_threshold, ARTIFACT_CONFIDENCE_THRESHOLD


def load_and_convert_image(image_path: str) -> str:
    """
    Load an image file and convert it to base64.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        str: Base64 encoded image
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image and convert to RGB
        with Image.open(image_path) as img:
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to JPEG format in memory
            jpeg_buffer = BytesIO()
            img.save(jpeg_buffer, format="JPEG", quality=95)
            image_data = jpeg_buffer.getvalue()
            
            # Encode to base64
            encoded_image = base64.b64encode(image_data).decode("utf-8")
            return encoded_image
            
    except Exception as e:
        raise Exception(f"Error loading image: {str(e)}")


# detect_artifacts function is now imported from api.py


def analyze_results(detection_result: dict, threshold: float = ARTIFACT_CONFIDENCE_THRESHOLD):
    """
    Analyze and display artifact detection results.
    
    Args:
        detection_result: The API response from artifact detection
        threshold: Minimum confidence threshold for highlighting artifacts
    """
    print("\n" + "=" * 80)
    print("ARTIFACT DETECTION RESULTS")
    print("=" * 80)
    
    # Basic info
    success = detection_result.get("success", False)
    message = detection_result.get("message", "No message")
    processing_time = detection_result.get("processing_time", 0)
    
    print(f"Status: {'✓ SUCCESS' if success else '✗ FAILED'}")
    print(f"Message: {message}")
    print(f"Processing Time: {processing_time:.2f} seconds")
    print()
    
    # Local detections
    local_detections = detection_result.get("local_detections", [])
    print(f"LOCAL DETECTIONS ({len(local_detections)} found):")
    print("-" * 40)
    
    if local_detections:
        local_above_threshold = 0
        for i, detection in enumerate(local_detections, 1):
            confidence = detection.get("confidence", 0)
            class_name = detection.get("class_name", "unknown")
            bbox = detection.get("bbox", [])
            
            status = "⚠️  ABOVE THRESHOLD" if confidence >= threshold else "✓ below threshold"
            if confidence >= threshold:
                local_above_threshold += 1
            
            print(f"  {i}. {class_name}")
            print(f"     Confidence: {confidence:.3f} ({status})")
            if bbox:
                print(f"     Bounding Box: {bbox}")
            print()
        
        print(f"Local detections above threshold ({threshold}): {local_above_threshold}")
    else:
        print("  No local detections found")
    
    print()
    
    # Global detections
    global_detections = detection_result.get("global_detections", [])
    print(f"GLOBAL DETECTIONS ({len(global_detections)} found):")
    print("-" * 40)
    
    if global_detections:
        global_above_threshold = 0
        for i, detection in enumerate(global_detections, 1):
            confidence = detection.get("confidence", 0)
            class_name = detection.get("class_name", "unknown")
            bbox = detection.get("bbox", [])
            
            status = "⚠️  ABOVE THRESHOLD" if confidence >= threshold else "✓ below threshold"
            if confidence >= threshold:
                global_above_threshold += 1
            
            print(f"  {i}. {class_name}")
            print(f"     Confidence: {confidence:.3f} ({status})")
            if bbox:
                print(f"     Bounding Box: {bbox}")
            print()
        
        print(f"Global detections above threshold ({threshold}): {global_above_threshold}")
    else:
        print("  No global detections found")
    
    print()
    
    # Summary
    total_above_threshold = len([d for d in local_detections if d.get("confidence", 0) >= threshold]) + \
                           len([d for d in global_detections if d.get("confidence", 0) >= threshold])
    
    print("SUMMARY:")
    print("-" * 40)
    print(f"Total detections: {len(local_detections) + len(global_detections)}")
    print(f"Detections above threshold ({threshold}): {total_above_threshold}")
    
    if total_above_threshold > 0:
        print(f"🚨 RESULT: Image would FAIL artifact check (retry would be triggered)")
        
        # List artifacts that would cause failure
        failing_artifacts = []
        for detection in local_detections:
            if detection.get("confidence", 0) >= threshold:
                failing_artifacts.append(f"{detection.get('class_name', 'unknown')} (confidence: {detection.get('confidence', 0):.2f})")
        for detection in global_detections:
            if detection.get("confidence", 0) >= threshold:
                failing_artifacts.append(f"{detection.get('class_name', 'unknown')} (confidence: {detection.get('confidence', 0):.2f})")
        
        print(f"Failing artifacts: {', '.join(failing_artifacts)}")
    else:
        print(f"✅ RESULT: Image would PASS artifact check")
    
    print("=" * 80)


def main():
    """Main function to handle command line arguments and run the check."""
    parser = argparse.ArgumentParser(
        description="Check an image for artifacts using the artifact detection API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_image.py image.jpg
  python check_image.py /path/to/image.png
  python check_image.py --threshold 0.2 image.jpg
        """
    )
    
    parser.add_argument("image_path", help="Path to the image file to check")
    parser.add_argument(
        "--threshold", "-t", 
        type=float, 
        default=ARTIFACT_CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for artifact detection (default: {ARTIFACT_CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--save-results", "-s",
        help="Save full API response to JSON file"
    )
    
    args = parser.parse_args()
    
    try:
        # Validate threshold
        if not 0.0 <= args.threshold <= 1.0:
            print("Error: Threshold must be between 0.0 and 1.0")
            sys.exit(1)
        
        print(f"Checking image: {args.image_path}")
        print(f"Using threshold: {args.threshold}")
        print()
        
        # Load and convert image
        print("Loading image...")
        image_base64 = load_and_convert_image(args.image_path)
        print(f"Image loaded successfully (size: {len(image_base64)} characters)")
        
        # Detect artifacts
        print("Sending image to artifact detection API...")
        detection_result = detect_artifacts(image_base64)
        
        # Analyze and display results
        analyze_results(detection_result, args.threshold)
        
        # Save results if requested
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(detection_result, f, indent=2)
            print(f"\nFull API response saved to: {args.save_results}")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 