#!/usr/bin/env python3
"""
HADM Server Quick Launcher
Skips all dependency checks and starts the FastAPI server directly.
"""

import os
import time

def log_time(message):
    """Log message with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def start_server():
    """Start the FastAPI server"""
    log_time("Starting FastAPI server...")
    print("\n" + "=" * 50)
    print("Starting HADM FastAPI Server...")
    print("Server will be available at: http://localhost:8080")
    print("API documentation: http://localhost:8080/docs")
    print("=" * 50)

    try:
        # Start the server
        log_time("Executing: python api.py")
        os.system("python api.py")
    except KeyboardInterrupt:
        log_time("Server stopped by user.")
        print("\nServer stopped by user.")
    except Exception as e:
        log_time(f"Error starting server: {e}")
        print(f"Error starting server: {e}")


def main():
    log_time("HADM Server Quick Launcher started")
    start_server()


if __name__ == "__main__":
    main() 