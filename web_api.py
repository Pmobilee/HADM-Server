#!/usr/bin/env python3
"""
HADM Web API Server
Lightweight FastAPI application that handles authentication, dashboard, and job submission.
This server does NOT load any heavy ML libraries to ensure fast startup.
"""

import os
import sys
import time
import json
import base64
import logging
import asyncio
import uuid
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Any, List

import redis
from rq import Queue
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends, Form, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="HADM Control Center",
    description="Human Artifact Detection Model Control Center",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Redis connection
redis_conn = redis.Redis(host='localhost', port=6379, db=0)
inference_queue = Queue('inference_q', connection=redis_conn)
control_queue = Queue('control_q', connection=redis_conn)

# Authentication
security = HTTPBasic()
ADMIN_USERNAME = "intelligents"
ADMIN_PASSWORD = "intelligentsintelligents"

# Pydantic models
class DetectionResult(BaseModel):
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_name: str
    artifacts: Dict[str, Any]

class InferenceResponse(BaseModel):
    success: bool
    message: str
    job_id: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    local_detections: Optional[List[DetectionResult]] = None
    global_detections: Optional[List[DetectionResult]] = None
    processing_time: Optional[float] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class ControlCommand(BaseModel):
    command: str

# Authentication functions
def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify HTTP Basic Auth credentials"""
    if credentials.username != ADMIN_USERNAME or credentials.password != ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to login page"""
    return RedirectResponse(url="/login")

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve login page"""
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login form submission"""
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
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
    
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection
        redis_conn.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/api/diagnostics")
async def get_diagnostics(username: str = Depends(verify_credentials)):
    """Get system diagnostics"""
    try:
        # Send status command to worker
        job = control_queue.enqueue('inference_worker.handle_control_command', {'command': 'status'})
        
        # Wait for result (with timeout)
        timeout = 10  # seconds
        start_time = time.time()
        while job.result is None and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)
        
        if job.result is None:
            return {
                "success": False,
                "message": "Timeout waiting for worker response",
                "worker_status": "UNKNOWN"
            }
        
        result = job.result
        if result.get("success"):
            status_info = result.get("status", {})
            return {
                "success": True,
                "worker_status": "RUNNING",
                "hadm_l_loaded": status_info.get("hadm_l_loaded", False),
                "hadm_g_loaded": status_info.get("hadm_g_loaded", False),
                "device": status_info.get("device", "Unknown"),
                "vram_info": status_info.get("vram_info", {})
            }
        else:
            return {
                "success": False,
                "message": result.get("message", "Unknown error"),
                "worker_status": "ERROR"
            }
    
    except Exception as e:
        logger.error(f"Error getting diagnostics: {str(e)}")
        return {
            "success": False,
            "message": f"Error: {str(e)}",
            "worker_status": "ERROR"
        }

@app.post("/api/control/{command}")
async def control_command(command: str, username: str = Depends(verify_credentials)):
    """Send control command to worker"""
    try:
        valid_commands = ['load_l', 'unload_l', 'load_g', 'unload_g', 'status']
        if command not in valid_commands:
            raise HTTPException(status_code=400, detail=f"Invalid command. Valid commands: {valid_commands}")
        
        # Send command to worker
        job = control_queue.enqueue('inference_worker.handle_control_command', {'command': command})
        
        # Wait for result (with timeout)
        timeout = 30  # seconds for model loading
        start_time = time.time()
        while job.result is None and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)
        
        if job.result is None:
            return {"success": False, "message": "Timeout waiting for worker response"}
        
        return job.result
    
    except Exception as e:
        logger.error(f"Error executing control command: {str(e)}")
        return {"success": False, "message": f"Error: {str(e)}"}

@app.post("/api/detect", response_model=InferenceResponse)
async def detect_artifacts(
    file: UploadFile = File(...), 
    mode: str = "both",
    username: str = Depends(verify_credentials)
):
    """Submit inference job to queue"""
    try:
        # Validate mode
        valid_modes = ["local", "global", "both"]
        if mode not in valid_modes:
            raise HTTPException(status_code=400, detail=f"Invalid mode. Valid modes: {valid_modes}")
        
        # Read and encode image
        image_data = await file.read()
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        # Create job data
        job_data = {
            'image_data': image_b64,
            'mode': mode
        }
        
        # Submit job to queue
        job = inference_queue.enqueue('inference_worker.execute_inference', job_data)
        
        return InferenceResponse(
            success=True,
            message="Job submitted successfully",
            job_id=job.id
        )
    
    except Exception as e:
        logger.error(f"Error submitting inference job: {str(e)}")
        return InferenceResponse(
            success=False,
            message=f"Error: {str(e)}"
        )

@app.get("/api/results/{job_id}")
async def get_results(job_id: str, username: str = Depends(verify_credentials)):
    """Get inference results by job ID"""
    try:
        job = inference_queue.fetch_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.result is None:
            # Job is still running
            return {
                "status": "running",
                "message": "Job is still processing"
            }
        
        # Job completed
        result = job.result
        return {
            "status": "completed",
            "result": result
        }
    
    except Exception as e:
        logger.error(f"Error getting job results: {str(e)}")
        return {"status": "error", "message": f"Error: {str(e)}"}

@app.get("/interface", response_class=HTMLResponse)
async def web_interface(request: Request):
    """Legacy interface endpoint - redirect to dashboard"""
    return RedirectResponse(url="/dashboard")

# Legacy endpoints for backward compatibility
@app.post("/detect", response_model=InferenceResponse)
async def legacy_detect_artifacts(file: UploadFile = File(...), mode: str = "both"):
    """Legacy detect endpoint without authentication"""
    try:
        # Validate mode
        valid_modes = ["local", "global", "both"]
        if mode not in valid_modes:
            raise HTTPException(status_code=400, detail=f"Invalid mode. Valid modes: {valid_modes}")
        
        # Read and encode image
        image_data = await file.read()
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        # Create job data
        job_data = {
            'image_data': image_b64,
            'mode': mode
        }
        
        # Submit job to queue
        job = inference_queue.enqueue('inference_worker.execute_inference', job_data)
        
        # Wait for result (for backward compatibility)
        timeout = 60  # seconds
        start_time = time.time()
        while job.result is None and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.5)
        
        if job.result is None:
            return InferenceResponse(
                success=False,
                message="Timeout waiting for inference result"
            )
        
        result = job.result
        return InferenceResponse(
            success=result.get("success", False),
            message=result.get("message", ""),
            local_detections=result.get("local_detections", []),
            global_detections=result.get("global_detections", []),
            processing_time=result.get("processing_time", 0)
        )
    
    except Exception as e:
        logger.error(f"Error in legacy detect endpoint: {str(e)}")
        return InferenceResponse(
            success=False,
            message=f"Error: {str(e)}"
        )

@app.get("/models/status")
async def models_status():
    """Legacy models status endpoint"""
    try:
        # Send status command to worker
        job = control_queue.enqueue('inference_worker.handle_control_command', {'command': 'status'})
        
        # Wait for result
        timeout = 10
        start_time = time.time()
        while job.result is None and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)
        
        if job.result is None:
            return {"status": "error", "message": "Timeout waiting for worker"}
        
        result = job.result
        if result.get("success"):
            status_info = result.get("status", {})
            return {
                "status": "ok",
                "hadm_l_loaded": status_info.get("hadm_l_loaded", False),
                "hadm_g_loaded": status_info.get("hadm_g_loaded", False),
                "device": status_info.get("device", "Unknown")
            }
        else:
            return {"status": "error", "message": result.get("message", "Unknown error")}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(
        "web_api:app",
        host="0.0.0.0",
        port=8080,
        reload=False
    ) 