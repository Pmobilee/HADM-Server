# HADM Server Optimization and Refactoring Plan

This document outlines the step-by-step plan to refactor the HADM server into a more robust, decoupled, and user-friendly application.

## High-Level Goals

1.  **Decouple Services**: Separate the web interface (frontend) from the model serving (backend) to allow for independent restarts and faster UI loading.
2.  **Enhance User Interface**: Implement a professional, dark-themed dashboard with authentication, real-time diagnostics, and full control over the backend services.
3.  **Robust Task Handling**: Use a proper task queue to manage inference requests asynchronously, preventing the UI from freezing and allowing for better scalability.
4.  **Simplified Orchestration**: Create a single shell script to manage the lifecycle of all application components (Redis, backend, frontend).

---

## Part 1: Architecture Refactoring

The new architecture will consist of three core components:

1.  **Web Server (`web_api.py`):** A lightweight FastAPI application.
    *   **Responsibilities**: Serving HTML pages (login, dashboard), handling user authentication, providing API endpoints for the frontend to interact with, and submitting jobs to the task queue.
    *   **Note**: This server will **not** load any models or heavy libraries like `torch` or `detectron2` into its main process, ensuring a near-instant startup time.

2.  **Inference Server (`inference_worker.py`):** A background Python worker process.
    *   **Responsibilities**: Listening for jobs on the Redis queue. It will load, manage, and hold the HADM-L and HADM-G models in VRAM. It executes inference tasks and posts results back to Redis. It also handles control commands (e.g., load/unload models).

3.  **Redis Server & RQ (Redis Queue):** The message broker connecting the two services.
    *   **Responsibilities**: Manages the queue of inference jobs, holds job results, and passes control messages from the Web Server to the Inference Server.

---

## Part 2: Step-by-Step Implementation Plan

### Step 1: Setup Environment and Redis

1.  **Install Dependencies**: Add `redis` and `rq` to the project's requirements.
    ```bash
    pip install redis rq
    ```
2.  **Install Redis**: Ensure a Redis server is installed. On Debian/Ubuntu:
    ```bash
    sudo apt-get update
    sudo apt-get install redis-server
    ```
3.  **Verify Redis**: Check that Redis is running.
    ```bash
    redis-cli ping  # Should return PONG
    ```

### Step 2: Create the Backend Inference Worker

1.  **Create `inference_worker.py`**: This new file will contain all the model-related logic.
    *   Move the `load_models`, `load_hadm_model`, `run_hadm_l_inference`, `run_hadm_g_inference` functions from `api.py` into this file.
    *   Remove all FastAPI-related code from this file.
    *   The worker will connect to Redis and listen on a queue (e.g., `inference_q`).
    *   It will be a simple loop: `listen -> get job -> process job -> store result -> repeat`.

2.  **Implement Job Processing**:
    *   Define a function, e.g., `execute_inference(image_data, mode)`, that gets called for each job.
    *   This function will run the appropriate model (`hadm_l_model` or `hadm_g_model`) on the image data.

3.  **Implement Control Commands**:
    *   The worker will also listen on a separate `control_q` queue.
    *   Define functions to handle commands like `load_l`, `unload_l`, `load_g`, `unload_g`, `status`.
    *   The `status` command will return the current VRAM usage, loaded models, etc., by posting the information back to Redis.

### Step 3: Refactor the FastAPI Application

1.  **Rename `api.py` to `web_api.py`**: To better reflect its new role.

2.  **Remove Model Logic**: Strip out all model loading and inference code that was moved to `inference_worker.py`. Remove the heavy imports (`torch`, `detectron2`, etc.) from the top-level scope to ensure fast startup. These will be imported dynamically only if needed for specific utility functions, or not at all.

3.  **Implement Authentication**:
    *   Create a simple login page template (`login.html`).
    *   Add a `/login` endpoint (both GET for the page and POST for form submission).
    *   Hardcode user/pass: `intelligents` / `intelligentsintelligents`.
    *   Use FastAPI's `Depends` with security schemes (e.g., OAuth2 with password flow) to protect the main dashboard page.

4.  **Create the Dashboard**:
    *   Create the main page template (`dashboard.html`). Style it with a dark/gray theme and accent colors.
    *   **Diagnostics View**:
        *   Create a `/api/diagnostics` endpoint in `web_api.py`.
        *   This endpoint will send a `status` command to the `inference_worker` via Redis and wait for the response.
        *   The frontend will call this endpoint periodically (e.g., every 2 seconds using JavaScript's `setInterval`) to display real-time VRAM usage, loaded models, and worker status.
    *   **Control Buttons**:
        *   Add buttons to the dashboard: "Load HADM-L", "Unload HADM-L", "Load HADM-G", "Unload HADM-G", "Restart Worker".
        *   Each button will trigger a JavaScript function that calls a corresponding API endpoint on `web_api.py` (e.g., `/api/control/load_l`).
        *   These API endpoints will simply place a control command onto the `control_q` in Redis for the worker to pick up.
    *   **Inference Interface**:
        *   Keep the file upload form.
        *   When a user uploads an image, the `/detect` endpoint in `web_api.py` will **not** run inference directly.
        *   Instead, it will enqueue a job to the `inference_q` in Redis, passing the image data and mode. It will immediately return a `job_id` to the user.
        *   The frontend will then use this `job_id` to poll another endpoint (`/api/results/{job_id}`) until the results are ready.

### Step 4: Create the Orchestration Script

1.  **Create `start_optimized.sh`**:
    *   This script will be the single point of entry to run the application.
    *   It will use `pgrep` or PID files to check if processes are already running.
    *   **Script Logic**:
        1.  Check if Redis server is active. If not, start it.
        2.  Check if `inference_worker.py` is running. If not, start it as a background process using `rq worker inference_q control_q --with-scheduler &`. Save its PID.
        3.  Check if `web_api.py` is running. If not, start it using `uvicorn web_api:app --host 0.0.0.0 --port 8080 &`. Save its PID.
        4.  Include `stop.sh` and `restart.sh` logic or separate scripts that read the PID files to kill and restart the services.

2.  **Refine `start_optimized.sh`**:
    *   Add flags to allow for selectively restarting components, e.g., `./start_optimized.sh --restart-web`.
    *   The script should not restart the inference worker unless explicitly told to, preserving the loaded models in VRAM across web server restarts.

---
## Part 3: UI Design Mockup

**Theme**: Dark/Gray (`#1a1a1a`, `#2b2b2b`) with a pleasant accent color (e.g., a muted teal `#4db6ac` or amber `#ffc107`). Font should be clean (e.g., Inter, Roboto).

**Layout**:
*   **Header**: "HADM Control Center"
*   **Main Grid (2-column)**:
    *   **Left Column (Control & Status)**:
        *   **Diagnostics Box**:
            *   Worker Status: `RUNNING` / `STOPPED`
            *   VRAM Usage: `XX.XX / YY.YY GB` (Progress Bar)
            *   HADM-L Status: `LOADED` / `UNLOADED`
            *   HADM-G Status: `LOADED` / `UNLOADED`
        *   **Model Control Box**:
            *   Buttons for Load/Unload for each model.
        *   **Service Control Box**:
            *   Buttons for Restart Worker, Restart Web API.
    *   **Right Column (Inference)**:
        *   **Uploader Box**: Drag-and-drop file upload area. Mode selection dropdown. "Detect" button.
        *   **Results Box**: Initially empty. After detection, it will show the output image with bounding boxes and the JSON results. A loading spinner will be shown while waiting for results.

This detailed plan provides a clear roadmap for the refactoring process.
