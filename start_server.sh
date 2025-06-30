#!/bin/bash

# HADM Enhanced Server Startup Script
# This script manages the unified FastAPI server with complete system setup
# Complete replacement for start_unified.sh with first-run setup capabilities

set -e

# Configuration
VENV_DIR="./venv"
SETUP_COMPLETE_FILE="./.system_setup_complete"

# Load port from .env file if it exists
if [ -f ".env" ]; then
    WEB_PORT=$(grep "^SERVER_PORT=" .env | cut -d'=' -f2 | tr -d ' ')
    if [ -z "$WEB_PORT" ]; then
        WEB_PORT="8080"
    fi
else
    WEB_PORT="8080"
fi

# --- Log Directory and PID Files Setup ---
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
DATE_STR=$(date +%Y-%m-%d)

API_PID_FILE="$LOG_DIR/hadm_api.pid"
API_LOG_FILE="$LOG_DIR/api-${DATE_STR}.log"
SETUP_LOG_FILE="$LOG_DIR/system-setup-${DATE_STR}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_setup() {
    echo -e "${CYAN}[SETUP]${NC} $1"
}

# Function to check if a process is running
is_process_running() {
    local pid_file=$1
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        else
            rm -f "$pid_file"
            return 1
        fi
    fi
    return 1
}

# Function to attempt sudo elevation
try_sudo() {
    print_setup "Attempting to elevate privileges with sudo..."
    if sudo -n true 2>/dev/null; then
        print_status "Already have sudo privileges"
        return 0
    elif sudo -v 2>/dev/null; then
        print_status "Successfully elevated to sudo"
        return 0
    else
        print_warning "Could not obtain sudo privileges, continuing without sudo"
        print_warning "Some system packages may not be installable"
        return 1
    fi
}

# Function to install system dependencies
install_system_dependencies() {
    local use_sudo=$1
    local sudo_cmd=""
    
    if [ "$use_sudo" = "true" ]; then
        sudo_cmd="sudo"
    fi
    
    print_setup "Installing system dependencies..."
    
    # Update package lists
    print_setup "Updating package lists..."
    if ! $sudo_cmd apt update >> "$SETUP_LOG_FILE" 2>&1; then
        print_warning "Failed to update package lists (continuing anyway)"
    fi
    
    # Upgrade existing packages
    print_setup "Upgrading existing packages..."
    if ! $sudo_cmd apt upgrade -y >> "$SETUP_LOG_FILE" 2>&1; then
        print_warning "Failed to upgrade packages (continuing anyway)"
    fi
    
    # Install required packages
    print_setup "Installing required development packages..."
    local packages=(
        "build-essential"
        "cmake"
        "git"
        "curl"
        "wget"
        "unzip"
        "pkg-config"
        "libjpeg-dev"
        "libpng-dev"
        "libtiff-dev"
        "libavcodec-dev"
        "libavformat-dev"
        "libswscale-dev"
        "libv4l-dev"
        "libxvidcore-dev"
        "libx264-dev"
        "libgtk-3-dev"
        "libfreetype6-dev"
        "libgl1-mesa-glx"
        "libglib2.0-0"
        "python3-dev"
        "python3-pip"
    )
    
    if $sudo_cmd apt install -y "${packages[@]}" >> "$SETUP_LOG_FILE" 2>&1; then
        print_status "System dependencies installed successfully"
        touch "$SETUP_COMPLETE_FILE"
        return 0
    else
        print_warning "Some system packages failed to install, check $SETUP_LOG_FILE for details"
        print_warning "Continuing with server startup..."
        return 1
    fi
}

# Function to perform system setup if needed
perform_system_setup() {
    if [ -f "$SETUP_COMPLETE_FILE" ]; then
        print_status "System setup already completed, skipping..."
        return 0
    fi
    
    print_setup "Performing first-time system setup..."
    
    # Try to get sudo privileges
    local has_sudo=false
    if try_sudo; then
        has_sudo=true
    fi
    
    # Install system dependencies
    install_system_dependencies "$has_sudo"
}

# Function to forcefully stop all processes on port
force_stop_port_processes() {
    local port=$1
    local stopped=0
    
    print_status "Checking for processes on port $port..."
    
    # Kill processes using lsof if available
    if command -v lsof &>/dev/null; then
        local pids=$(lsof -t -i:$port 2>/dev/null || true)
        if [ -n "$pids" ]; then
            for pid in $pids; do
                print_warning "Force killing process on port $port (PID: $pid)"
                kill -9 "$pid" 2>/dev/null || true
                stopped=1
            done
        fi
    fi
    
    # Kill processes using ss/netstat if lsof not available
    if command -v ss &>/dev/null; then
        local pids=$(ss -tulpn | grep ":$port " | grep -oP 'pid=\K[0-9]+' 2>/dev/null || true)
        if [ -n "$pids" ]; then
            for pid in $pids; do
                print_warning "Force killing process on port $port (PID: $pid)"
                kill -9 "$pid" 2>/dev/null || true
                stopped=1
            done
        fi
    fi
    
    if [ $stopped -eq 1 ]; then
        print_status "Cleared processes from port $port"
        sleep 2  # Give the system time to clean up
    fi
}

# Function to stop services (enhanced version)
stop_service() {
    local service_name=$1
    local pid_file=$2
    local port=$3
    
    local stopped=0

    # Attempt to stop via PID file first
    if [ -f "$pid_file" ] && is_process_running "$pid_file"; then
        local pid=$(cat "$pid_file")
        print_status "Stopping $service_name (PID: $pid) via PID file..."
        kill "$pid" &>/dev/null || true
        sleep 2
        if kill -0 "$pid" &>/dev/null 2>&1; then
            print_warning "$service_name did not stop gracefully. Force killing (PID: $pid)."
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pid_file"
        stopped=1
    fi

    # Always force stop any lingering processes on the port
    if [ -n "$port" ]; then
        force_stop_port_processes "$port"
    fi
    
    # Clean up any remaining PID files
    if [ -f "$pid_file" ]; then
        rm -f "$pid_file"
    fi

    if [ $stopped -eq 1 ]; then
        print_status "$service_name stopped."
    else
        print_status "$service_name was not running"
    fi
}

# Function to start unified API server
start_api() {
    local api_args=$1
    print_status "Starting HADM Unified API server... (Args: ${api_args:-none})"
    
    # Activate virtual environment
    if [ -d "$VENV_DIR" ]; then
        source "$VENV_DIR/bin/activate"
    else
        print_warning "Virtual environment not found at $VENV_DIR"
    fi
    
    # Start API server in background
    nohup python api.py $api_args > "$API_LOG_FILE" 2>&1 &
    echo $! > "$API_PID_FILE"
    
    sleep 5 # Give it more time to potentially fail during model loading
    
    if is_process_running "$API_PID_FILE"; then
        print_status "HADM Unified API server started successfully on port $WEB_PORT"
    else
        print_error "Failed to start HADM Unified API server"
        print_warning "Showing last 40 lines of API log ($API_LOG_FILE) for debugging:"
        tail -n 40 "$API_LOG_FILE" 2>/dev/null || print_warning "No log file found"
        exit 1
    fi
}

# Function to show status
show_status() {
    echo -e "${BLUE}=== HADM Enhanced Server Status ===${NC}"
    
    if is_process_running "$API_PID_FILE"; then
        echo -e "HADM Unified API: ${GREEN}RUNNING${NC} (PID: $(cat $API_PID_FILE))"
        echo -e "Web Dashboard: ${BLUE}http://localhost:$WEB_PORT/dashboard${NC}"
        echo -e "API Documentation: ${BLUE}http://localhost:$WEB_PORT/docs${NC}"
    else
        echo -e "HADM Unified API: ${RED}STOPPED${NC}"
    fi
    
    if [ -f "$SETUP_COMPLETE_FILE" ]; then
        echo -e "System Setup: ${GREEN}COMPLETE${NC}"
    else
        echo -e "System Setup: ${YELLOW}PENDING${NC}"
    fi
}

# Function to show logs
show_logs() {
    local lines=${1:-50}
    if [ -f "$API_LOG_FILE" ]; then
        print_status "Showing last $lines lines of API log:"
        tail -n "$lines" "$API_LOG_FILE"
    else
        print_warning "API log file not found: $API_LOG_FILE"
    fi
}

# Function to show setup logs
show_setup_logs() {
    local lines=${1:-50}
    if [ -f "$SETUP_LOG_FILE" ]; then
        print_status "Showing last $lines lines of setup log:"
        tail -n "$lines" "$SETUP_LOG_FILE"
    else
        print_warning "Setup log file not found: $SETUP_LOG_FILE"
    fi
}

# Main script logic
case "${1:-start}" in
    start)
        print_status "Starting HADM Enhanced Server (with pre-loaded models)..."
        perform_system_setup
        print_status "Ensuring clean shutdown of any existing services..."
        stop_service "HADM Unified API" "$API_PID_FILE" "$WEB_PORT"
        start_api ""
        echo
        show_status
        ;;
    --lazy)
        print_status "Starting HADM Enhanced Server in lazy mode (models load on demand)..."
        perform_system_setup
        print_status "Ensuring clean shutdown of any existing services..."
        stop_service "HADM Unified API" "$API_PID_FILE" "$WEB_PORT"
        start_api "--lazy"
        echo
        show_status
        ;;
    stop)
        print_status "Stopping HADM Enhanced Server..."
        stop_service "HADM Unified API" "$API_PID_FILE" "$WEB_PORT"
        ;;
    restart)
        print_status "Restarting HADM Enhanced Server..."
        $0 stop
        sleep 2
        $0 start
        ;;
    restart-lazy)
        print_status "Restarting HADM Enhanced Server in lazy mode..."
        $0 stop
        sleep 2
        $0 --lazy
        ;;
    setup)
        print_status "Running system setup only..."
        perform_system_setup
        ;;
    reset-setup)
        print_status "Resetting system setup flag..."
        rm -f "$SETUP_COMPLETE_FILE"
        print_status "Setup flag reset. Next start will re-run system setup."
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "${2:-50}"
        ;;
    setup-logs)
        show_setup_logs "${2:-50}"
        ;;
    force-clean)
        print_status "Force cleaning all processes and files..."
        stop_service "HADM Unified API" "$API_PID_FILE" "$WEB_PORT"
        force_stop_port_processes "$WEB_PORT"
        rm -f "$API_PID_FILE"
        print_status "Force clean completed."
        ;;
    *)
        echo "Usage: $0 {start|--lazy|stop|restart|restart-lazy|setup|reset-setup|status|logs [lines]|setup-logs [lines]|force-clean}"
        echo
        echo "Commands:"
        echo "  start         - Start the unified server (pre-loads heavy imports and models)"
        echo "  --lazy        - Start the unified server in lazy mode (loads imports/models on demand)"
        echo "  stop          - Stop the unified server"
        echo "  restart       - Restart the unified server"
        echo "  restart-lazy  - Restart the unified server in lazy mode"
        echo "  setup         - Run system setup only (install dependencies)"
        echo "  reset-setup   - Reset setup flag to force re-running system setup"
        echo "  status        - Show server status"
        echo "  logs [lines]  - Show server logs (default: 50 lines)"
        echo "  setup-logs [lines] - Show system setup logs (default: 50 lines)"
        echo "  force-clean   - Force stop all processes and clean up"
        echo
        echo "Features:"
        echo "  - Automatic system dependency installation on first run"
        echo "  - Automatic sudo elevation (continues without if failed)"
        echo "  - Always stops existing services before starting"
        echo "  - Force kills any processes hogging ports"
        echo "  - Enhanced logging and status reporting"
        echo
        echo "Access points:"
        echo "  Web Dashboard: http://localhost:$WEB_PORT/dashboard"
        echo "  API Docs:      http://localhost:$WEB_PORT/docs"
        exit 1
        ;;
esac 