#!/bin/bash

# HADM Unified Server Startup Script
# This script manages the unified FastAPI server that includes both API and web dashboard

set -e

# Configuration
VENV_DIR="./venv"

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

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Function to start unified API server
start_api() {
    if is_process_running "$API_PID_FILE"; then
        print_status "HADM Unified API server is already running"
        return 0
    fi
    
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
    
    sleep 3 # Give it a moment to potentially fail
    
    if is_process_running "$API_PID_FILE"; then
        print_status "HADM Unified API server started successfully on port $WEB_PORT"
    else
        print_error "Failed to start HADM Unified API server"
        print_warning "Showing last 40 lines of API log ($API_LOG_FILE) for debugging:"
        tail -n 40 "$API_LOG_FILE" 2>/dev/null || print_warning "No log file found"
        exit 1
    fi
}

# Function to stop services
stop_service() {
    local service_name=$1
    local pid_file=$2
    local port=$3
    
    local stopped=0

    # Attempt to stop via PID file first
    if [ -f "$pid_file" ] && is_process_running "$pid_file"; then
        local pid=$(cat "$pid_file")
        print_status "Stopping $service_name (PID: $pid) via PID file..."
        kill "$pid" &>/dev/null
        sleep 2
        if kill -0 "$pid" &>/dev/null; then
            print_warning "$service_name did not stop gracefully. Force killing (PID: $pid)."
            kill -9 "$pid"
        fi
        rm -f "$pid_file"
        stopped=1
    fi

    # Forcefully stop any process lingering on the port as a fallback
    if [ -n "$port" ]; then
        local lingering_pid=""
        if command -v lsof &>/dev/null; then
            lingering_pid=$(lsof -t -i:$port 2>/dev/null)
        elif command -v ss &>/dev/null; then
            lingering_pid=$(ss -tulpn | grep ":$port " | grep -oP 'pid=\K[0-9]+')
        fi

        if [ -n "$lingering_pid" ]; then
            for pid in $lingering_pid; do
                print_warning "$service_name is still running on port $port (PID: $pid). Forcefully stopping."
                kill -9 "$pid"
                stopped=1
            done
        fi
    fi
    
    # Final check and cleanup
    if [ -f "$pid_file" ] && ! is_process_running "$pid_file"; then
        rm -f "$pid_file"
    fi

    if [ $stopped -eq 1 ]; then
        print_status "$service_name stopped."
    else
        print_status "$service_name is not running"
    fi
}

# Function to show status
show_status() {
    echo -e "${BLUE}=== HADM Unified Server Status ===${NC}"
    
    if is_process_running "$API_PID_FILE"; then
        echo -e "HADM Unified API: ${GREEN}RUNNING${NC} (PID: $(cat $API_PID_FILE))"
        echo -e "Web Dashboard: ${BLUE}http://localhost:$WEB_PORT/dashboard${NC}"
        echo -e "API Documentation: ${BLUE}http://localhost:$WEB_PORT/docs${NC}"
    else
        echo -e "HADM Unified API: ${RED}STOPPED${NC}"
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

# Main script logic
case "${1:-start}" in
    start)
        print_status "Starting HADM Unified Server..."
        start_api ""
        echo
        show_status
        ;;
    --lazy)
        print_status "Starting HADM Unified Server in lazy mode (models load on demand)..."
        start_api "--lazy"
        echo
        show_status
        ;;
    stop)
        print_status "Stopping HADM Unified Server..."
        stop_service "HADM Unified API" "$API_PID_FILE" "$WEB_PORT"
        ;;
    restart)
        print_status "Restarting HADM Unified Server..."
        $0 stop
        sleep 2
        $0 start
        ;;
    restart-lazy)
        print_status "Restarting HADM Unified Server in lazy mode..."
        $0 stop
        sleep 2
        $0 --lazy
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "${2:-50}"
        ;;
    *)
        echo "Usage: $0 {start|--lazy|stop|restart|restart-lazy|status|logs [lines]}"
        echo
        echo "Commands:"
        echo "  start         - Start the unified server (pre-loads heavy imports)"
        echo "  --lazy        - Start the unified server in lazy mode (loads imports on demand)"
        echo "  stop          - Stop the unified server"
        echo "  restart       - Restart the unified server"
        echo "  restart-lazy  - Restart the unified server in lazy mode"
        echo "  status        - Show server status"
        echo "  logs [lines]  - Show server logs (default: 50 lines)"
        echo
        echo "Access points:"
        echo "  Web Dashboard: http://localhost:$WEB_PORT/dashboard"
        echo "  API Docs:      http://localhost:$WEB_PORT/docs"
        echo "  Login:         intelligents / intelligentsintelligents"
        exit 1
        ;;
esac 