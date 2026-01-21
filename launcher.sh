#!/bin/bash
#
# DPInterview Pipeline Launcher
# Launches each pipeline runner in its own tmux session
#

set -e

# Configuration
PROJECT_ROOT="/home/dpinterview"
CONFIG_FILE="/home/dpinterview/config.ini"
VENV_PATH="/home/linlab/cpp_venv"
WEB_ROOT="/home/dpinterview-web"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    log_error "tmux is not installed. Please install it first:"
    echo "  sudo apt-get install tmux"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    log_error "Virtual environment not found at: $VENV_PATH"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    log_error "Config file not found at: $CONFIG_FILE"
    exit 1
fi

# Function to create and launch a tmux session for a runner
launch_runner() {
    local session_name=$1
    local script_path=$2
    local description=$3
    
    # Check if session already exists
    if tmux has-session -t "$session_name" 2>/dev/null; then
        log_warn "Session '$session_name' already exists. Skipping."
        return 1
    fi
    
    log_step "Launching: $description"
    
    # Create new detached session
    tmux new-session -d -s "$session_name"
    
    # Send commands to the session
    tmux send-keys -t "$session_name" "cd $PROJECT_ROOT" C-m
    tmux send-keys -t "$session_name" "source $VENV_PATH/bin/activate" C-m
    tmux send-keys -t "$session_name" "python3 $script_path -c $CONFIG_FILE" C-m
    
    log_info "Started '$session_name'"
    return 0
}

# Function to launch web server
launch_webserver() {
    local session_name="webserver"
    
    # Check if session already exists
    if tmux has-session -t "$session_name" 2>/dev/null; then
        log_warn "Session '$session_name' already exists. Skipping."
        return 1
    fi
    
    log_step "Launching: Web Server"
    
    # Create new detached session
    tmux new-session -d -s "$session_name"
    
    # Send commands to the session
    tmux send-keys -t "$session_name" "cd $PROJECT_ROOT" C-m
    tmux send-keys -t "$session_name" "source $VENV_PATH/bin/activate" C-m
    tmux send-keys -t "$session_name" "python3 pipeline/web/server.py" C-m
    
    log_info "Started '$session_name'"
    return 0
}

# Function to launch web frontend
launch_webfrontend() {
    local session_name="webrun"
    
    # Check if session already exists
    if tmux has-session -t "$session_name" 2>/dev/null; then
        log_warn "Session '$session_name' already exists. Skipping."
        return 1
    fi
    
    log_step "Launching: Web Frontend"
    
    # Create new detached session
    tmux new-session -d -s "$session_name"
    
    # Send commands to the session
    tmux send-keys -t "$session_name" "cd $WEB_ROOT" C-m
    tmux send-keys -t "$session_name" "npm run preview" C-m
    
    log_info "Started '$session_name'"
    return 0
}

# Function to launch both web services
launch_web_services() {
    log_info "Launching web services..."
    echo ""
    
    launch_webserver
    sleep 2
    launch_webfrontend
    
    echo ""
    log_info "Web services launched!"
}

# Function to kill all pipeline sessions
kill_all_sessions() {
    log_warn "Killing all pipeline sessions..."
    
    local sessions=(
        "1_fetch_video"
        "2_importer"
        "3_metadata"
        "4_video_qqc"
        "5_split_streams"
        "6_openface"
        "7_openface_qc"
        "8_load_openface"
        "9_face_pipe"
        "10_report_gen"
        "11_exporter"
        "12_vid_exporter"
        "webserver"
        "webrun"
    )
    
    for session in "${sessions[@]}"; do
        if tmux has-session -t "$session" 2>/dev/null; then
            tmux kill-session -t "$session"
            log_info "Killed session: $session"
        fi
    done
}

# Function to list all pipeline sessions
list_sessions() {
    log_info "Active pipeline sessions:"
    tmux list-sessions 2>/dev/null | grep -E "1_fetch_video|2_importer|3_metadata|4_video_qqc|5_split_streams|6_openface|7_openface_qc|8_load_openface|9_face_pipe|10_report_gen|11_exporter|12v_vid_exporter|webserver|webrun" || echo "  No pipeline sessions running"
}

# Function to attach to a specific session
attach_session() {
    local session_name=$1
    
    if tmux has-session -t "$session_name" 2>/dev/null; then
        log_info "Attaching to session: $session_name"
        tmux attach-session -t "$session_name"
    else
        log_error "Session '$session_name' does not exist"
        list_sessions
        exit 1
    fi
}

# Main menu
show_menu() {
    echo ""
    echo "════════════════════════════════════════"
    echo "  DPInterview Pipeline Launcher"
    echo "════════════════════════════════════════"
    echo ""
    echo "1)  Launch all runners"
    echo "2)  Launch core pipeline (fetch → facepipe)"
    echo "3)  Launch web services only"
    echo "4)  Launch individual runner"
    echo "5)  List active sessions"
    echo "6)  Attach to session"
    echo "7)  Kill all pipeline sessions"
    echo "8)  Exit"
    echo ""
}

# Launch all runners
launch_all() {
    log_info "Launching complete pipeline..."
    echo ""
    
    launch_runner "1_fetch_video" "pipeline/runners/01_fetch_video.py" "Fetch Video"
    sleep 2
    
    launch_runner "2_importer" "pipeline/runners/study_specific/predictor/services/importer.py" "Importer"
    sleep 2
    
    launch_runner "3_metadata" "pipeline/runners/02_metadata.py" "Metadata Extraction"
    sleep 2
    
    launch_runner "4_video_qqc" "pipeline/runners/03_video_qqc.py" "Video Quick QC"
    sleep 2
    
    launch_runner "5_split_streams" "pipeline/runners/04_split_streams.py" "Split Streams"
    sleep 2
    
    launch_runner "6_openface" "pipeline/runners/05_openface.py" "OpenFace Processing"
    sleep 2
    
    launch_runner "7_openface_qc" "pipeline/runners/06_openface_qc.py" "OpenFace QC"
    sleep 2
    
    launch_runner "8_load_openface" "pipeline/runners/08_load_openface.py" "Load OpenFace"
    sleep 2
    
    #launch_runner "9_face_pipe" "pipeline/runners/41_face_pipe.py" "Face Pipe"
    #sleep 2
    
    launch_runner "10_report_gen" "pipeline/runners/70_report_generation.py" "Report Generation"
    sleep 2
    
    launch_runner "11_exporter" "pipeline/runners/study_specific/predictor/services/exporter.py" "Exporter (to NAS)"
    sleep 2

    launch_runner "12_vid_exporter" "pipeline/runners/study_specific/predictor/services/vid_exporter.py" "Early downscaled video exporter (to NAS)"
    sleep 2
    
    
    echo ""
    log_info "All runners launched!"
    echo ""
    list_sessions
}

# Launch core pipeline only
launch_core() {
    log_info "Launching core pipeline (fetch → facepipe)..."
    echo ""
    
    launch_runner "1_fetch_video" "pipeline/runners/01_fetch_video.py" "Fetch Video"
    sleep 2
    launch_runner "2_importer" "pipeline/runners/study_specific/predictor/services/importer.py" "Importer"
    sleep 2
    launch_runner "3_metadata" "pipeline/runners/02_metadata.py" "Metadata"
    sleep 2
    launch_runner "4_video_qqc" "pipeline/runners/03_video_qqc.py" "Video QQC"
    sleep 2
    launch_runner "5_split_streams" "pipeline/runners/04_split_streams.py" "Split Streams"
    sleep 2
    launch_runner "6_openface" "pipeline/runners/05_openface.py" "OpenFace"
    sleep 2
    launch_runner "7_openface_qc" "pipeline/runners/06_openface_qc.py" "OpenFace QC"
    sleep 2
    launch_runner "8_load_openface" "pipeline/runners/08_load_openface.py" "Load OpenFace"
    sleep 2
    launch_runner "9_face_pipe" "pipeline/runners/41_face_pipe.py" "Face Pipe"
    
    echo ""
    log_info "Core pipeline launched!"
    echo ""
    list_sessions
}

# Launch individual runner
launch_individual() {
    echo ""
    echo "Select runner to launch:"
    echo "1)  Fetch Video"
    echo "2)  Importer"
    echo "3)  Metadata Extraction"
    echo "4)  Video Quick QC"
    echo "5)  Split Streams"
    echo "6)  OpenFace"
    echo "7)  OpenFace QC"
    echo "8)  Load OpenFace"
    echo "9)  Face Pipe"
    echo "10) Report Generation"
    echo "11) Exporter"
    echo "12) Vid Exporter"
    echo "13) Web Server"
    echo "14) Web Frontend"
    echo ""
    read -p "Enter choice [1-13]: " runner_choice
    
    case $runner_choice in
        1) launch_runner "1_fetch_video" "pipeline/runners/01_fetch_video.py" "Fetch Video" ;;
        2) launch_runner "2_importer" "pipeline/runners/study_specific/predictor/services/importer.py" "Importer" ;;
        3) launch_runner "3_metadata" "pipeline/runners/02_metadata.py" "Metadata" ;;
        4) launch_runner "4_video_qqc" "pipeline/runners/03_video_qqc.py" "Video QQC" ;;
        5) launch_runner "5_split_streams" "pipeline/runners/04_split_streams.py" "Split Streams" ;;
        6) launch_runner "6_openface" "pipeline/runners/05_openface.py" "OpenFace" ;;
        7) launch_runner "7_openface_qc" "pipeline/runners/06_openface_qc.py" "OpenFace QC" ;;
        8) launch_runner "8_load_openface" "pipeline/runners/08_load_openface.py" "Load OpenFace" ;;
        9) launch_runner "9_face_pipe" "pipeline/runners/41_face_pipe.py" "Face Pipe" ;;
        10) launch_runner "10_report_gen" "pipeline/runners/70_report_generation.py" "Report Generation" ;;
        11) launch_runner "11_exporter" "pipeline/runners/study_specific/predictor/services/exporter.py" "Exporter" ;;
        12) launch_runner "12_vid_exporter" "pipeline/runners/study_specific/predictor/services/vid_exporter.py" "Vid Exporter" ;;
        13) launch_webserver ;;
        14) launch_webfrontend ;;
        *) log_error "Invalid choice" ;;
    esac
}

# Main loop
if [ "$1" == "--kill" ]; then
    kill_all_sessions
    exit 0
elif [ "$1" == "--list" ]; then
    list_sessions
    exit 0
elif [ "$1" == "--attach" ]; then
    if [ -z "$2" ]; then
        log_error "Usage: $0 --attach <session_name>"
        list_sessions
        exit 1
    fi
    attach_session "$2"
    exit 0
elif [ "$1" == "--all" ]; then
    launch_all
    exit 0
elif [ "$1" == "--core" ]; then
    launch_core
    exit 0
elif [ "$1" == "--web" ]; then
    launch_web_services
    exit 0
fi

# Interactive menu
while true; do
    show_menu
    read -p "Enter choice [1-8]: " choice
    
    case $choice in
        1) launch_all ;;
        2) launch_core ;;
        3) launch_web_services ;;
        4) launch_individual ;;
        5) list_sessions ;;
        6)
            echo ""
            read -p "Enter session name to attach: " session_name
            attach_session "$session_name"
            ;;
        7) kill_all_sessions ;;
        8)
            log_info "Exiting..."
            exit 0
            ;;
        *)
            log_error "Invalid choice"
            ;;
    esac
    
    echo ""
done