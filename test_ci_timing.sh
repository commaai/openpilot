#!/bin/bash

# CI Timing Test Script for OpenPilot
# Tests setup time and execution time for various CI components
# Based on the GitHub workflow in .github/workflows/selfdrive_tests.yaml

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to time execution
time_command() {
    local description="$1"
    local command="$2"
    local start_time=$(date +%s.%N)
    
    log "Starting: $description"
    
    if eval "$command"; then
        local end_time=$(date +%s.%N)
        local duration=$(echo "scale=2; $end_time - $start_time" | bc -l 2>/dev/null || echo "$(($end_time - $start_time))")
        success "$description completed in ${duration}s"
        return 0
    else
        local end_time=$(date +%s.%N)
        local duration=$(echo "scale=2; $end_time - $start_time" | bc -l 2>/dev/null || echo "$(($end_time - $start_time))")
        error "$description failed after ${duration}s"
        return 1
    fi
}

# Function to check system requirements
check_system() {
    log "Checking system requirements..."
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version)
        success "Python: $python_version"
    else
        error "Python3 not found"
        return 1
    fi
    
    # Check pytest
    if python3 -m pytest --version &> /dev/null; then
        local pytest_version=$(python3 -m pytest --version)
        success "Pytest: $pytest_version"
    else
        warning "Pytest not available - will affect unit tests"
    fi
    
    # Check available memory
    if command -v free &> /dev/null; then
        local memory=$(free -h | grep '^Mem:' | awk '{print $2}')
        success "Available memory: $memory"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        local memory=$(system_profiler SPHardwareDataType | grep "Memory:" | awk '{print $2" "$3}')
        success "Available memory: $memory"
    fi
    
    # Check CPU info
    if [[ "$OSTYPE" == "darwin"* ]]; then
        local cpu_count=$(sysctl -n hw.ncpu)
        success "CPU cores: $cpu_count"
    else
        local cpu_count=$(nproc)
        success "CPU cores: $cpu_count"
    fi
}

# Function to simulate CI environment setup
setup_ci_environment() {
    log "Setting up CI environment simulation..."
    
    # Set environment variables similar to CI
    export PYTHONWARNINGS=error
    export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    
    # Ultra-fast direct execution (no Docker) - similar to CI
    export RUN="/bin/bash -c"
    
    # Optimized pytest with fast collection - similar to CI
    export PYTEST="python3 -m pytest --continue-on-collection-errors --durations=0 --durations-min=5 --cache-clear"
    
    success "CI environment variables set"
}

# Function to test dependency installation/checking
test_dependency_check() {
    time_command "Dependency check" "
        # Check if virtual environment exists
        if [ -d '.venv' ]; then
            echo 'Virtual environment found'
        else
            echo 'No virtual environment found'
        fi
        
        # Check Python packages
        python3 -c 'import numpy; print(f\"NumPy version: {numpy.__version__}\")' 2>/dev/null || echo 'NumPy not available'
        python3 -c 'import pytest; print(f\"Pytest version: {pytest.__version__}\")' 2>/dev/null || echo 'Pytest not available'
    "
}

# Function to test pytest collection speed
test_pytest_collection() {
    if ! python3 -m pytest --version &> /dev/null; then
        warning "Skipping pytest collection test - pytest not available"
        return 0
    fi
    
    time_command "Pytest test collection" "
        python3 -m pytest --collect-only --quiet common/ 2>/dev/null || echo 'Collection completed with warnings'
    "
}

# Function to test unit tests execution
test_unit_tests() {
    if ! python3 -m pytest --version &> /dev/null; then
        warning "Skipping unit tests - pytest not available"
        return 0
    fi
    
    time_command "Unit tests execution (common/)" "
        python3 -m pytest common/ --tb=no -q
    "
}

# Function to test specific transformation tests
test_transformation_tests() {
    if ! python3 -m pytest --version &> /dev/null; then
        warning "Skipping transformation tests - pytest not available"
        return 0
    fi
    
    time_command "Transformation tests (coordinate)" "
        python3 -m pytest common/transformations/tests/test_coordinates.py --tb=no -q
    "
    
    time_command "Transformation tests (orientation)" "
        python3 -m pytest common/transformations/tests/test_orientation.py --tb=no -q
    "
}

# Function to simulate static analysis
test_static_analysis() {
    if command -v flake8 &> /dev/null; then
        time_command "Static analysis (flake8)" "
            flake8 common/transformations/transformations.py --count --select=E9,F63,F7,F82 --show-source --statistics
        "
    else
        time_command "Static analysis (syntax check)" "
            python3 -m py_compile common/transformations/transformations.py openpilot/common/transformations/transformations.py
        "
    fi
}

# Function to simulate build process
test_build_simulation() {
    time_command "Build simulation (import test)" "
        python3 -c '
import sys
sys.path.insert(0, \".\")
try:
    from openpilot.common.transformations import coordinates, orientation
    print(\"Successfully imported transformation modules\")
    
    # Test basic functionality
    import numpy as np
    result = coordinates.geodetic2ecef([0, 0, 0])
    print(f\"Geodetic conversion test: {result.shape}\")
    
    result2 = orientation.euler2quat([0, 0, 0])
    print(f\"Orientation conversion test: {result2.shape}\")
    
except Exception as e:
    print(f\"Import/functionality test failed: {e}\")
    exit(1)
        '
    "
}

# Function to generate timing report
generate_report() {
    local total_time="$1"
    
    echo ""
    echo "========================================"
    echo "        CI TIMING REPORT"
    echo "========================================"
    echo "Total execution time: ${total_time}s"
    echo "Timestamp: $(date)"
    echo "System: $(uname -s) $(uname -r)"
    echo "Python: $(python3 --version 2>&1)"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "CPU: $(sysctl -n machdep.cpu.brand_string)"
        echo "Cores: $(sysctl -n hw.ncpu)"
    else
        echo "CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)"
        echo "Cores: $(nproc)"
    fi
    
    echo "========================================"
    echo ""
}

# Main execution
main() {
    local script_start_time=$(date +%s.%N)
    
    echo "=========================================="
    echo "    OpenPilot CI Timing Test Script"
    echo "=========================================="
    echo ""
    
    # Run all tests
    check_system
    echo ""
    
    setup_ci_environment
    echo ""
    
    test_dependency_check
    echo ""
    
    test_pytest_collection
    echo ""
    
    test_unit_tests
    echo ""
    
    test_transformation_tests
    echo ""
    
    test_static_analysis
    echo ""
    
    test_build_simulation
    echo ""
    
    # Calculate total time
    local script_end_time=$(date +%s.%N)
    local total_duration=$(echo "scale=2; $script_end_time - $script_start_time" | bc -l 2>/dev/null || echo "$(($script_end_time - $script_start_time))")
    
    generate_report "$total_duration"
    
    success "CI timing test completed successfully!"
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h          Show this help message"
        echo "  --unit-tests-only   Run only unit tests timing"
        echo "  --static-only       Run only static analysis timing"
        echo "  --system-check      Run only system requirement checks"
        echo ""
        echo "This script benchmarks various CI operations to help optimize"
        echo "GitHub Actions workflow timing."
        ;;
    --unit-tests-only)
        setup_ci_environment
        test_unit_tests
        ;;
    --static-only)
        test_static_analysis
        ;;
    --system-check)
        check_system
        ;;
    *)
        main
        ;;
esac
