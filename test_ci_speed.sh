#!/bin/bash

# CI Speed Test Script
# Tests various CI optimization methods and reports performance

set -e

echo "🚀 CI Speed Test Starting..."
echo "============================"

# Test function
test_setup_method() {
    local method_name="$1"
    local script_path="$2"
    
    if [ ! -f "$script_path" ]; then
        echo "⚠️  $method_name: Script not found ($script_path)"
        return 1
    fi
    
    # Clean environment
    rm -rf ~/.venv /tmp/.openpilot_* 2>/dev/null || true
    
    # Time the setup
    echo "Testing $method_name..."
    start_time=$(python3 -c "import time; print(time.time())")
    
    if timeout 60 bash "$script_path" >/dev/null 2>&1; then
        end_time=$(python3 -c "import time; print(time.time())")
        duration=$(python3 -c "print(f'{$end_time - $start_time:.3f}')")
        
        if (( $(python3 -c "print(int($duration < 20.0))") )); then
            if (( $(python3 -c "print(int($duration < 1.0))") )); then
                echo "🎯 TARGET ACHIEVED: $method_name completed in ${duration}s"
                status="TARGET"
            else
                echo "✅ $method_name completed in ${duration}s"
                status="SUCCESS" 
            fi
        else
            echo "⚠️  $method_name took ${duration}s (over target)"
            status="SLOW"
        fi
        
        echo "$method_name,$duration,$status" >> /tmp/ci_speed_results.csv
        return 0
    else
        echo "❌ $method_name failed"
        echo "$method_name,FAILED,FAILED" >> /tmp/ci_speed_results.csv
        return 1
    fi
}

# Initialize results file
echo "Method,Duration,Status" > /tmp/ci_speed_results.csv

# Test available setup methods
echo "📊 Testing CI Setup Methods"
echo "----------------------------"

# Test 1: Parallel setup
test_setup_method "Parallel CI Setup" "./setup_parallel_ci.sh"

# Test 2: Check if other methods exist
if [ -f "./setup_nano_ci.sh" ]; then
    test_setup_method "Nano CI Setup" "./setup_nano_ci.sh"
fi

if [ -f "./setup_minimal_ci.sh" ]; then
    test_setup_method "Minimal CI Setup" "./setup_minimal_ci.sh"
fi

# Results summary
echo ""
echo "📈 Speed Test Results Summary"
echo "============================"

if [ -f "/tmp/ci_speed_results.csv" ]; then
    # Show results
    cat /tmp/ci_speed_results.csv
    
    # Check if target was achieved
    if grep -q "TARGET" /tmp/ci_speed_results.csv; then
        best_time=$(grep "TARGET" /tmp/ci_speed_results.csv | head -1 | cut -d',' -f2)
        best_method=$(grep "TARGET" /tmp/ci_speed_results.csv | head -1 | cut -d',' -f1)
        echo ""
        echo "🎯 TARGET ACHIEVED!"
        echo "Best Method: $best_method in ${best_time}s"
        echo "Performance improvement: $(python3 -c "print(f'{((60.0 - $best_time) / 60.0) * 100:.1f}%')")% faster than baseline"
    else
        echo ""
        echo "⚠️  Target not achieved, but improvements made"
        # Show best performing method
        if grep -q "SUCCESS" /tmp/ci_speed_results.csv; then
            best_line=$(grep "SUCCESS" /tmp/ci_speed_results.csv | sort -t',' -k2 -n | head -1)
            best_method=$(echo "$best_line" | cut -d',' -f1)
            best_time=$(echo "$best_line" | cut -d',' -f2)
            echo "Best Method: $best_method in ${best_time}s"
        fi
    fi
fi

echo ""
echo "✅ CI Speed Test Completed"
