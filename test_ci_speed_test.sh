#!/bin/bash

# Unit Tests for ci_speed_test.sh
# Simple bash testing framework

set -euo pipefail

# Test configuration
SCRIPT_UNDER_TEST="./ci_speed_test.sh"
TEST_COUNT=0
PASS_COUNT=0
FAIL_COUNT=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test utilities
run_test() {
    local test_name="$1"
    local test_function="$2"
    
    TEST_COUNT=$((TEST_COUNT + 1))
    echo -e "${YELLOW}Running test: ${test_name}${NC}"
    
    if $test_function; then
        echo -e "${GREEN}✅ PASS: ${test_name}${NC}"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo -e "${RED}❌ FAIL: ${test_name}${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo
}

assert_equals() {
    local expected="$1"
    local actual="$2"
    local message="${3:-Assertion failed}"
    
    if [[ "$expected" == "$actual" ]]; then
        return 0
    else
        echo "  $message: expected '$expected', got '$actual'"
        return 1
    fi
}

assert_contains() {
    local haystack="$1"
    local needle="$2"
    local message="${3:-String not found}"
    
    if [[ "$haystack" == *"$needle"* ]]; then
        return 0
    else
        echo "  $message: '$needle' not found in '$haystack'"
        return 1
    fi
}

assert_exit_code() {
    local expected_code="$1"
    local actual_code="$2"
    local message="${3:-Exit code assertion failed}"
    
    if [[ "$expected_code" -eq "$actual_code" ]]; then
        return 0
    else
        echo "  $message: expected exit code $expected_code, got $actual_code"
        return 1
    fi
}

# Test functions
test_script_exists() {
    [[ -f "$SCRIPT_UNDER_TEST" ]] || {
        echo "  Script $SCRIPT_UNDER_TEST does not exist"
        return 1
    }
    return 0
}

test_script_executable() {
    [[ -x "$SCRIPT_UNDER_TEST" ]] || {
        echo "  Script $SCRIPT_UNDER_TEST is not executable"
        return 1
    }
    return 0
}

test_shebang_correct() {
    local first_line
    first_line=$(head -n1 "$SCRIPT_UNDER_TEST")
    assert_equals "#!/bin/bash" "$first_line" "Shebang check"
}

test_shellcheck_passes() {
    if command -v shellcheck >/dev/null 2>&1; then
        shellcheck "$SCRIPT_UNDER_TEST" >/dev/null 2>&1 || {
            echo "  shellcheck failed"
            return 1
        }
    else
        echo "  shellcheck not available, skipping"
    fi
    return 0
}

test_missing_setup_script() {
    # Create a temporary directory without the setup script
    local temp_dir
    temp_dir=$(mktemp -d)
    local temp_script="$temp_dir/ci_speed_test.sh"
    
    # Copy the script to temp location
    cp "$SCRIPT_UNDER_TEST" "$temp_script"
    chmod +x "$temp_script"
    
    # Run the script in the temp directory (where setup_parallel_ci.sh doesn't exist)
    local output
    local exit_code
    
    cd "$temp_dir"
    output=$(bash ci_speed_test.sh 2>&1) && exit_code=0 || exit_code=$?
    cd - >/dev/null
    
    # Cleanup
    rm -rf "$temp_dir"
    
    # Should exit with code 1 and contain error message
    assert_exit_code 1 "$exit_code" "Missing setup script should cause exit 1" &&
    assert_contains "$output" "setup_parallel_ci.sh not found" "Should show missing file error"
}

test_python3_dependency_check() {
    # Mock python3 command to fail
    local temp_dir
    temp_dir=$(mktemp -d)
    local mock_python3="$temp_dir/python3"
    local temp_script="$temp_dir/ci_speed_test.sh"
    
    # Create a mock python3 that doesn't exist
    echo '#!/bin/bash\nexit 127' > "$mock_python3"
    chmod +x "$mock_python3"
    
    # Copy script and modify PATH
    cp "$SCRIPT_UNDER_TEST" "$temp_script"
    chmod +x "$temp_script"
    
    # Run with modified PATH that doesn't include real python3
    local output
    local exit_code
    output=$(PATH="$temp_dir:/bin:/usr/bin" bash "$temp_script" 2>&1) && exit_code=0 || exit_code=$?
    
    # Cleanup
    rm -rf "$temp_dir"
    
    # Should exit with code 1 and contain error message about python3
    assert_exit_code 1 "$exit_code" "Missing python3 should cause exit 1" &&
    assert_contains "$output" "python3 is required" "Should show python3 requirement error"
}

test_script_syntax() {
    # Test that the script has valid bash syntax
    bash -n "$SCRIPT_UNDER_TEST" >/dev/null 2>&1 || {
        echo "  Script has syntax errors"
        return 1
    }
    return 0
}

test_contains_required_sections() {
    local content
    content=$(cat "$SCRIPT_UNDER_TEST")
    
    assert_contains "$content" "set -euo pipefail" "Should use strict error handling" &&
    assert_contains "$content" "start_time=" "Should record start time" &&
    assert_contains "$content" "end_time=" "Should record end time" &&
    assert_contains "$content" "duration=" "Should calculate duration" &&
    assert_contains "$content" "python -c 'import pytest" "Should verify pytest import"
}

test_error_output_to_stderr() {
    local content
    content=$(cat "$SCRIPT_UNDER_TEST")
    
    # Check that error messages go to stderr
    assert_contains "$content" ">&2" "Should redirect errors to stderr"
}

# Mock setup for successful run test
create_mock_setup() {
    local setup_script="./setup_parallel_ci.sh"
    cat > "$setup_script" << 'EOF'
#!/bin/bash
echo "Mock setup script running..."
sleep 0.1
echo "Mock setup completed"
EOF
    chmod +x "$setup_script"
}

cleanup_mock_setup() {
    rm -f "./setup_parallel_ci.sh"
}

test_successful_run() {
    # This test requires python3 to be available
    if ! command -v python3 >/dev/null 2>&1; then
        echo "  Skipping successful run test - python3 not available"
        return 0
    fi
    
    # Create mock setup script
    create_mock_setup
    
    # Run the script
    local output
    local exit_code
    output=$(timeout 30s bash "$SCRIPT_UNDER_TEST" 2>&1) && exit_code=0 || exit_code=$?
    
    # Cleanup
    cleanup_mock_setup
    
    # Should succeed and contain expected output
    assert_exit_code 0 "$exit_code" "Script should succeed with mock setup" &&
    assert_contains "$output" "Starting CI speed test" "Should show start message" &&
    assert_contains "$output" "CI setup completed" "Should show completion message"
}

# Run all tests
main() {
    echo "Starting unit tests for $SCRIPT_UNDER_TEST"
    echo "========================================"
    echo
    
    # Make script executable if it isn't already
    chmod +x "$SCRIPT_UNDER_TEST" 2>/dev/null || true
    
    run_test "Script exists" test_script_exists
    run_test "Script is executable" test_script_executable
    run_test "Correct shebang" test_shebang_correct
    run_test "Valid bash syntax" test_script_syntax
    run_test "shellcheck passes" test_shellcheck_passes
    run_test "Contains required sections" test_contains_required_sections
    run_test "Error output to stderr" test_error_output_to_stderr
    run_test "Missing setup script handling" test_missing_setup_script
    run_test "Python3 dependency check" test_python3_dependency_check
    run_test "Successful run" test_successful_run
    
    echo "========================================"
    echo "Test Results:"
    echo "  Total tests: $TEST_COUNT"
    echo -e "  ${GREEN}Passed: $PASS_COUNT${NC}"
    echo -e "  ${RED}Failed: $FAIL_COUNT${NC}"
    
    if [[ $FAIL_COUNT -eq 0 ]]; then
        echo -e "${GREEN}All tests passed! ✅${NC}"
        exit 0
    else
        echo -e "${RED}Some tests failed! ❌${NC}"
        exit 1
    fi
}

# Run tests if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
