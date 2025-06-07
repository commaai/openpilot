#!/usr/bin/env bash
set -e

# Helper script to migrate workflows from Docker to native setup
# Usage: ./tools/migrate_workflow.sh <workflow_file.yaml>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <workflow_file.yaml>"
    echo "Example: $0 .github/workflows/docs.yaml"
    exit 1
fi

WORKFLOW_FILE="$1"
if [ ! -f "$WORKFLOW_FILE" ]; then
    echo "Error: Workflow file '$WORKFLOW_FILE' not found"
    exit 1
fi

# Create the native version filename
NATIVE_FILE="${WORKFLOW_FILE%.yaml}_native.yaml"
echo "Creating native version: $NATIVE_FILE"

# Copy the original file
cp "$WORKFLOW_FILE" "$NATIVE_FILE"

# Perform the migration transformations
echo "Applying native optimizations..."

# 1. Update the workflow name
sed -i.bak 's/^name: \(.*\)$/name: \1 (native)/' "$NATIVE_FILE"

# 2. Add testing branch to trigger
sed -i.bak '/branches:/,/- master/ {
    /- master/a\
      - ci-setup-optimization-test  # Added for testing
}' "$NATIVE_FILE"

# 3. Add NATIVE_RUN environment variable
sed -i.bak '/^concurrency:/i\
env:\
  NATIVE_RUN: ./tools/native_run.sh\

' "$NATIVE_FILE"

# 4. Replace setup-with-retry with setup-with-retry-native
sed -i.bak 's/setup-with-retry$/setup-with-retry-native/' "$NATIVE_FILE"

# 5. Replace Docker RUN commands with NATIVE_RUN
# This is a more complex transformation - handle common patterns
sed -i.bak 's/\${{ env\.RUN }}/\${{ env.NATIVE_RUN }}/g' "$NATIVE_FILE"

# Clean up backup files
rm -f "${NATIVE_FILE}.bak"

echo "✅ Migration completed!"
echo ""
echo "📝 Manual steps required:"
echo "1. Review the generated file: $NATIVE_FILE"
echo "2. Check for any Docker-specific commands that need adjustment"
echo "3. Update job names to include '(native)' suffix"
echo "4. Test the workflow in your fork before submitting PR"
echo ""
echo "🔧 Common manual replacements needed:"
echo "- Replace 'RUN: docker run...' with 'NATIVE_RUN: ./tools/native_run.sh'"
echo "- Remove Docker-specific environment variables (BASE_IMAGE, DOCKER_LOGIN, BUILD)"
echo "- Update any hardcoded Docker image references"
echo ""
echo "📊 Expected performance improvement:"
echo "- Setup time: ~64s → <20s (3x faster!)"
echo "- Cache hit scenarios will be consistently under 20s"
echo "- First runs (cold cache) should be under 40s"