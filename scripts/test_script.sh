#!/bin/bash

# Test environment setup script for parent_ion_classifier
# This script creates a fresh test environment and installs the package

set -e  # Exit on any error

# Configuration - use script's directory and navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGE_DIR="$PROJECT_ROOT"
BUILD_ENV="parent_classifier_packaging"
TEST_ENV="test_env"
PYTHON_VERSION="3.12"

echo "=========================================="
echo "Setting up test environment for parent_ion_classifier"
echo "=========================================="

# Navigate to package directory (project root)
echo "Navigating to package directory: $PACKAGE_DIR"
cd "$PACKAGE_DIR"

# Function to initialize conda in the script
initialize_conda() {
    # Try common conda locations
    CONDA_BASES=(
        "/opt/anaconda3"
        "/home/ec2-user/anaconda3"
        "$HOME/anaconda3"
        "$HOME/miniconda3"
    )
    
    for CONDA_BASE in "${CONDA_BASES[@]}"; do
        if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            echo "Initializing conda from: $CONDA_BASE/etc/profile.d/conda.sh"
            source "$CONDA_BASE/etc/profile.d/conda.sh"
            return 0
        fi
    done
    
    # If conda is already available (already initialized)
    if command -v conda >/dev/null 2>&1; then
        echo "Conda command already available"
        return 0
    fi
    
    echo "Could not find conda installation"
    return 1
}

# Initialize conda
if ! initialize_conda; then
    echo "Failed to initialize conda. Please ensure conda is installed and available."
    echo "You can also run this script from a conda environment:"
    echo "  conda activate $BUILD_ENV"
    echo "  ./scripts/test_script.sh"
    exit 1
fi

# Remove existing test environment if it exists
echo "Removing existing test environment if it exists..."
conda remove -y --name "$TEST_ENV" --all 2>/dev/null || echo "No existing $TEST_ENV environment found"

# Create new test environment
echo "Creating new test environment: $TEST_ENV with Python $PYTHON_VERSION"
conda create -y -n "$TEST_ENV" python="$PYTHON_VERSION"

# Verify the package file exists (supports both .tar.bz2 and .conda formats)
echo "Looking for package in: $PACKAGE_DIR/bin"
sleep 1  # Give filesystem a moment to sync
PACKAGE_FILE=$(find "$PACKAGE_DIR/bin" -type f \( -name "parent_ion_classifier*.tar.bz2" -o -name "parent_ion_classifier*.conda" \) 2>/dev/null | head -1)
if [ -z "$PACKAGE_FILE" ]; then
    echo "No package file found in bin directory!"
    echo "Looking in: $PACKAGE_DIR/bin"
    echo "Searching for: parent_ion_classifier*.tar.bz2 or parent_ion_classifier*.conda"
    echo ""
    echo "Available package files:"
    find "$PACKAGE_DIR/bin" -type f \( -name "*.tar.bz2" -o -name "*.conda" \) 2>/dev/null || echo "No package files found"
    echo ""
    echo "Full directory structure:"
    ls -laR "$PACKAGE_DIR/bin" 2>/dev/null || echo "bin directory does not exist"
    echo ""
    echo "Please run build_script.sh first:"
    echo "  ./scripts/build_script.sh"
    exit 1
fi

echo "Found package file: $PACKAGE_FILE"

# Install the package in test environment
echo "Installing parent_ion_classifier from local package..."
echo "Command: conda install -y parent_ion_classifier -c file://$PACKAGE_DIR/bin -n $TEST_ENV"
conda install -y parent_ion_classifier -c "file://$PACKAGE_DIR/bin" -n "$TEST_ENV"

# Activate test environment for verification
echo "Activating test environment: $TEST_ENV"
conda activate "$TEST_ENV"

# Verify installation
echo "=========================================="
echo "Installation verification"
echo "=========================================="

echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"

# Test basic import
echo "Testing package import..."
python -c "import parent_ion_classifier; print('Package import successful')" || echo "Package import failed"

# Test CLI commands
echo "Testing CLI commands..."

if command -v parent-ion-classifier &> /dev/null; then
    echo "parent-ion-classifier command available"
    
    echo "Available models:"
    parent-ion-classifier list
    
    echo ""
    echo "Model status:"
    parent-ion-classifier status
    
    echo ""
    echo "Help output:"
    parent-ion-classifier --help
    
else
    echo "parent-ion-classifier command not found"
fi

if command -v classify_parent_ion &> /dev/null; then
    echo "classify_parent_ion command available"
else
    echo "classify_parent_ion command not found"
fi

echo "=========================================="
echo "Test environment setup complete!"
echo "=========================================="
echo "To use the test environment:"
echo "  conda activate $TEST_ENV"
echo ""
echo "To test model downloading:"
echo "  parent-ion-classifier download"
echo ""
echo "To run unit tests:"
echo "  parent-ion-classifier test"
echo "=========================================="