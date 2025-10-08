#!/bin/bash

# Build script for parent_ion_classifier conda package
# This script sets up the build environment and creates the conda package

set -e  # Exit on any error

# Configuration - use script's directory and navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGE_DIR="$PROJECT_ROOT"
BUILD_ENV="parent_classifier_packaging"
OUTPUT_DIR="bin"

echo "=========================================="
echo "Building parent_ion_classifier package"
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
    echo "  ./scripts/build_script.sh"
    exit 1
fi

# Activate build environment
echo "Activating conda environment: $BUILD_ENV"
if ! conda activate "$BUILD_ENV"; then
    echo "Failed to activate environment '$BUILD_ENV'"
    echo "Available environments:"
    conda env list
    echo ""
    echo "Please create the environment first:"
    echo "  conda create -n $BUILD_ENV python=3.12"
    exit 1
fi

# Verify we're in the right environment
echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"

# Verify conda-build is available
if ! command -v conda-build >/dev/null 2>&1; then
    echo "conda-build not found. Installing..."
    conda install -y conda-build
fi

# Clean previous builds (optional)
if [ -d "$OUTPUT_DIR" ]; then
    echo "Cleaning previous build artifacts..."
    rm -rf "$OUTPUT_DIR"
fi

# Build the package
echo "Building conda package..."
echo "Command: conda build conda-recipe --output-folder $OUTPUT_DIR"
conda build conda-recipe --output-folder "$OUTPUT_DIR"

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Build completed successfully!"
    echo "Package location: $PACKAGE_DIR/$OUTPUT_DIR"
    echo "Contents:"
    find "$OUTPUT_DIR" -name "*.tar.bz2" -exec echo "  {}" \;
    echo "=========================================="
else
    echo "Build failed!"
    exit 1
fi

# Optional: Show package info
echo "Package details:"
find "$OUTPUT_DIR" -name "*.tar.bz2" -exec conda package -w {} \; | head -20 2>/dev/null || echo "Package info not available"
