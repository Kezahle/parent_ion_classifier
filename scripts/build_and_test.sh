#!/bin/bash

# Combined build and test script for parent_ion_classifier
# Usage: ./scripts/build_and_test.sh [build|test|all]

set -e  # Exit on any error

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

function show_usage() {
    echo "Usage: $0 [build|test|all]"
    echo ""
    echo "Commands:"
    echo "  build  - Build the conda package only"
    echo "  test   - Set up test environment and install package only"
    echo "  all    - Build package and set up test environment (default)"
    echo ""
    echo "Examples:"
    echo "  $0 build    # Just build the package"
    echo "  $0 test     # Just set up test environment"
    echo "  $0 all      # Build and test (default)"
    echo "  $0          # Same as 'all'"
}

function run_build() {
    echo "🔨 Running build script..."
    bash "$SCRIPT_DIR/build_script.sh"
}

function run_test() {
    echo "🧪 Running test environment setup..."
    bash "$SCRIPT_DIR/test_script.sh"
}

# Parse command line argument
COMMAND=${1:-all}

case $COMMAND in
    "build")
        run_build
        ;;
    "test")
        run_test
        ;;
    "all")
        run_build
        echo ""
        echo "Build completed. Setting up test environment..."
        echo ""
        run_test
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        echo "❌ Unknown command: $COMMAND"
        echo ""
        show_usage
        exit 1
        ;;
esac

echo ""
echo "🎉 Operation completed successfully!"