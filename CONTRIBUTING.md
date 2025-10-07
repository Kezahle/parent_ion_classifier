# Contributing to Parent Ion Classifier

Thank you for your interest in contributing to the Parent Ion Classifier project!

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kezahle/parent_ion_classifier.git
   cd parent_ion_classifier
   ```

2. **Create a development environment**
   ```bash
   conda create -n parent_classifier_dev python=3.12
   conda activate parent_classifier_dev
   conda install -y conda-build
   ```

3. **Build and test**
   ```bash
   ./scripts/build_and_test.sh all
   ```

## Code Style

- Follow PEP 8 guidelines
- Use snake_case for functions and variables
- Use descriptive variable names
- Add docstrings to all public functions and classes
- Keep lines under 100 characters where practical

## Making Changes

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, descriptive commit messages
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   ./scripts/build_and_test.sh all
   conda activate test_env
   parent-ion-classifier test
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "Description of your changes"
   git push origin feature/your-feature-name
   ```

5. **Submit a Pull Request**
   - Provide a clear description of the changes
   - Reference any related issues
   - Ensure all tests pass

## Testing

### Running Tests

```bash
parent-ion-classifier test
```

### Adding New Tests

When adding new functionality:
1. Add test data to `src/parent_ion_classifier/test/`
2. Update test cases in `test.py`
3. Ensure tests pass on both Mac and Linux

### Cross-Platform Testing

The project aims for reproducibility across platforms. When making changes:
- Test on both macOS and Linux if possible
- Ensure deterministic behavior (avoid platform-specific sorting/randomness)
- Use appropriate tolerances for floating-point comparisons

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Update docs/ for significant features

## Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages (if any)

## Questions?

Feel free to open an issue for discussion before making major changes.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
