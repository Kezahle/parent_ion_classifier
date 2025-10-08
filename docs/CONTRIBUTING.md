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

3. **Install with development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set up pre-commit hooks** (Recommended)
   ```bash
   pre-commit install
   ```

   This will automatically run code quality checks before each commit:
   - Format code with black
   - Sort imports with isort
   - Check for common issues with ruff
   - Validate YAML/TOML files
   - Remove trailing whitespace

5. **Build and test**
   ```bash
   ./scripts/build_and_test.sh all
   ```

## Code Style

- Follow PEP 8 guidelines
- Use snake_case for functions and variables
- Use descriptive variable names
- Add type hints to all functions
- Add docstrings to all public functions and classes
- Keep lines under 100 characters

**Pre-commit hooks will automatically enforce most style rules!**

## Making Changes

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, descriptive commit messages
   - Add type hints to new functions
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run pre-commit checks
   pre-commit run --all-files

   # Build and test the package
   ./scripts/build_and_test.sh all

   # Run the unit tests
   conda activate test_env
   parent-ion-classifier test
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "Description of your changes"
   # Pre-commit hooks run automatically here!

   git push origin feature/your-feature-name
   ```

5. **Submit a Pull Request**
   - Provide a clear description of the changes
   - Reference any related issues
   - Ensure all tests pass
   - Ensure pre-commit checks pass

## Code Quality Tools

We use several tools to maintain code quality:

- **black**: Code formatter (100 character line length)
- **isort**: Import sorter
- **ruff**: Fast Python linter
- **mypy**: Type checker (optional, not enforced yet)

These run automatically via pre-commit hooks before each commit.

### Running Quality Checks Manually

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Run specific hooks
pre-commit run black --all-files
pre-commit run ruff --all-files

# Format code
black src/
isort src/

# Check types (if mypy is configured)
mypy src/
```

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
- Use appropriate tolerances for floating-point comparisons (rtol=1e-4, atol=1e-6)

## Type Hints

All new code should include type hints:

```python
# Good
def process_data(input_path: str, max_size: int = 100) -> pd.DataFrame:
    """Process data from file."""
    pass

# Bad - no type hints
def process_data(input_path, max_size=100):
    """Process data from file."""
    pass
```

Type hints help:
- Catch bugs before runtime
- Improve IDE autocomplete
- Serve as inline documentation

## Documentation

- Update README.md for user-facing changes
- Add docstrings with type information to new functions/classes
- Update docs/ for significant features
- Include examples in docstrings

**Example docstring:**

```python
def canonise_and_truncate_df(
    self,
    in_df: pd.DataFrame,
    reset_index: bool = False,
    sort_by_MS2_first: bool = True
) -> pd.DataFrame:
    """
    Complete canonization pipeline for MS data.

    This is the main entry point for data preprocessing. It performs:
    1. Data validation
    2. m/z rounding
    3. Deterministic sorting by intensity
    4. Truncation to top-N rows

    Args:
        in_df: Input DataFrame with columns matching data_columns
        reset_index: Whether to reset index after processing
        sort_by_MS2_first: If True, sort by MS2 then MS1; otherwise MS1 then MS2

    Returns:
        Canonized DataFrame with exactly N rows, ready for model input

    Raises:
        AssertionError: If input validation fails

    Example:
        >>> dc = DataCanonizer(N=150, data_columns=['mz', 'MS1', 'MS2'],
        ...                    label_columns=['parent'])
        >>> df = pd.DataFrame({'mz': [100, 200], 'MS1': [1000, 500],
        ...                    'MS2': [2000, 1500]})
        >>> canonized = dc.canonise_and_truncate_df(df)
        >>> len(canonized)
        150
    """
    pass
```

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
