# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python Machine Learning (ML) Plot is a library providing reusable functions for machine learning visualization plotting. The primary focus is on classification model visualization with decision boundaries and classified regions.

## Architecture

### Module Structure

- `src/opengood/py_ml_plot/` - Main package implementing ML plotting utilities
  - `classification_plot.py` - Contains `setup_classification_plot()` function for creating 2D classification visualizations
  - `__init__.py` - Exports public API (`setup_classification_plot`)

### Classification Plot Architecture

The `setup_classification_plot()` function visualizes classification models through several key steps:

1. **Meshgrid Generation**: Creates a grid of points across the feature space using configurable padding and step sizes for each axis
2. **Feature Scaling** (optional): Applies inverse transformations via `feature_scale` lambda to handle models trained on scaled data
3. **Prediction**: Uses the `predict` lambda to generate class predictions across the entire meshgrid
4. **Visualization**: Combines filled contour plots (decision regions) with scatter plots (actual data points)

The function is designed to work with any scikit-learn classifier by accepting prediction and feature scaling logic as lambda functions.

## Development Commands

### Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
python3 -m pip install matplotlib numpy pandas scikit-learn
```

### Testing

```bash
# Run all tests with coverage
python -m pytest tests/

# Run a specific test
python -m pytest tests/py_ml_plot/test_classification_plot.py::TestClassificationPlot::test_logistic_regression_setup_classification_plot_with_shaded_regions

# Run tests with verbose output
python -m pytest tests/ -v

# Run tests with coverage report
python -m pytest tests/ --cov=src.opengood.py_ml_plot --cov-report=term-missing
```

**Note**: Some tests are marked with `@skip` decorator for long-running tests (e.g., K-NN visualization). These are intended for local verification only.

### Test Configuration

Test configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`:
- Coverage threshold: 70%
- Test markers: `slow`, `smoke`, `unit`
- Coverage report outputs to `coverage.xml`

### Project Structure

```
src/opengood/py_ml_plot/
  - classification_plot.py  # Core plotting function
  - __init__.py             # Public API exports
tests/
  - py_ml_plot/
    - test_classification_plot.py  # Tests for classification plotting
  - resources/
    - data.csv                     # Test dataset
```

## Key Implementation Details

### Meshgrid Parameter Structure

The `meshgrid` parameter controls the visualization bounds and resolution:

```python
meshgrid = {
    0: {"min": 10, "max": 10, "step": 0.25},    # x-axis padding and step
    1: {"min": 1000, "max": 1000, "step": 0.25}  # y-axis padding and step
}
```

- Keys `0` and `1` represent x-axis and y-axis respectively
- `min`/`max` define padding subtracted/added from data min/max values
- `step` controls the density of the meshgrid (smaller = higher resolution but slower)

### Lambda Function Patterns

**Feature Scaling Lambda**:
```python
feature_scale = lambda x_set, y_set: (sc.inverse_transform(x_set), y_set)
```
- Inverts feature scaling applied during training to show original data ranges
- Returns tuple of (transformed_x, original_y)

**Prediction Lambda**:
```python
predict = lambda x1, x2: classifier.predict(
    sc.transform(np.array([x1.ravel(), x2.ravel()]).T)
).reshape(x1.shape)
```
- Takes meshgrid coordinates (x1, x2)
- Applies feature scaling if the classifier was trained on scaled data
- Uses `ravel()` to flatten, combines into 2D array, transforms, predicts, then reshapes

## Package Management

- Build system: setuptools
- Python version: >=3.12
- Package name: `opengood.py_ml_plot`
- Current version tracked in `pyproject.toml`
- Version bumping configured in `.bumpversion.toml`

## Dependencies

Core runtime dependencies:
- matplotlib >= 3.10.3 (visualization)
- numpy >= 2.3.0rc1 (numerical operations)
- pandas >= 2.2.3 (data handling)
- scikit-learn >= 1.7.0rc1 (ML models in examples/tests)