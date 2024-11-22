# Regression Model Validation Library

This library provides a comprehensive set of tools for validating regression models. It enables developers and data scientists to evaluate model performance through various statistical metrics, visualization techniques, and density estimation methods. The primary goal is to facilitate better insights into the relationship between predicted and actual values in regression tasks.

## Features

- **Visualization Tools**:
  - Joint density plots for visualizing the relationship between predictions and actual values.
  - Precision and recall curves with confidence intervals for detailed analysis.
  
- **Performance Metrics**:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R-squared (R²)

- **Customizable Plotting**:
  - Confidence intervals for precision and recall.
  - Flexible options for plot orientation, axes limits, and rug plots.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Bogachevv/PredictiveValueAnalysis
cd dictiveValueAnalysis
pip install -r requirements.txt
```

## Usage

1. **Initializing the Validation Class**

```python
from validation import Validation
import numpy as np

# Example data
y_pred = np.array([3.1, 2.8, 4.5, 3.7])
y_act = np.array([3.0, 3.0, 4.0, 4.0])

# Initialize
validation = Validation(y_pred=y_pred, y_act=y_act, kde_eval='fast')
```

2. **Plotting joint distribution**

```python
validation.plot_joint(
    x_label="Predicted Values",
    y_label="Actual Values",
    title="Joint Density Plot"
)
```

3. **Calculating metrics**

```python
print("R² Score:", validation.r2_score())
print("MSE:", validation.mse_score())
print("RMSE:", validation.rmse_score())
print("MAE:", validation.mae_score())
```

4. **Plotting Precision and Recall curves**

```python
validation.plot_precision_curve(confidence_prob=[0.5, 0.95])
validation.plot_recall_curve(confidence_prob=[0.5, 0.95])
```

## API Reference

### `Validation` Class

#### Constructor

```python
Validation(
    y_pred: np.ndarray,
    y_act: np.ndarray,
    kde_eval: Literal['auto', 'fast', 'direct'] = 'auto'
)
```

**Parameters**:

1. y_pred (np.ndarray): The predicted values from the regression model.
2. y_act (np.ndarray): The actual target values.
3. kde_eval (Literal['auto', 'fast', 'direct']): The method for kernel density estimation. Defaults to 'auto'

### Methods

#### plot_joint

```python
plot_joint(
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    orient: Literal['actual', 'predictions'] = 'predictions',
    show_bisect: Optional[bool] = True,
    ax = None,
    x_lims: tuple[float, float] = None,
    y_lims: tuple[float, float] = None
)
```

* **Description**: Visualizes the joint density of predictions and actual values.
* **Parameters**:
    1. x_label: Label for the x-axis (default: "Model predictions")
    2. y_label: Label for the y-axis (default: "Actual")
    3. title: Plot title (default: "Density of joint distribution")
    4. orient: Orientation of the plot ('actual' or 'predictions')
    5. show_bisect: Whether to show a bisecting line
    6. ax: Matplotlib axis object for customization
    7. x_lims, y_lims: Custom axis limits

#### plot_precision_curve

```python
plot_precision_curve(
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    confidence_prob: Optional[list] = None,
    ax = None,
    rug_plot: bool = True,
    plot_optimal: bool = False,
    plot_mode: Literal['raw', 'deviation', 'relative'] = 'raw',
    show_legend: bool = True,
    alpha: float = 0.25
)
```

* **Description**: Plots the precision curve
* **Parameters**:
    1. x_label: Label for the x-axis (default: "Model predictions")
    2. y_label: Label for the y-axis (default: "Actual target values")
    3. confidence_prob: List of probabilities (default: [0.5, 0.95])
    4. rug_plot: Whether to add a rug plot for data distribution
    5. plot_optimal: Whether to include the optimal line
    6. plot_mode: Plot type ('raw', 'deviation', 'relative')
    7. show_legend: Whether to show the legend
    8. alpha: Transparency of the confidence intervals

#### plot_recall_curve

```python
plot_recall_curve(
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    confidence_prob: Optional[list] = None,
    ax = None,
    rug_plot: bool = True,
    plot_optimal: bool = False,
    plot_mode: Literal['raw', 'deviation', 'relative'] = 'raw',
    show_legend: bool = True,
    alpha: float = 0.25
)
```

* **Description**: Plots the recall curve
* **Parameters**: Same as plot_precision_curve, but with default labels swapped