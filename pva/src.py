import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Callable, Tuple, Union, Literal
import warnings
from bisect import bisect_left


def _check_quantiles_input(x, y, quantiles: List[float]):
    if not all(isinstance(q, float) for q in quantiles):
        raise ValueError(f'quantiles must be list of floats, not list of {set(type(q) for q in quantiles)}')

    if not all(0 <= q <= 1 for q in quantiles):
        raise ValueError(f'All quantiles must be in [0, 1]')


def _compute_cdf(y, pdf_sp):
    cdf = scipy.integrate.cumulative_trapezoid(
        y=pdf_sp,
        x=y,
        initial=0
    )

    if cdf[-1] > 0:
        cdf /= cdf[-1]
    else:
        warnings.warn('pdf_sp is equal to zero')

    return cdf


def _compute_conditional_cdf(x_val, y_min: float, y_max: float, min_dots_cnt: int, max_dots_cnt: int, pdf: Callable):
    y, stp = np.linspace(y_min, y_max, 2 ** min_dots_cnt, retstep=True)

    pdf_sp = pdf(
        np.vstack([
            np.full((y.shape[0],), fill_value=x_val),
            y
        ])
    )

    cdf = _compute_cdf(y, pdf_sp)

    for p_cnt in np.logspace(min_dots_cnt+1, max_dots_cnt+1, num=(max_dots_cnt - min_dots_cnt + 1), dtype=int, base=2):
        cdf_old = cdf

        pdf_sp_old = pdf_sp
        pdf_sp_add = pdf(
            np.vstack([
                np.full((y.shape[0],), fill_value=x_val),
                np.linspace(y_min + stp, y_max, p_cnt // 2, retstep=False)
            ])
        )

        y, stp = np.linspace(y_min, y_max, p_cnt, retstep=True)
        pdf_sp = np.full_like(y, np.nan)
        pdf_sp[::2] = pdf_sp_old
        pdf_sp[1::2] = pdf_sp_add

        cdf = _compute_cdf(y, pdf_sp)

        integral_quality = np.max(cdf_old - cdf[::2])
        interpolation_quality = np.max(np.abs(cdf[1:] - cdf[:-1]))  # rename

        if (interpolation_quality < 1e-2) and (integral_quality < 5 * 1e-2):  # TODO: add eps_1, eps_2
            break

    return cdf, y


def _qet_quantiles(x: Union[np.ndarray, Tuple[float, float]],
                   y: Union[np.ndarray, Tuple[float, float]],
                   pdf: Callable, quantiles: List[float],
                   *,
                   num_x_dots: int = 2 ** 8,
                   num_y_dots: int = 2 ** 16,
                   ) -> Dict[float, np.ndarray]:
    """
    :param x: If x is a numpy array, compute the quantiles for all elements in the array.
    Else compute the quantiles by np.linspace(x[0], x[1], num_x_dots)
    :param y: If y is a numpy array, evaluate cdf by y.
    Else evaluate cdf by np.linspace(y[0], y[1]), with len <= num_y_dots
    :param pdf: Probability distribution function used to compute the quantiles.
    :param quantiles: List of quantiles to be computed
    :return:
    """

    _check_quantiles_input(x, y, quantiles)

    if isinstance(x, np.ndarray):
        q_res = {q: np.zeros_like(x, dtype=np.float64) for q in quantiles}
    else:
        q_res = {q: np.zeros(num_x_dots, dtype=np.float64) for q in quantiles}
        x = np.linspace(x[0], x[1], num_x_dots)

    assert not isinstance(y, np.ndarray), "Not implemented yet"
    for i, x_val in enumerate(x):
        cdf, y_sp = _compute_conditional_cdf(x_val, y[0], y[1], 6, 16, pdf)  # TODO: use num_y_dots
        for q in quantiles:
            q_res[q][i] = y_sp[bisect_left(cdf, q)]

    return q_res


def plot_precision_curve(y_true, y_pred, ax=None, proba: List[float] = None,
                         rug_plot: bool = False,
                         plot_optimal: bool = False,
                         plot_mode: Literal['raw', 'deviation'] = 'raw',
                         **kwargs):
    ax = plt.subplots()[1] if ax is None else ax
    proba = [0.5, 0.95] if proba is None else sorted(proba)

    pdf = scipy.stats.gaussian_kde((y_true, y_pred))

    q = [0.5]
    for p in proba:
        q.append(0.5 - p/2)
        q.append(0.5 + p/2)

    x = np.linspace(np.min(y_true), np.max(y_true), 2 ** 8)
    q_res = _qet_quantiles(
        x=x,
        y=(np.min(y_pred), np.max(y_pred)),
        pdf=pdf,
        quantiles=q,
        num_y_dots=2 ** 16
    )

    match plot_mode:
        case 'raw':
            def plt_mode_preprocessor(x_v, y_v): return y_v
        case 'deviation':
            def plt_mode_preprocessor(x_v, y_v): return y_v - x_v
        case _:
            raise ValueError(f'Unknown plot mode: {plot_mode}')

    if plot_optimal:
        ax.plot(x, plt_mode_preprocessor(x, x), c='r', label='Best regressor', linestyle='--')

    ax.plot(x, plt_mode_preprocessor(x, q_res[0.5]), c='b', label='median')

    for p in proba:
        ax.fill_between(x, plt_mode_preprocessor(x, q_res[0.5 - p/2]), plt_mode_preprocessor(x, q_res[0.5 + p/2]),
                        alpha=0.2, label=f'{p*100:.1f}%')

    if rug_plot:
        sns.rugplot(x=y_true, ax=ax, label='Distribution of actual values')
        if plot_mode == 'raw':
            sns.rugplot(y=y_pred, ax=ax, label='Distribution of predicted values')
        elif plot_mode == 'deviation':
            sns.rugplot(y=y_pred - y_true, ax=ax, label='Distribution of predicted values')

    ax.legend()
