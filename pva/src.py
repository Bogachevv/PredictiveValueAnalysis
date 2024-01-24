import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Callable, Tuple, Union, Literal
import warnings
from bisect import bisect_left
import itertools
from fastkde import fastKDE


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

    for p_cnt in np.logspace(min_dots_cnt + 1, max_dots_cnt + 1, num=(max_dots_cnt - min_dots_cnt + 1), dtype=int,
                             base=2):
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


def _qet_quantiles_kde(x: Union[np.ndarray, Tuple[float, float]],
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


def _compute_conditional_cdf_fastkde(x_val: float, fastkde):
    pos = bisect_left(fastkde.coords['var0'], x_val)  # get position of x_val in x axis array, given by fastkde
    assert x_val == fastkde.coords['var0'][pos], \
        f"{x_val=}\n{fastkde.coords['var0'].to_numpy()=}\n{fastkde.coords['var1'].to_numpy()}"

    valid_idx = ~np.isnan(fastkde.data[:, pos])  # get indexes, where fastkde compute conditional pdf for x_val
    pdf = fastkde.data[:, pos][valid_idx]
    y = fastkde.coords['var1'][valid_idx]

    return _compute_cdf(y, pdf), y


def _qet_quantiles_fastkde(fastkde, quantiles: List[float]) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
    x = fastkde.coords['var0']
    x_valid = [not np.all(np.isnan(fastkde.data[:, i])) for i in range(x.shape[0])]
    x = x[x_valid].to_numpy()

    q_res = {q: np.zeros_like(x, dtype=np.float64) for q in quantiles}

    for i, x_val in enumerate(x):
        cdf, y_sp = _compute_conditional_cdf_fastkde(x_val, fastkde)
        for q in quantiles:
            q_res[q][i] = y_sp[min(bisect_left(cdf, q), len(x))]

    return x, q_res


def _gen_proba_bounds(proba: list[float]):
    for p in proba:
        yield 0.5 + p / 2

    yield 0.5

    for p in reversed(proba):
        yield 0.5 - p / 2


def _gen_proba_bound_pairs(proba: list[float]):
    g1, g2 = itertools.tee(_gen_proba_bounds(proba), 2)
    next(g2)

    yield from zip(g1, g2)


def _raw_plt_mode(x_v, y_v):
    return y_v


def _deviation_plt_mode(x_v, y_v):
    return y_v - x_v


@np.vectorize
def _relative_plt_mode(x_v, y_v, eps: float = 1):
    if np.abs(x_v) > eps:
        return (y_v - x_v) / np.abs(x_v)

    return (y_v - x_v) / eps


def plot_precision_curve(y_true, y_pred, ax=None, proba: List[float] = None,
                         rug_plot: bool = False,
                         plot_optimal: bool = False,
                         plot_mode: Literal['raw', 'deviation', 'relative'] = 'raw',
                         strategy: Literal['kde', 'fastkde', 'auto'] = 'auto',
                         x_label: str = 'Actual value', y_label: str = 'Predicted value',
                         show_legend: bool = True,
                         alpha: float = 0.25,
                         **kwargs):
    ax = plt.subplots()[1] if ax is None else ax
    proba = [0.95, 0.5] if proba is None else sorted(proba, reverse=True)

    q = [0.5]
    for p in proba:
        q.append(0.5 - p / 2)
        q.append(0.5 + p / 2)

    if strategy == 'auto':
        raise NotImplementedError(f"Strategy {strategy} not implemented")

    if strategy == 'kde':
        x = np.linspace(np.min(y_true), np.max(y_true), 2 ** 8)
        pdf = scipy.stats.gaussian_kde((y_true, y_pred))
        q_res = _qet_quantiles_kde(
            x=x,
            y=(np.min(y_pred), np.max(y_pred)),
            pdf=pdf,
            quantiles=q,
            num_y_dots=2 ** 16
        )
    else:
        fastkde = fastKDE.conditional(inputVars=y_pred, conditioningVars=y_true, peak_frac=1e-4)
        x, q_res = _qet_quantiles_fastkde(
            fastkde=fastkde,
            quantiles=q,
        )

    plt_modes = {
        'raw': _raw_plt_mode,
        'deviation': _deviation_plt_mode,
        'relative': _relative_plt_mode,
    }

    plt_mode_preprocessor = plt_modes[plot_mode]

    if plot_optimal:
        ax.plot(x, plt_mode_preprocessor(x, x), c='r', label='Best regressor', linestyle='--')

    ax.plot(x, plt_mode_preprocessor(x, q_res[0.5]), c='b', label='median')

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (p_u, p_d) in enumerate(_gen_proba_bound_pairs(proba)):
        c = colors[i if i < len(proba) else 2 * len(proba) - i - 1]
        label = f'{(p_u - 0.5) * 2 * 100:.1f}%' if p_u > 0.5 else None
        print(f"p_u: {p_u}, p_d: {p_d}, c: {c}")
        ax.fill_between(x, plt_mode_preprocessor(x, q_res[p_d]), plt_mode_preprocessor(x, q_res[p_u]),
                        alpha=alpha, label=label, color=c)

    if rug_plot:
        sns.rugplot(x=y_true, ax=ax, label='Distribution of actual values')
        if plot_mode == 'raw':
            sns.rugplot(y=y_pred, ax=ax, label='Distribution of predicted values')
        elif plot_mode == 'deviation':
            sns.rugplot(y=y_pred - y_true, ax=ax, label='Distribution of predicted values')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if show_legend:
        ax.legend()
