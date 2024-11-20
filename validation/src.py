import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

import functools
import itertools

from typing import Literal, Optional, Union, Callable
import warnings

from fastkde import fastKDE
from bisect import bisect_left

class Validation:
    def __init__(
            self,
            y_pred: np.ndarray,
            y_act: np.ndarray,
            kde_eval: Literal['auto', 'fast', '...'] = 'auto',
        ):
        
        self._y_pred = y_pred
        self._y_act = y_act
        self._kde_eval = kde_eval

        # cached
        self._gaussian_kde_cached = None
        self._fast_kde_precision_cached = None
        self._fast_kde_recall_cached = None

    def plot_joint(
            self,
            x_label: Optional[str] = None,
            y_label: Optional[str] = None,
            title: Optional[str] = None,
            orient: Literal['actual', 'predictions'] = 'predictions',
            show_bisect: Optional[bool] = True,
            ax = None,
            x_lims: tuple[float, float] = None,
            y_lims: tuple[float, float] = None,
        ):

        if orient == 'predictions':
            x, y = self._y_pred, self._y_act
        elif orient == 'actual':
            x, y = self._y_act, self._y_pred
        else:
            raise ValueError(f"Incorrect orient arg: {orient}")

        if ax is None:
            ax = plt.gca()

        sns.kdeplot(
            x=x,
            y=y,
            ax=ax
        )

        if show_bisect:
            if x_lims:
                ax.set_xlim(x_lims)
            if y_lims:
                ax.set_ylim(y_lims)

            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_xlim()

            bisect_min = min(x_min, y_min)
            bisect_max = min(x_max, y_max)

            sns.lineplot(
                x=[bisect_min, bisect_max],
                y=[bisect_min, bisect_max],
                ax=ax,
                linestyle='--',
            )

        x_label = 'Model predictions' if x_label is None else x_label
        y_label = 'Actual' if y_label is None else y_label
        title = 'Density of joint distribution' if title is None else title
        
        legend = ['density']
        if show_bisect:
            legend.append('bisect')

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend(legend)

    def _get_recall_fast_kde(self):
        if self._fast_kde_recall_cached is None:
            self._fast_kde_recall_cached = fastKDE.conditional(inputVars=self._y_pred, conditioningVars=self._y_act, peak_frac=1e-4, num_points=2**12 + 1)
        
        return self._fast_kde_recall_cached

    def _get_precision_fast_kde(self):
        if self._fast_kde_precision_cached is None:
            self._fast_kde_precision_cached = fastKDE.conditional(inputVars=self._y_act, conditioningVars=self._y_pred, peak_frac=1e-4, num_points=2**12 + 1)

        return self._fast_kde_precision_cached

    def _get_gaussian_kde(self):
        return None

    def _get_kde_kwargs(self, is_precision: bool):
        use_fast_kde = False
        gaussian_kde = None
        fastkde = None

        kde_eval = self._kde_eval

        if kde_eval == 'auto':
            raise NotImplementedError
            kde_eval = None

        if kde_eval == 'fast':
            use_fast_kde = True
            if is_precision:
                fastkde = self._get_precision_fast_kde()
            else:
                fastkde = self._get_recall_fast_kde()
        elif kde_eval == 'direct':
            use_fast_kde = False
            gaussian_kde = self._get_gaussian_kde()

        return use_fast_kde, gaussian_kde, fastkde

    def plot_precision_curve(
            self, 
            x_label: Optional[str] = None,
            y_label: Optional[str] = None,
            confidence_prob: Optional[list] = None,
            ax = None,
            rug_plot: bool = True,
            plot_optimal: bool = False,
            plot_mode: Literal['raw', 'deviation', 'relative'] = 'raw',
            show_legend: bool = True,
            alpha: float = 0.25
        ):

        confidence_prob = [0.5, 0.95] if confidence_prob is None else confidence_prob
        use_fast_kde, gaussian_kde, fastkde = self._get_kde_kwargs(is_precision=True)

        x_label = 'Model predictions' if x_label is None else x_label
        y_label = 'Actual target values' if y_label is None else y_label

        _plot_quantile_curve(
            x=self._y_pred,
            y=self._y_act,
            x_label=x_label, y_label=y_label,
            confidence_prob=confidence_prob,
            use_fast_kde=use_fast_kde, 
            gaussian_kde=gaussian_kde, 
            fastkde=fastkde,
            ax=ax,
            rug_plot=rug_plot,
            plot_optimal=plot_optimal,
            plot_mode=plot_mode,
            show_legend=show_legend,
            alpha=alpha
        )

    def plot_recall_curve(
            self, 
            x_label: Optional[str] = None,
            y_label: Optional[str] = None,
            confidence_prob: Optional[list] = None,
            ax = None,
            rug_plot: bool = True,
            plot_optimal: bool = False,
            plot_mode: Literal['raw', 'deviation', 'relative'] = 'raw',
            show_legend: bool = True,
            alpha: float = 0.25
        ):

        confidence_prob = [0.5, 0.95] if confidence_prob is None else confidence_prob
        use_fast_kde, gaussian_kde, fastkde = self._get_kde_kwargs(is_precision=False)
        
        x_label = 'Actual target values' if x_label is None else x_label
        y_label = 'Model predictions' if y_label is None else y_label

        _plot_quantile_curve(
            x=self._y_act,
            y=self._y_pred,
            x_label=x_label, y_label=y_label,
            confidence_prob=confidence_prob,
            use_fast_kde=use_fast_kde, 
            gaussian_kde=gaussian_kde,
            fastkde=fastkde,
            ax=ax,
            rug_plot=rug_plot,
            plot_optimal=plot_optimal,
            plot_mode=plot_mode,
            show_legend=show_legend,
            alpha=alpha
        )

    def r2_score(self, ):
        return r2_score(y_true=self._y_act, y_pred=self._y_pred)

    def mse_score(self, ):
        return mean_squared_error(y_true=self._y_act, y_pred=self._y_pred)
    
    def rmse_score(self, ):
        return root_mean_squared_error(y_true=self._y_act, y_pred=self._y_pred)

    def mae_score(self, ):
        return mean_absolute_error(y_true=self._y_act, y_pred=self._y_pred)


def _raw_plt_mode(x_v: np.ndarray, y_v: np.ndarray, **kwargs) -> np.ndarray:
    return y_v


def _deviation_plt_mode(x_v: np.ndarray, y_v: np.ndarray, **kwargs) -> np.ndarray:
    return y_v - x_v


def _relative_plt_mode(x_v: np.ndarray, y_v: np.ndarray, **kwargs) -> np.ndarray:
    eps = kwargs.get('eps', 1e-6)
    divisor = np.maximum(np.abs(x_v), eps)

    return (x_v - y_v) / divisor


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


def _get_plotter(plot_mode: Literal['raw', 'deviation', 'relative'], **kwargs) -> Callable:
    if plot_mode == 'raw':
        return functools.partial(_raw_plt_mode, **kwargs)
    if plot_mode == 'deviation' :
        return functools.partial(_deviation_plt_mode, **kwargs)
    if plot_mode == 'relative':
        return functools.partial(_relative_plt_mode, **kwargs)
    
    raise ValueError(f"Incorrect plot_mode: {plot_mode}")


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


def _compute_conditional_cdf_fastkde(x_val, fastkde):
    pos = bisect_left(fastkde.coords['var0'], x_val)  # get position of x_val in x axis array, given by fastkde
    assert x_val == fastkde.coords['var0'][pos], \
        f"{x_val=}\n{fastkde.coords['var0'].to_numpy()=}\n{fastkde.coords['var1'].to_numpy()}"

    valid_idx = ~np.isnan(fastkde.data[:, pos])  # get indexes, where fastkde compute conditional pdf for x_val
    pdf = fastkde.data[:, pos][valid_idx]
    y = fastkde.coords['var1'][valid_idx]

    return _compute_cdf(y, pdf), y


def _get_quantile_curves_fast(x: np.ndarray, y: np.ndarray, quantiles: list[float], fastkde): 
    xs = fastkde.coords['var0']
    xs_valid = [not np.all(np.isnan(fastkde.data[:, i])) for i in range(xs.shape[0])]
    xs = xs[xs_valid].to_numpy()

    q_res = {q: np.zeros_like(xs, dtype=np.float64) for q in quantiles}

    for i, x_val in enumerate(xs):
        cdf, y_sp = _compute_conditional_cdf_fastkde(x_val, fastkde)
        for q in quantiles:
            q_res[q][i] = y_sp[min(bisect_left(cdf, q), len(xs))]

    return xs, q_res


def _qet_quantiles_kde(x: Union[np.ndarray, tuple[float, float]],
                       y: Union[np.ndarray, tuple[float, float]],
                       pdf: Callable, quantiles: list[float],
                       *,
                       num_x_dots: int = 2 ** 8,
                       num_y_dots: int = 2 ** 16,
                       ) -> dict[float, np.ndarray]:
    """
    :param x: If x is a numpy array, compute the quantiles for all elements in the array.
    Else compute the quantiles by np.linspace(x[0], x[1], num_x_dots)
    :param y: If y is a numpy array, evaluate cdf by y.
    Else evaluate cdf by np.linspace(y[0], y[1]), with len <= num_y_dots
    :param pdf: Probability distribution function used to compute the quantiles.
    :param quantiles: List of quantiles to be computed
    :return:
    """

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


def _get_quantile_curves_qual(x: np.ndarray, y: np.ndarray, quantiles, gaussian_kde):
    xs = np.linspace(np.min(x), np.max(x), 2 ** 8)
    pdf = scipy.stats.gaussian_kde((x, y))
    q_res = _qet_quantiles_kde(
        x=xs,
        y=(np.min(y), np.max(y)),
        pdf=pdf,
        quantiles=quantiles,
        num_y_dots=2 ** 16
    )

    return xs, q_res


def _get_quantile_curves(x: np.ndarray, y: np.ndarray, quantiles: list[float], use_fast_kde: bool, gaussian_kde = None, fastkde = None):
    if use_fast_kde:
        assert fastkde is not None
        return _get_quantile_curves_fast(x, y, quantiles, fastkde=fastkde)
    
    # assert gaussian_kde is not None
    return _get_quantile_curves_qual(x, y,quantiles, gaussian_kde=gaussian_kde)


def _get_quantiles_list(confidence_prob: list[float]):
    q = [0.5]
    for p in confidence_prob:
        q.append(0.5 - p / 2)
        q.append(0.5 + p / 2)

    return q


def _plot_quantile_curve(
        x: np.ndarray, y: np.ndarray,
        x_label: str, y_label: str,
        confidence_prob: list[float],
        use_fast_kde: bool,
        gaussian_kde = None,
        fastkde = None,
        ax=None,
        rug_plot: bool = True,
        plot_optimal: bool = False,
        plot_mode: Literal['raw', 'deviation', 'relative'] = 'raw',
        show_legend: bool = True,
        alpha: float = 0.25,
):
    if ax is None:
        ax = plt.gca()
    
    xs, quantiles = _get_quantile_curves(x, y, _get_quantiles_list(confidence_prob), use_fast_kde, gaussian_kde, fastkde)

    plotter = _get_plotter(plot_mode, eps=1e-6)

    if plot_optimal:
        ax.plot(xs, plotter(xs, xs), c='r', label='Best regressor', linestyle='--')

    ax.plot(xs, plotter(xs, quantiles[0.5]), c='b', label='median')

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (p_u, p_d) in enumerate(_gen_proba_bound_pairs(confidence_prob)):
        c = colors[i if i < len(confidence_prob) else 2 * len(confidence_prob) - i - 1]
        label = f'{(p_u - 0.5) * 2 * 100:.1f}%' if p_u > 0.5 else None
        # print(f"p_u: {p_u}, p_d: {p_d}, c: {c}")
        ax.fill_between(xs, plotter(xs, quantiles[p_d]), plotter(xs, quantiles[p_u]),
                        alpha=alpha, label=label, color=c)

    if rug_plot:
        sns.rugplot(x=x, ax=ax, label='Distribution of actual values')
        if plot_mode == 'raw':
            sns.rugplot(y=y, ax=ax, label='Distribution of predicted values')
        elif plot_mode == 'deviation':
            sns.rugplot(y=y - x, ax=ax, label='Distribution of predicted values')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if show_legend:
        ax.legend()
