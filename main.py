import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from pva import plot_precision_curve


def main():
    X, y = make_regression(n_samples=100_000, n_features=1, noise=2, random_state=42)

    if np.min(y) < 0:
        y -= np.min(y) - 10

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    ridge = Ridge(alpha=0.5)

    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)

    y_pred += 100

    print(ridge.coef_)

    _, ax = plt.subplots(ncols=2, sharey=True, sharex=True, figsize=(14, 5))

    # plot_precision_curve(y_true=y_test, y_pred=y_pred, plot_optimal=True, rug_plot=True, show_legend=False,
    #                      plot_mode='deviation', strategy='kde', ax=ax[0])

    plot_precision_curve(y_true=y_test, y_pred=y_pred, plot_optimal=True, rug_plot=True, show_legend=False,
                         plot_mode='deviation', strategy='fastkde', ax=ax[1])

    plt.show()


if __name__ == '__main__':
    main()
