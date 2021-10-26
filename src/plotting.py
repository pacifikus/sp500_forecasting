import matplotlib.pyplot as plt
import numpy as np
import uuid


def plot_evaluation_results(y_pred, y_test, error_value, cross_validation, postfix):
    plt.figure(figsize=(15, 7))
    plt.plot(y_pred, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test, label="actual", linewidth=2.0)

    mae = cross_validation.mean() * (-1)
    deviation = cross_validation.std()

    scale = 1.96
    lower = y_pred - (mae + scale * deviation)
    upper = y_pred + (mae + scale * deviation)

    plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
    plt.plot(upper, "r--", alpha=0.5)

    anomalies = np.array([np.NaN] * len(y_test))
    anomalies[y_test<lower] = y_test[y_test < lower]
    anomalies[y_test>upper] = y_test[y_test > upper]
    plt.plot(anomalies, "o", markersize=10, label="Anomalies")

    plt.title("Mean absolute percentage error {0:.2f}%".format(error_value))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f'imgs/evaluation_result_{postfix}.png')
