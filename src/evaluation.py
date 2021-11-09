from sklearn.metrics import mean_absolute_percentage_error
from plotting import plot_evaluation_results
from sklearn.model_selection import cross_val_score, TimeSeriesSplit


def compute_mape(y_test, y_pred):
    return mean_absolute_percentage_error(y_pred, y_test)


def evaluate_model(model, X_train, y_train, X_test, y_test, postfix, need_plot=False):
    y_pred = model.predict(X_test)
    mape = compute_mape(y_test, y_pred)
    if need_plot:
        tscv = TimeSeriesSplit(n_splits=5)
        cv = cross_val_score(
            model,
            X_train,
            y_train,
            cv=tscv,
            scoring="neg_mean_absolute_error"
        )
        plot_evaluation_results(
            y_pred,
            y_test,
            error_value=mape,
            cross_validation=cv,
            postfix=postfix
        )
    return mape
