import numpy as np
from sklearn.linear_model import LogisticRegression


def get_params(X, y):
    """
        X: {np.array} Unbiased data
        y: {np.array}
    """
    n, m = X.shape
    lr = LogisticRegression()
    lr.fit(X, y)

    # w = [lr.intercept_[0]]
    # for c in lr.coef_[0]:
    #     w.append(c)
    w = np.squeeze(lr.coef_)

    return np.array(w)
