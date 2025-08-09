import time
from sklearn.preprocessing import OneHotEncoder
import numpy as np
seed_value = 0

def wrapper(direc, X, y):
    """
    Create, train and test model, use SHAP explainer on model, save SHAP values
    :param direc: Result directory
    :param X: Data
    :param y: Labels
    """
    t0 = time()
    print("X shape: ", X.shape, "y shape: ", y.shape, "n target classes (before OHE): ", np.unique(y), "\n")
    enc = OneHotEncoder()
    y = enc.fit_transform(y.reshape(-1, 1)).toarray()
    X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=0.1, random_state=seed_value)
    k = 5
    n_output = 3
    print("Training....")
    model, foldperf = perform_cv(X_training, y_training, k, n_output, direc, (768, 128))
    performance_plots(foldperf, k, direc)
    test_model(model, X_test, y_test, k, direc)
    y_test = enc.inverse_transform(y_test)
    confusionmatrix(model, X_test, y_test, direc)
    print("\nTime to train model: ", (time() - t0) / 60, "mins")

    t1 = time()
    print("\nShap explainer...")
    explainer, shap_values = explain(model, X_training, X_test)
    shap_values = np.array(shap_values)
    p = direc + 'shap_cnn.npy'
    with open(p, 'wb') as h:
        np.save(h, shap_values)
    print("\nTime to train SHAP: ", (time() - t1) / 60, "mins")