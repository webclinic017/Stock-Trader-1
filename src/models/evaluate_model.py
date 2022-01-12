import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_estimators_cvperf(estimators_list, figsize=(12, 6), error_metric=''):
    data = [pd.DataFrame(estimator.cv_results_)['mean_test_score'].dropna().values for estimator in estimators_list]
    estimator_names = [str(estimator.estimator).replace('()', '') for estimator in estimators_list]
        
    plt.figure(figsize=figsize)
    plt.title('Model(s) CV performance')
    plt.boxplot(data)
    plt.xticks([i+1 for i in range(len(estimator_names))], estimator_names)
    plt.xlabel('model techniques')
    plt.ylabel(error_metric)
    plt.tight_layout()
    plt.show()
