#import numpy as np

    
#for SVM where labels are [-1,1]
import numpy as np
def metric(y_true, y_pred):
    y_true = y_true.astype("int")
    # The try except is needed because when the metric is batched some batches
    # have one class only
    try:
        # return roc_auc_score(y_true, y_pred)
        # proposed modification in order to get a metric that calcs on center 2
        # (y=1 only on that center)
        return ((y_pred > 0) == y_true).mean() #0 for SVM, 0.5 otherwise
    except ValueError:
        return np.nan

