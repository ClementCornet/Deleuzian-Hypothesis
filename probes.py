try:
    # Try to import cuML and check for a CUDA device
    import cuml # type: ignore
    from cuml.linear_model import LogisticRegression as CumlLogisticRegression # type: ignore
    import numba.cuda # type: ignore

    if numba.cuda.is_available():
        LogisticRegression = CumlLogisticRegression
    else:
        raise ImportError("CUDA not available")
except ImportError:
    # Fallback to scikit-learn if cuML or CUDA is not available
    from sklearn.linear_model import LogisticRegression



from sklearn.metrics import log_loss
from rich import print
import torch
import numpy as np
from tqdm import tqdm
from torch.distributions import Normal


import warnings
warnings.filterwarnings('ignore')


def learn_probes(
        train_concept_space,
        test_concept_space,
        y_train,
        y_test
):
    """
    
    Returns:
        - scores : dict { 
                            <label_type: str>:  {
                                CrossEntropy : (float, float), ###### train, test 
                                GaussianF1   : (float, float)  ###### train, test 
                            }
                        }
    """

    train_no_test = set(y_train) - set(y_test)
    test_no_train = set(y_test) - set(y_train)
    if len(train_no_test) > 0: print(f'{len(train_no_test)} training labels not present in testing set : {train_no_test}')
    if len(test_no_train) > 0: print(f'{len(test_no_train)} testing labels not present in training set : {test_no_train}')
    label_set = sorted(list(set(y_train)))

    scores = {}

    for label in tqdm(label_set, desc='Learning Probes'):
        if label not in  set(y_train).intersection(set(y_test)): continue
        y_train_label = [yy == label for yy in y_train]
        y_test_label = [yy == label for yy in y_test]

        best_ce_train = np.inf
        best_f1_train = None
        best_ce_test  = np.inf
        best_f1_test  = None

        for feat_train, feat_test in zip(train_concept_space.T, test_concept_space.T):
            X_train = torch.tensor(feat_train).unsqueeze(1).cpu().numpy().astype(np.float32)
            X_test  = torch.tensor(feat_test).unsqueeze(1).cpu().numpy().astype(np.float32)
            clf = LogisticRegression().fit(X_train, y_train_label)
            ce_train  = log_loss(y_train_label, clf.predict_proba(X_train))
            if ce_train < best_ce_train:
                best_ce_train = ce_train
                best_ce_test = log_loss(y_test_label, clf.predict_proba(X_test))
    
        scores[label] = {'Crossentropy':(best_ce_train, best_ce_test)}
        print(f'{label} | {scores[label]}')

    return scores