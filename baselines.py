"""The classical ML baseline techniques that have been  trained under the same protocol (same splits, same training data; SMOTE-Tomek on the
UNSW-NB15 training split only). Like our proposed EENet-IPS, each model has been trained three times (seeds 1-3;  seed s is utilized for
the resampling and for the model). The JSON holds the mean metrics of each given model, their standard deviation over
the runs ('sd'), the individual runs ('runs'), and, in the 'hgb_fl_cost', the communication that the a histogram-based
federated  GBDT will be needed in order  to rebuild the seed-1 gradient-boosting model   (tree_fl_cost.py).
usage: python baselines.py <dataset>  ......    """

import sys, time, numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from common import *
from train import resample
from tree_fl_cost import histogram_fl_cost



name = sys.argv[1]
D = load(name)
R = {'lr': [], 'rf': [], 'hgb': []}; cost = None
for s in (1, 2, 3):
    Xtr, ytr = resample(name, D['X_train'], D['y_train'], s)
    for key, clf in [('lr', LogisticRegression(max_iter=2000)),
                     ('rf', RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=s)),
                     ('hgb', HistGradientBoostingClassifier(max_iter=200, random_state=s))]:
        t0 = time.time(); clf.fit(Xtr, ytr); tt = time.time() - t0
        p = clf.predict_proba(D['X_test'])[:, 1]
        if name == 'unsw':      # same prior correction as the deep models (training split balanced by SMOTE-Tomek)
            pv, pt = D['y_val'].mean(), ytr.mean(); pc = np.clip(p, 1e-6, 1 - 1e-6)
            p = 1 / (1 + np.exp(-(np.log(pc / (1 - pc)) + np.log(pv / (1 - pv)) - np.log(pt / (1 - pt)))))
        m = metrics(D['y_test'], p, np.maximum(p, 1 - p)); m.update(seed=s, train_sec=tt)
        R[key].append(m)
        if key == 'hgb' and s == 1:
            cost = histogram_fl_cost(clf, n_clients=5)
        print(name, key, s, m, flush=True)

out = {}

for key, runs_ in R.items():
    num = [k for k, v in runs_[0].items() if isinstance(v, float) and k != 'train_sec']
    out[key] = {k: float(np.mean([r[k] for r in runs_])) for k in num}
    out[key]['sd'] = {k: float(np.std([r[k] for r in runs_], ddof=1)) for k in num}
    out[key]['runs'] = runs_
out['hgb_fl_cost'] = cost
dump(out, RES + f'{name}_baselines.json')


print('DONE', name, cost, flush=True)
