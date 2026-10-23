"""THE   Re-fit the (deterministic)  training-split preprocessing and save its parameters, 
so that the per-flow preprocessing latency can be timed with the real transform......"""

import pickle, sys, numpy as np
from sklearn.model_selection import train_test_split
from prep import load, dedup, Prep, SEED, OUT



for name in sys.argv[1:]:
    X, y, ycat, n_raw, n_cols, cfg = load(name)
    keep, _ = dedup(X, y, ycat)
    X, y = X[keep].reset_index(drop=True), y[keep]
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    tr, va = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=SEED)
    p = Prep().fit(X.iloc[tr], cfg['cat'])
    raw_test = X.iloc[te[:5000]].reset_index(drop=True)
    pickle.dump(dict(prep=p, raw_test=raw_test), open(OUT + f'{name}_prep.pkl', 'wb'))
    print(name, len(p.names))



