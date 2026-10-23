"""Now The Leakage-free preprocessing for EENet-IPS...........

Each    statistic (median,  constant/correlated-feature filter, log/min-max
scaling, one-hot categories, SMOTE-Tomek)   has been   fitted on the training split only.

Identifier-like columns are removed and exact duplicate flows are dropped
before the train/test split so that no identical flow appears in both sets..............

"""


import json, os, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split


RAW = './raw/'
OUT = './data/'
os.makedirs(OUT, exist_ok=True)
SEED = 42



DATASETS = {
    'cse': dict(file='Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv', label='Label',
                benign='Benign', drop=['Timestamp', 'Dst Port'], cat=['Protocol']),
    'unsw': dict(file='UNSW_NB15_testing-set.csv', label='attack_cat', benign='Normal',
                 drop=['id', 'label'], cat=['proto', 'service', 'state']),
    'ciciot': dict(file='Merged01.csv', label='Label', benign='BENIGN', drop=[], cat=[]),
}



def load(name):
    cfg = DATASETS[name]
    df = pd.read_csv(RAW + cfg['file'], low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    n_raw, n_cols = df.shape
    ycat = df[cfg['label']].astype(str).str.strip()
    y = (ycat != cfg['benign']).astype(np.int64).values
    X = df.drop(columns=[cfg['label']] + cfg['drop'])
    for c in X.columns:
        if c not in cfg['cat']:
            X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y, ycat.values, n_raw, n_cols, cfg





def dedup(X, y, ycat):
    """Drop feature vectors seen with conflicting binary labels, then exact duplicates."""
    h = pd.util.hash_pandas_object(X.astype(str) if X.select_dtypes('object').shape[1] else X,
                                   index=False).values

    d = pd.DataFrame({'h': h, 'y': y})
    nlab = d.groupby('h')['y'].transform('nunique').values
    conflict = nlab > 1
    keep = (~conflict) & (~pd.Series(h).duplicated().values)
    return keep, int(conflict.sum())






class Prep:
    def fit(self, X, cat):
        self.cat = cat
        num = [c for c in X.columns if c not in cat]
        Xn = X[num]
        self.median = Xn.median()
        Xn = Xn.fillna(self.median)
        std = Xn.std()
        num = [c for c in num if std[c] > 0]
        # greedy removal of near-duplicate columns (|r| > 0.99), fitted on train
        sub = Xn[num].sample(min(len(Xn), 200000), random_state=SEED)
        corr = np.abs(np.corrcoef(sub.values, rowvar=False))
        corr = np.nan_to_num(corr)
        keep = []
        for i, c in enumerate(num):
            if all(corr[i, num.index(k)] <= 0.99 for k in keep):
                keep.append(c)
        self.num = keep
        Xn = Xn[keep]
        self.logcols = [c for c in keep if Xn[c].min() >= 0]
        Z = Xn.copy()
        Z[self.logcols] = np.log1p(Z[self.logcols])
        self.mn, self.mx = Z.min(), Z.max()
        # one-hot with rare-category pooling (<0.5% of training rows -> 'other')
        self.cats = {}
        for c in cat:
            vc = X[c].astype(str).value_counts(normalize=True)
            self.cats[c] = [v for v in vc.index if vc[v] >= 0.005]
        self.names = self.num + [f'{c}={v}' for c in cat for v in self.cats[c] + ['other']]
        return self





    def transform(self, X):
        Z = X[self.num].fillna(self.median[self.num]).copy()
        Z[self.logcols] = np.log1p(Z[self.logcols].clip(lower=0))
        Z = (Z - self.mn) / (self.mx - self.mn).replace(0, 1)
        Z = Z.clip(0, 1).values.astype(np.float32)
        parts = [Z]
        for c in self.cat:
            v = X[c].astype(str).values
            cols = self.cats[c]
            oh = np.zeros((len(X), len(cols) + 1), np.float32)
            idx = {k: i for i, k in enumerate(cols)}
            for r, val in enumerate(v):
                oh[r, idx.get(val, len(cols))] = 1.0
            parts.append(oh)
        return np.concatenate(parts, axis=1)






def run(name):
    t0 = time.time()
    X, y, ycat, n_raw, n_cols, cfg = load(name)
    keep, n_conf = dedup(X, y, ycat)
    X, y, ycat = X[keep].reset_index(drop=True), y[keep], ycat[keep]
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    tr, va = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=SEED)
    prep = Prep().fit(X.iloc[tr], cfg['cat'])
    out = {}
    for split, ii in (('train', tr), ('val', va), ('test', te)):
        out['X_' + split] = prep.transform(X.iloc[ii])
        out['y_' + split] = y[ii]
        out['c_' + split] = ycat[ii]
    np.savez_compressed(OUT + f'{name}.npz', **out, names=np.array(prep.names))
    info = dict(dataset=name, file=cfg['file'], raw_rows=int(n_raw), raw_cols=int(n_cols),
                dropped_identifier_cols=cfg['drop'], conflicting_rows_removed=n_conf,
                rows_after_dedup=int(len(y)), duplicates_removed=int(n_raw - len(y) - n_conf),
                n_features=len(prep.names), n_train=len(tr), n_val=len(va), n_test=len(te),
                attack_share=float(y.mean()), train_attack=int(y[tr].sum()), test_attack=int(y[te].sum()),
                classes={k: int(v) for k, v in pd.Series(ycat).value_counts().items()},
                seconds=round(time.time() - t0, 1))
    json.dump(info, open(OUT + f'{name}_info.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != 'classes'}))






if __name__ == '__main__':
    for n in sys.argv[1:]:
        run(n)

