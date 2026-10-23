import json, time, os, numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix



DATA = './data/'

RES = './results/'

os.makedirs(RES, exist_ok=True)

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
from threadpoolctl import threadpool_limits          # also limit BLAS/OpenMP pools (numpy, sklearn, shap)
_pools = threadpool_limits(int(os.environ.get("THREADS", "1")))
TAU = (0.95, 0.90)       # early-exit thresholds of Exit 1 and Exit 2
TAU_ESC = 0.85           # Exit-3 escalation threshold of the mitigation policy



def load(name):
    z = np.load(DATA + f'{name}.npz', allow_pickle=True)
    return {k: z[k] for k in z.files}



def seed_all(s):
    np.random.seed(s); torch.manual_seed(s)




def metrics(y, p_att, conf=None):
    """Binary metrics with 'attack' as the positive class."""
    pred = (p_att >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / len(y)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    fpr = fp / max(fp + tn, 1)
    try:
        auc = roc_auc_score(y, p_att)
    except ValueError:
        auc = float('nan')
    out = dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=auc, fpr=fpr,
               tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    if conf is not None:
        out['ece'] = ece(conf, pred == y)
    return {k: (float(v) if not isinstance(v, int) else v) for k, v in out.items()}




def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)





@torch.no_grad()
def logits_all(model, X, bs=4096):
    model.eval(); outs = None
    for i in range(0, len(X), bs):
        z = model(torch.from_numpy(X[i:i + bs]))
        if outs is None:
            outs = [[] for _ in z]
        for k, zk in enumerate(z):
            outs[k].append(zk)
    return [torch.cat(o) for o in outs]





def fit_temperature(z, y):
    """The   temperature scaling: grid search of T minimising validation NLL."""
    y = torch.from_numpy(y)
    grid = np.exp(np.linspace(np.log(0.05), np.log(10), 200))
    nll = [F.cross_entropy(z / t, y).item() for t in grid]
    return float(grid[int(np.argmin(nll))])


def per_class(yc, pred):
    out = {}
    for c in np.unique(yc):
        m = yc == c
        out[str(c)] = dict(n=int(m.sum()), flagged=float(pred[m].mean()))
    return out






def dump(obj, path):
    json.dump(obj, open(path, 'w'), indent=1, default=float)
