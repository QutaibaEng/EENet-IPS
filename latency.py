"""The   Latency protocol and the   flow-volume (scalability) test.......



All of the   timings:   one CPU thread, model in eval mode, no gradient, 100 warm-up runs.

Per-flow latency: batch size 1, 1,000 test flows, cumulative time of each exit path.

Throughput: routed inference over the whole test set in micro-batches of 1,024.

usage: python latency.py <dataset> [seed]     |   python latency.py volume cse [seed]
"""


import sys, time, pickle, numpy as np, torch, psutil, gc
from common import *
from model import EENet




def load_model(name, seed):
    D = load(name); r = json.load(open(RES + f'{name}_eenet_s{seed}.json'))
    m = EENet(D['X_train'].shape[1]); m.load_state_dict(torch.load(RES + f'{name}_eenet_s{seed}.pt')); m.eval()
    return D, r['T'], m




def mitigation(exit_k, conf, attack):
    """Rule-based policy of Section 3.3 (decision only; enforcement is not executed)."""
    if not attack:
        return 'forward'
    if exit_k == 1:
        return 'block'
    if exit_k == 2:
        return 'quarantine+alert'
    return 'block' if conf >= TAU_ESC else 'escalate'







@torch.no_grad()
def per_flow(name, seed, n=1000, warm=100):
    D, T, m = load_model(name, seed)
    X = D['X_test']; rng = np.random.default_rng(1); idx = rng.choice(len(X), n + warm, replace=False)
    t = {1: [], 2: [], 3: []}; tr = []
    for j, i in enumerate(idx):
        x = torch.from_numpy(X[i:i + 1])
        t0 = time.perf_counter(); a1 = m.s1(x); z1 = m.h1(a1.mean(1)); t1 = time.perf_counter()
        a2 = m.s2(a1); z2 = m.h2(a2.mean(1)); t2 = time.perf_counter()
        a3 = m.s3(a2); z3 = m.h3(a3.mean(1)); t3 = time.perf_counter()
        t0r = time.perf_counter(); p, e, c = m.route_single(x, T, TAU)
        act = mitigation(e, c, bool(p[0, 1] >= .5)); t1r = time.perf_counter()
        if j >= warm:
            t[1].append(t1 - t0); t[2].append(t2 - t0); t[3].append(t3 - t0); tr.append(t1r - t0r)
    # preprocessing (scaling of an already-extracted flow record), numpy fast path
    P = pickle.load(open(DATA + f'{name}_prep.pkl', 'rb')); prep, raw = P['prep'], P['raw_test']
    med = prep.median[prep.num].values; lg = np.isin(prep.num, prep.logcols)
    mn, mx = prep.mn.values, prep.mx.values; rng_ = np.where(mx - mn == 0, 1, mx - mn)
    rawn = raw[prep.num].values.astype(np.float64)
    tp = []
    for k in range(warm + n):
        v = rawn[k % len(rawn)].copy(); t0 = time.perf_counter()
        v = np.where(np.isnan(v), med, v); v[lg] = np.log1p(np.clip(v[lg], 0, None))
        v = np.clip((v - mn) / rng_, 0, 1); tp.append(time.perf_counter() - t0)
    tp = tp[warm:]
    # throughput: routed batches of 1,024 over the full test set
    Xt = torch.from_numpy(X); t0 = time.perf_counter()
    for i in range(0, len(X), 1024):
        m.route(Xt[i:i + 1024], T, TAU)
    thr = len(X) / (time.perf_counter() - t0)
    f = lambda a: dict(ms_mean=1e3 * np.mean(a), ms_std=1e3 * np.std(a), ms_p95=1e3 * np.percentile(a, 95))
    out = dict(dataset=name, seed=seed, path={k: f(v) for k, v in t.items()}, routed_single=f(tr),
               preprocess=f(tp), throughput_fps=thr, batch_ms_per_flow=1e3 / thr,
               cpu=psutil.cpu_count(logical=True), threads=torch.get_num_threads())
    dump(out, RES + f'{name}_latency.json'); print(json.dumps(out, default=float))






@torch.no_grad()
def volume(name, seed, sizes=(10_000, 50_000, 100_000, 500_000, 1_000_000), reps=3):
    D, T, m = load_model(name, seed)
    X, y = D['X_test'], D['y_test']; proc = psutil.Process(); res = []
    for rep in range(reps):              # repetitions in the outer loop, so that slow drifts of the
        for N in sizes:                  # machine affect all sizes alike
            rng = np.random.default_rng(100 + rep); idx = rng.integers(0, len(X), N)
            gc.collect(); base = proc.memory_info().rss; peak = base
            ex = np.empty(N, np.int8); pr = np.empty(N, np.float32)
            t0 = time.perf_counter()
            for i in range(0, N, 1024):
                b = idx[i:i + 1024]
                p, e, c = m.route(torch.from_numpy(X[b]), T, TAU)
                ex[i:i + len(b)] = e.numpy(); pr[i:i + len(b)] = p[:, 1].numpy()
                peak = max(peak, proc.memory_info().rss)
            el = time.perf_counter() - t0
            res.append(dict(N=N, rep=rep, sec=el, fps=N / el, ms_per_flow=1e3 * el / N,
                            peak_MB=peak / 1e6, acc=float(((pr >= .5) == y[idx]).mean()),
                            share1=float((ex == 1).mean())))
            print(res[-1], flush=True)
    dump(dict(dataset=name, seed=seed, runs=res), RES + f'{name}_volume.json')




if __name__ == '__main__':
    if sys.argv[1] == 'volume':
        volume(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    else:
        per_flow(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)


