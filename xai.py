"""THE   Exit-specific explanations and their cost/quality. .......

Exit 1:   KernelSHAP (100 coalition samples, 50 k-means background centroids)
Exit 2:   Integrated Gradients (zero baseline, 256-step trapezoidal Riemann sum)
Exit 3:  constrained counterfactual for flows that reach Exit 3 (box [0,1], one-hot/protocol features frozen,
         L1-regularised gradient search on the logit margin, at most 100 Adam steps, followed by a
         greedy pass that undoes the smallest changes while the class stays flipped)

usage:   python xai.py <dataset> [seed]
"""


import sys, time, warnings, numpy as np, torch, torch.nn.functional as F, shap
from captum.attr import IntegratedGradients
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import *
from model import EENet
warnings.filterwarnings('ignore')


FIG = './figs/'; os.makedirs(FIG, exist_ok=True)
PRETTY = dict(cse='CSE-CIC-IDS2018', unsw='UNSW-NB15', ciciot='CICIoT2023')




def counterfactual(m, x0, T3, mut, steps=100, lr=0.02, lam=0.01, margin=0.05):
    x0 = torch.from_numpy(x0).unsqueeze(0)
    with torch.no_grad():
        target = 1 - int(m.exit3(x0).argmax(1))
    xc = x0.clone().requires_grad_(True)
    opt = torch.optim.Adam([xc], lr=lr)
    kappa = float(np.log((0.5 + margin) / (0.5 - margin)))   # logit margin that gives p = 0.5 + margin
    for s in range(steps):
        z = m.exit3(xc)[0] / T3
        if F.softmax(z, 0)[target].item() >= 0.5 + margin:
            break
        # hinge on the calibrated logit margin (no vanishing gradient for confident flows) + L1 distance
        loss = F.relu(z[1 - target] - z[target] + kappa) + lam * (xc - x0).abs().sum()
        opt.zero_grad(); loss.backward()
        xc.grad *= mut
        opt.step()
        with torch.no_grad():
            xc.clamp_(0, 1); xc[:, ~mut.bool()] = x0[:, ~mut.bool()]
    with torch.no_grad():
        p = F.softmax(m.exit3(xc) / T3, 1)[0, target].item()
        if p >= 0.5:                      # greedy sparsification: undo the smallest changes
            xc = xc.detach().clone()      # while the prediction stays flipped
            for j in torch.argsort((xc - x0).abs()[0]).tolist():
                if (xc[0, j] - x0[0, j]).abs() <= 1e-6:
                    continue
                old = xc[0, j].item(); xc[0, j] = x0[0, j]
                if F.softmax(m.exit3(xc) / T3, 1)[0, target].item() < 0.5:
                    xc[0, j] = old
            p = F.softmax(m.exit3(xc) / T3, 1)[0, target].item()
    dx = (xc - x0).detach().abs()[0]
    viol = int(((xc < 0) | (xc > 1)).sum() + (dx[~mut.bool()] > 0).sum())
    return p >= 0.5, int((dx > 0.01).sum()), float(dx.sum()), viol, s + 1




def run(name, seed=1, n_time=100, n_summary=300, n_cf=200):
    D = load(name); names = [str(n) for n in D['names']]
    r = json.load(open(RES + f'{name}_eenet_s{seed}.json')); T = r['T']
    pz = np.load(RES + f'{name}_eenet_s{seed}_pred.npz')
    d = D['X_train'].shape[1]
    m = EENet(d); m.load_state_dict(torch.load(RES + f'{name}_eenet_s{seed}.pt')); m.eval()
    rng = np.random.default_rng(0)
    Xte = D['X_test']
    out = dict(dataset=name, seed=seed)

# ---------====- Exit 1: The KernelSHAP ---
    f1 = lambda X: F.softmax(m.exit1(torch.from_numpy(np.asarray(X, np.float32))) / T[0], 1)[:, 1].detach().numpy()
    bg = shap.kmeans(D['X_train'][rng.choice(len(D['X_train']), 5000, replace=False)], 50)
    ex = shap.KernelExplainer(f1, bg)
    idx = rng.choice(len(Xte), n_summary, replace=False)
    t = []
    with torch.no_grad():
        for i in idx[:n_time]:
            t0 = time.perf_counter()
            ex.shap_values(Xte[i:i + 1], nsamples=100, l1_reg='num_features(10)', silent=True)
            t.append(time.perf_counter() - t0)
        SV = np.asarray(ex.shap_values(Xte[idx], nsamples=100, l1_reg='num_features(10)', silent=True))
        gap = np.abs(SV.sum(1) + ex.expected_value - f1(Xte[idx]))
    out['shap'] = dict(ms_mean=1e3 * np.mean(t), ms_std=1e3 * np.std(t), ms_p95=1e3 * np.percentile(t, 95),
                       local_acc_gap=float(gap.mean()),
                       top=[names[j] for j in np.argsort(-np.abs(SV).mean(0))[:10]])
 

   # summary  (beeswarm) and SHAP-correlation heatmap figures.....
    plt.figure()
    shap.summary_plot(SV, Xte[idx], feature_names=names, max_display=15, show=False, plot_size=(6.5, 6))
    plt.title(PRETTY[name], fontsize=12); plt.tight_layout()
    plt.savefig(FIG + f'shap_{name}.png', dpi=200); plt.close('all')
    top = np.argsort(-np.abs(SV).mean(0))[:12]
    C = np.corrcoef(SV[:, top], rowvar=False); C = np.nan_to_num(C)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(top))); ax.set_yticks(range(len(top)))
    ax.set_xticklabels([names[j] for j in top], rotation=60, ha='right', fontsize=8)
    ax.set_yticklabels([names[j] for j in top], fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Correlation of SHAP values')
    ax.set_title(PRETTY[name], fontsize=11); fig.tight_layout()
    fig.savefig(FIG + f'shapcorr_{name}.png', dpi=200); plt.close(fig)


    # -#####3-- Exit 2: Integrated Gradients -#############--
    ig = IntegratedGradients(lambda x: m.exit2(x)[:, 1])
    t, rel = [], []
    for i in idx[:n_time]:
        x = torch.from_numpy(Xte[i:i + 1])
        t0 = time.perf_counter()
        a, delta = ig.attribute(x, baselines=torch.zeros_like(x), n_steps=256,
                                method='riemann_trapezoid', return_convergence_delta=True)
        t.append(time.perf_counter() - t0)
        with torch.no_grad():
            diff = (m.exit2(x)[:, 1] - m.exit2(torch.zeros_like(x))[:, 1]).abs().item()
        rel.append(abs(delta.item()) / max(diff, 1e-8))
    out['ig'] = dict(ms_mean=1e3 * np.mean(t), ms_std=1e3 * np.std(t), ms_p95=1e3 * np.percentile(t, 95),
                     completeness_rel_err=float(np.median(rel)), completeness_rel_err_mean=float(np.mean(rel)))



   #---- Exit 3: constrained counterfactuals -+++++++++++++
    mut = torch.tensor([0. if ('=' in n or n == 'Protocol Type') else 1. for n in names])
    ex3 = np.where(pz['route_exit'] == 3)[0]
    if len(ex3) >= n_cf:                  # flows that reach Exit 3 (random subset)
        sel = rng.permutation(ex3)[:n_cf]
    else:                                 # too few: complete with the least confident flows at Exit 3
        conf3 = pz['probs'][2].max(1); s3 = set(ex3.tolist())
        fill = [i for i in np.argsort(conf3) if i not in s3][:n_cf - len(ex3)]
        sel = np.concatenate([ex3, np.array(fill, dtype=ex3.dtype)])
    res = []
    for i in sel:
        t0 = time.perf_counter()
        v, sp, l1, viol, it = counterfactual(m, Xte[i], T[2], mut)
        res.append((v, sp, l1, viol, it, time.perf_counter() - t0))
    res = np.array(res, dtype=float)
    out['cf'] = dict(n=len(sel), from_exit3=int(min(len(ex3), n_cf)), validity=float(res[:, 0].mean()),
                     sparsity_mean=float(res[res[:, 0] == 1, 1].mean()) if res[:, 0].any() else None,
                     l1_mean=float(res[res[:, 0] == 1, 2].mean()) if res[:, 0].any() else None,
                     violations=int(res[:, 3].sum()), iters_mean=float(res[:, 4].mean()),
                     ms_mean=1e3 * res[:, 5].mean(), ms_std=1e3 * res[:, 5].std(),
                     ms_p95=1e3 * np.percentile(res[:, 5], 95))
    dump(out, RES + f'{name}_xai.json')
    print(json.dumps(out, default=float))


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)
