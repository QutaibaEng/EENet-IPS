"""The  Federated training of the EENet-IPS backbone (in-process simulation).......



usage: python fl.py <dataset> <config> <seed>


config: fedavg     - plain FedAvg (dense updates, no perturbation)..
        fedprox   - FedProx baseline (mu = 0.001), dense updates.   
        eenetfl    - FedAvg + update clipping + Gaussian noise + top-k sparsification
                       with error feedback (the EENet-IPS federated configuration)  .....



Label skew:   on UNSW-NB15 each client balances its data with SMOTE-Tomek (as in
centralized training). On CSE-CIC-IDS2018 and CICIoT2023 (no resampling), clients
train with logit-adjusted cross-entropy (log local class prior added to the logits).
In both cases the server adds the log class prior of the validation split to the exit logits.
The global model of the round with the lowest validation loss is kept.

"""



import sys, time, copy, numpy as np, torch, torch.nn.functional as F
from common import *
from model import EENet, add_prior
from train import evaluate



K, PER_ROUND, LOCAL_EPOCHS, MAX_STEPS, BS, LR = 5, 4, 1, 50, 512, 2e-3
ALPHA, MU, CLIP, SIGMA, TOPK = 0.5, 0.001, 5.0, 0.01, 0.10





def dirichlet_split(y, k, alpha, rng):
    parts = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        p = rng.dirichlet(alpha * np.ones(k))
        cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
        for j, chunk in enumerate(np.split(idx, cuts)):
            parts[j].extend(chunk.tolist())
    return [np.array(sorted(p)) for p in parts]




def flat(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])





def set_flat(model, v):
    i = 0
    for p in model.parameters():
        n = p.numel(); p.data.copy_(v[i:i + n].view_as(p)); i += n







def flat_params(model):
    return torch.cat([p.view(-1) for p in model.parameters()])





def log_prior(y):
    return torch.log(torch.bincount(torch.as_tensor(y), minlength=2).float().clamp(min=1) / len(y))





def local_train(model, X, y, global_vec, prox, seed, lp=None):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    model.train(); steps = 0
    for _ in range(LOCAL_EPOCHS):
        perm = torch.randperm(len(yt), generator=g)
        for i in range(0, len(perm), BS):
            if steps >= MAX_STEPS:          # one local epoch, capped at MAX_STEPS updates
                return
            steps += 1
            b = perm[i:i + BS]
            zs = model(Xt[b])
            if lp is not None:              # logit-adjusted loss for label-skewed clients
                zs = [z + lp for z in zs]
            loss = sum(F.cross_entropy(z, yt[b]) for z in zs) / len(zs)
            if prox:
                loss = loss + MU / 2 * (flat_params(model) - global_vec).pow(2).sum()
            opt.zero_grad(); loss.backward(); opt.step()







def run(name, cfg, seed):
    ROUNDS = int(os.environ.get('ROUNDS', 60 if name == 'unsw' else 15))
    D = load(name); seed_all(seed); rng = np.random.default_rng(seed)
    parts = dirichlet_split(D['y_train'], K, ALPHA, rng)
    resample = name == 'unsw'
    clients = []

    for idx in parts:
        X, y = D['X_train'][idx], D['y_train'][idx]
        if resample and len(np.unique(y)) == 2 and min(np.bincount(y)) > 6:
            from imblearn.combine import SMOTETomek   # local resampling on each client
            X, y = SMOTETomek(random_state=seed).fit_resample(X, y)
            X = X.astype(np.float32)
        clients.append((X, y, None if resample else log_prior(y)))
    lp_server = log_prior(D['y_val'])     # clients learn balanced logits (SMOTE-Tomek or logit adjustment)
    gmodel = EENet(D['X_train'].shape[1]); P = sum(p.numel() for p in gmodel.parameters())
    resid = [torch.zeros(P) for _ in range(K)]
    k_send = int(TOPK * P)
    hist, up_bytes, t0 = [], 0, time.time()
    best = (float('inf'), 0, None)
    yv = torch.from_numpy(D['y_val'])
    for r in range(1, ROUNDS + 1):
        chosen = rng.choice(K, PER_ROUND, replace=False)
        gvec = flat(gmodel).clone(); agg = torch.zeros(P); ntot = 0; norms = []
        for c in chosen:
            m = copy.deepcopy(gmodel)
            X, y, lp = clients[c]
            local_train(m, X, y, gvec, cfg == 'fedprox', seed * 1000 + r * 10 + int(c), lp)
            delta = flat(m) - gvec; norms.append(float(delta.norm()))
            if cfg == 'eenetfl':
                delta = delta * min(1.0, CLIP / (delta.norm().item() + 1e-12))
                acc = delta + resid[c]
                top = acc.abs().topk(k_send).indices
                sent = torch.zeros(P); sent[top] = acc[top]
                resid[c] = acc - sent
                sent[top] += torch.randn(k_send) * SIGMA
                delta = sent; up_bytes += k_send * 8          # float32 value + int32 index
            else:
                up_bytes += P * 4
            n = len(y); agg += n * delta; ntot += n
        set_flat(gmodel, gvec + agg / ntot)
        em = add_prior(gmodel, lp_server)
        zv = logits_all(em, D['X_val'])
        vloss = float(sum(F.cross_entropy(z, yv) for z in zv) / 3)
        if vloss < best[0]:
            best = (vloss, r, copy.deepcopy(em.state_dict()))
        hist.append(dict(round=r, val_acc=[float((z.argmax(1) == yv).float().mean()) for z in zv],
                         val_loss=vloss, clients=[int(c) for c in chosen], delta_norm=norms,
                         up_MB=up_bytes / 1e6, sec=time.time() - t0))


        print(name, cfg, seed, hist[-1], flush=True)
    final = EENet(D['X_train'].shape[1]); final.load_state_dict(best[2]); final.eval()
    res, out = evaluate(final, D, 'eenet')
    res.update(dataset=name, config=cfg, seed=seed, history=hist, params=P, best_round=best[1],
               client_sizes=[int(len(c[1])) for c in clients],
               client_attack_share=[float(np.mean(D['y_train'][p])) for p in parts],
               upload_MB_total=up_bytes / 1e6, dense_round_MB=PER_ROUND * P * 4 / 1e6,
               setup=dict(K=K, rounds=ROUNDS, per_round=PER_ROUND, local_epochs=LOCAL_EPOCHS, max_steps=MAX_STEPS,
                          batch=BS, lr=LR, alpha=ALPHA, mu=MU, clip=CLIP, sigma=SIGMA, topk=TOPK,
                          label_skew='smote-tomek' if resample else 'logit-adjusted', selection='best val loss'),
               train_sec=time.time() - t0)
    tag = f'{name}_fl_{cfg}_s{seed}'
    dump(res, RES + tag + '.json')
    np.savez_compressed(RES + tag + '_pred.npz', **out)
    torch.save(final.state_dict(), RES + tag + '.pt')


    print('DONE', tag, 'best round', best[1], res['routed'], flush=True)






if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], int(sys.argv[3])) 



