"""THE centralized  training/evaluation of the EENet-IPS trchnique and the deep ablation models..

usage: python train.py <dataset> <kind> <seed>
kind:   eenet  |  noexit  | cnn | bilstm |  transformer

"""


import sys, time, copy, numpy as np, torch, torch.nn.functional as F
from common import *
from model import EENet, NoExitNet, SingleModel, add_prior




def build(kind, d):
    if kind == 'eenet':
        return EENet(d)
    if kind == 'noexit':
        return NoExitNet(d)
    return SingleModel(d, kind)





def resample(name, X, y, seed):
    if name != 'unsw':
        return X, y
    from imblearn.combine import SMOTETomek       # training split only
    Xr, yr = SMOTETomek(random_state=seed).fit_resample(X, y)
    return Xr.astype(np.float32), yr


# The  training settings; UNSW-NB15 batch size and learning rate selected on its validation split
CONFIG = dict(cse=dict(bs=512, lr=2e-3, max_epochs=25), ciciot=dict(bs=512, lr=2e-3, max_epochs=25),
              unsw=dict(bs=128, lr=0.002, max_epochs=60, patience=8))   # small, noisier validation split
PATIENCE = 4



def log_prior(y):
    return torch.log(torch.bincount(torch.as_tensor(y), minlength=2).float().clamp(min=1) / len(y))


def train(name, kind, seed, max_epochs=25, patience=PATIENCE, bs=512, lr=2e-3):
    D = load(name); seed_all(seed)
    Xtr, ytr = resample(name, D['X_train'], D['y_train'], seed)
    d = Xtr.shape[1]
    model = build(kind, d)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    hist, best, best_state, bad = [], 1e9, None, 0
    t_start = time.time()
    for ep in range(1, max_epochs + 1):
        model.train(); perm = torch.randperm(len(yt)); tl, tc, n = 0., 0, 0
        t0 = time.time()
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]; xb, yb = Xt[b], yt[b]
            zs = model(xb)
            loss = sum(F.cross_entropy(z, yb) for z in zs) / len(zs)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(b); tc += (zs[-1].argmax(1) == yb).sum().item(); n += len(b)
        zv = logits_all(model, D['X_val']); yv = torch.from_numpy(D['y_val'])
        vloss = float(sum(F.cross_entropy(z, yv) for z in zv) / len(zv))
        vacc = [float((z.argmax(1) == yv).float().mean()) for z in zv]
        hist.append(dict(epoch=ep, train_loss=tl / n, train_acc=tc / n, val_loss=vloss,
                         val_acc=vacc, sec=time.time() - t0, lr=opt.param_groups[0]['lr']))
        print(name, kind, seed, hist[-1], flush=True)
        sched.step(vloss)
        if vloss < best - 1e-5:
            best, best_state, bad = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    train_sec = time.time() - t_start
    if name == 'unsw':      # SMOTE-Tomek balanced the training prior: restore the prior of the validation split
        model = add_prior(model, log_prior(D['y_val']) - log_prior(ytr))
    return model, hist, train_sec, D




def evaluate(model, D, kind):
    zv = logits_all(model, D['X_val']); yv = D['y_val']
    T = [fit_temperature(z, yv) for z in zv]
    zt = logits_all(model, D['X_test']); yt = D['y_test']
    res = dict(T=T, exits={})
    probs = [F.softmax(z / t, 1).numpy() for z, t in zip(zt, T)]
    for k, p in enumerate(probs, 1):            # every exit on every test flow
        res['exits'][k] = metrics(yt, p[:, 1], p.max(1))
    out = dict(probs=np.stack(probs, 0).astype(np.float32))
    if kind == 'eenet':                         # routed early-exit system
        model.eval()
        prob, ex, conf = [], [], []
        for i in range(0, len(yt), 8192):
            p, e, c = model.route(torch.from_numpy(D['X_test'][i:i + 8192]), T, TAU)
            prob.append(p); ex.append(e); conf.append(c)
        prob = torch.cat(prob).numpy(); ex = torch.cat(ex).numpy(); conf = torch.cat(conf).numpy()
        res['routed'] = metrics(yt, prob[:, 1], conf)
        res['share'] = {k: float((ex == k).mean()) for k in (1, 2, 3)}
        res['exit_acc'] = {k: float(((prob[:, 1] >= .5) == yt)[ex == k].mean()) if (ex == k).any() else None
                           for k in (1, 2, 3)}
        res['exit_n'] = {k: int((ex == k).sum()) for k in (1, 2, 3)}
        res['escalated'] = float(((ex == 3) & (conf < TAU_ESC)).mean())
        res['per_class'] = per_class(D['c_test'], (prob[:, 1] >= .5).astype(int))
        out.update(route_prob=prob[:, 1].astype(np.float32), route_exit=ex.astype(np.int8),
                   route_conf=conf.astype(np.float32))
        # The  validation-set threshold sweep (justifies the chosen thresholds)
        pv = [F.softmax(z / t, 1).numpy() for z, t in zip(zv, T)]
        sweep = []
        for t1 in (0.80, 0.85, 0.90, 0.95, 0.99):
            for t2 in (0.80, 0.85, 0.90, 0.95, 0.99):
                c1, c2 = pv[0].max(1), pv[1].max(1)
                e1 = c1 >= t1; e2 = (~e1) & (c2 >= t2); e3 = ~(e1 | e2)
                pred = np.where(e1, pv[0][:, 1], np.where(e2, pv[1][:, 1], pv[2][:, 1])) >= .5
                sweep.append(dict(t1=t1, t2=t2, acc=float((pred == yv).mean()),
                                  s1=float(e1.mean()), s2=float(e2.mean()), s3=float(e3.mean())))
        res['val_sweep'] = sweep
        res['val_full_acc'] = float((pv[2].argmax(1) == yv).mean())
    return res, out



if __name__ == '__main__':
    name, kind, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    cfg = dict(CONFIG[name])
    if len(sys.argv) > 4:                 # batch-size override (UNSW-NB15 ablation models: 128 vs 512)
        cfg['bs'] = int(sys.argv[4])
    model, hist, sec, D = train(name, kind, seed, cfg['max_epochs'], cfg.get('patience', PATIENCE), cfg['bs'], cfg['lr'])
    res, out = evaluate(model, D, kind)
    res.update(dataset=name, kind=kind, seed=seed, history=hist, train_sec=sec, setup=dict(dict(patience=PATIENCE), **cfg),
               params=sum(p.numel() for p in model.parameters()))
    tag = f'{name}_{kind}_s{seed}' if len(sys.argv) <= 4 else f'{name}_{kind}_bs{cfg["bs"]}_s{seed}'
    dump(res, RES + tag + '.json')
    np.savez_compressed(RES + tag + '_pred.npz', **out)
    torch.save(model.state_dict(), RES + tag + '.pt')



    print('DONE', tag, res.get('routed', res['exits'][len(res['exits'])]), flush=True)
