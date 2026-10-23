"""THE  Validation-only selection of the batch size and learning rate  (the test split is not utilized).....
usage:   python sweep.py <dataset> <batch_size> <learning_rate> <seed>
"""

import sys, json, numpy as np, torch, torch.nn.functional as F
from common import *
from train import train, CONFIG, PATIENCE



name, bs, lr, seed = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
model, hist, sec, D = train(name, 'eenet', seed, CONFIG[name]['max_epochs'], CONFIG[name].get('patience', PATIENCE), bs, lr)
zv = logits_all(model, D['X_val']); yv = D['y_val']
T = [fit_temperature(z, yv) for z in zv]
pv = [F.softmax(z / t, 1).numpy() for z, t in zip(zv, T)]
c1, c2 = pv[0].max(1), pv[1].max(1)
e1 = c1 >= TAU[0]; e2 = (~e1) & (c2 >= TAU[1])
pred = np.where(e1, pv[0][:, 1], np.where(e2, pv[1][:, 1], pv[2][:, 1])) >= .5
out = dict(dataset=name, bs=bs, lr=lr, seed=seed, epochs=len(hist), sec=sec,
           val_routed_acc=float((pred == yv).mean()), val_exit_acc=[float((p.argmax(1) == yv).mean()) for p in pv],
           val_s1=float(e1.mean()), best_val_loss=min(h['val_loss'] for h in hist))
os.makedirs('./results/sweep/', exist_ok=True)
json.dump(out, open(f'./results/sweep/{name}_bs{bs}_lr{lr:g}_s{seed}.json', 'w'), indent=1)


print(json.dumps(out), flush=True)
