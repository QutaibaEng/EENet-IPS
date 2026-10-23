"""The   Pick  the UNSW-NB15 batch size / the learning rate with the best validation routed accuracy
(ties: lower validation loss)  and   write it into train.py CONFIG. .........."""

import json, glob, re

R = [json.load(open(f)) for f in glob.glob('./results/sweep/unsw_*_s1.json')]
best = max(R, key=lambda r: (round(r['val_routed_acc'], 6), -r['best_val_loss']))
print('selected', best['bs'], best['lr'], best['val_routed_acc'])
s = open('./train.py').read()
new = f"unsw=dict(bs={best['bs']}, lr={best['lr']:g}, max_epochs=60, patience=8))"
s2 = re.sub(r"unsw=dict\(bs=\d+, lr=[0-9.e-]+, max_epochs=60, patience=8\)\)", new, s)
assert s2 != s or new in s
open('./train.py', 'w').write(s2)

