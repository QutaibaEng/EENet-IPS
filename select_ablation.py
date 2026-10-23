"""The UNSW-NB15 ablation models: keep the batch size   (128, as selected for EENet-IPS,  or the
default 512) with    the higher validation accuracy. The test split is not really  leveraged ....."""




import json, shutil, torch, numpy as np
from common import *
from train import build


D = load('unsw'); d = D['X_train'].shape[1]; yv = D['y_val']
for kind in ('noexit', 'cnn', 'bilstm', 'transformer'):
    acc = {}
    for bs, tag in ((128, f'unsw_{kind}_s1'), (512, f'unsw_{kind}_bs512_s1')):
        m = build(kind, d); m.load_state_dict(torch.load(RES + tag + '.pt')); m.eval()
        z = logits_all(m, D['X_val'])[-1]
        acc[bs] = float((z.argmax(1).numpy() == yv).mean())
    win = max(acc, key=acc.get)
    print(kind, 'val acc', acc, '-> batch', win, flush=True)
    if win == 512:                      # keep the batch-128 files under their own name, promote batch 512
        for ext in ('.json', '.pt', '_pred.npz'):
            shutil.move(RES + f'unsw_{kind}_s1' + ext, RES + f'unsw_{kind}_bs128_s1' + ext)
            shutil.copy(RES + f'unsw_{kind}_bs512_s1' + ext, RES + f'unsw_{kind}_s1' + ext)
    r = json.load(open(RES + f'unsw_{kind}_s1.json'))
    r['bs_selection'] = dict(val_acc={str(k): v for k, v in acc.items()}, selected=win)
    dump(r, RES + f'unsw_{kind}_s1.json')
