"""NOW: builds each results table (LaTeX), figure (PNG) and in-text number of the
manuscript  from the saved run files, so that the text, tables and figures all come
from the same runs.     usage: python report.py......"""




import json, os, glob, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc


RES, DATA, FIG = './results/', './data/', './figs/'
OUT = './generated/'
os.makedirs(FIG, exist_ok=True); os.makedirs(OUT, exist_ok=True)
DS = ['cse', 'unsw', 'ciciot']
PRETTY = dict(cse='CSE-CIC-IDS2018', unsw='UNSW-NB15', ciciot='CICIoT2023')
C3 = ['#2a78d6', '#eb6834', '#1baf7a']           # validated categorical slots 1-3
C4 = C3 + ['#eda100']
INK, INK2 = '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'lines.linewidth': 2, 'savefig.dpi': 220, 'savefig.bbox': 'tight'})
J = lambda p: json.load(open(p)) if os.path.exists(p) else None
V = {}                                                # in-text values
TEX = lambda t: str(t).replace('\\', '/').replace('_', '\\_').replace('%', '\\%').replace('&', '\\&').replace('#', '\\#')





def runs(name, kind='eenet', seeds=(1, 2, 3)):
    return [r for r in (J(RES + f'{name}_{kind}_s{s}.json') for s in seeds) if r]



def ms(vals, scale=1, nd=2):
    a = np.array(vals, float) * scale
    if len(a) == 0:
        return '--'
    if len(a) == 1:
        return f'{a[0]:.{nd}f}'
    return f'{a.mean():.{nd}f} $\\pm$ {a.std(ddof=1):.{nd}f}'




def mean(vals, scale=1):
    return float(np.mean(np.array(vals, float) * scale)) if len(vals) else float('nan')


def write(name, body):
    open(OUT + name + '.tex', 'w').write(body)


# -----############------------------------- Table 2: The datasets....
rows = []
for n in DS:
    i = J(DATA + f'{n}_info.json')
    rows.append(f"{PRETTY[n]} & {i['raw_rows']:,} & {i['rows_after_dedup']:,} & {i['raw_cols']} / {i['n_features']} & "
                f"{i['n_train']:,} / {i['n_val']:,} / {i['n_test']:,} & {100*i['attack_share']:.1f} \\\\")
    V[f'DUP_{n.upper()}'] = f"{100*(i['raw_rows']-i['rows_after_dedup'])/i['raw_rows']:.0f}"
    V[f'NF_{n.upper()}'] = str(i['n_features'])
    V[f'MAL_{n.upper()}'] = f"{100*i['attack_share']:.1f}"
write('tab_datasets', '\n'.join(rows))



# --------- Table: the main centralized results__________+++++
rows = []; cm_rows = []
for n in DS:
    R = runs(n)
    if not R:
        continue
    g = lambda k: [r['routed'][k] for r in R]
    rows.append(f"{PRETTY[n]} & {ms(g('acc'),100)} & {ms(g('prec'),1,4)} & {ms(g('rec'),1,4)} & {ms(g('f1'),1,4)} & "
                f"{ms(g('auc'),1,4)} & {ms(g('fpr'),100)} & {ms(g('ece'),1,4)} \\\\")
    r1 = R[0]['routed']
    cm_rows.append(f"{PRETTY[n]} & {r1['tn']:,} & {r1['fp']:,} & {r1['fn']:,} & {r1['tp']:,} & "
                   f"{100*r1['fp']/max(r1['fp']+r1['tn'],1):.2f} & {100*r1['fn']/max(r1['fn']+r1['tp'],1):.2f} \\\\")
    key = dict(cse='CSE', unsw='UNSW', ciciot='CIC')[n]
    V[f'ACC_{key}'] = f"{mean(g('acc'),100):.2f}"; V[f'F1_{key}'] = f"{mean(g('f1')):.4f}"
    V[f'AUC_{key}'] = f"{mean(g('auc')):.4f}"; V[f'FPR_{key}'] = f"{mean(g('fpr'),100):.2f}"
    V[f'ECE_{key}'] = f"{mean(g('ece')):.4f}"; V[f'NRUN_{key}'] = str(len(R))
write('tab_main', '\n'.join(rows)); write('tab_cm', '\n'.join(cm_rows))
_u = runs('unsw')
if _u and 'setup' in _u[0]:
    _bs, _lr = _u[0]['setup']['bs'], _u[0]['setup']['lr']
    _m = round(_lr / 10 ** np.floor(np.log10(_lr)), 3); _e = int(np.floor(np.log10(_lr)))
    _lt = (f'10^{{{_e}}}' if _m == 1 else f'{_m:g}\\times10^{{{_e}}}')
    V['UNSW_CFG'] = f'batch size {_bs}, $\\eta={_lt}$'
aucs = [mean([r['routed']['auc'] for r in runs(n)]) for n in DS if runs(n)]
if aucs:
    V['AUC_MIN'] = f'{np.floor(min(aucs) * 1000) / 1000:.3f}'      # 'at least'
eces = [mean([r['routed']['ece'] for r in runs(n)]) for n in DS if runs(n)]
if eces:
    V['ECE_MAX'] = f'{np.ceil(max(eces) * 1000) / 1000:.3f}'       # 'at or below'




# -######_______+++----- Table: routing per dataset (+ latency)..........
rows = []; s1s = []; latf = []; latr = []
for n in DS:
    R = runs(n); L = J(RES + f'{n}_latency.json')
    if not R:
        continue
    for k in ('1', '2', '3'):
        sh = [r['share'][k] for r in R]; ea = [r['exit_acc'][k] for r in R if r['exit_acc'][k] is not None]
        lat = f"{L['path'][k]['ms_mean']:.3f}" if L else '--'
        lead = f"\\multirow{{4}}{{*}}{{{PRETTY[n]}}}" if k == '1' else ''
        rows.append(f"{lead} & Exit {k} & {ms(sh,100,1)} & {ms(ea,100,2)} & {lat} \\\\")
    acc = [r['routed']['acc'] for r in R]
    latline = f"{L['routed_single']['ms_mean']:.3f}" if L else '--'
    rows.append(f" & Routed (all flows) & 100 & {ms(acc,100,2)} & {latline} \\\\ \\midrule")
    s1s.append(mean([r['share']['1'] for r in R], 100))
    V[f'S1_{n.upper()}'] = f"{mean([r['share']['1'] for r in R],100):.1f}"
    V[f'S2_{n.upper()}'] = f"{mean([r['share']['2'] for r in R],100):.1f}"
    V[f'S3_{n.upper()}'] = f"{mean([r['share']['3'] for r in R],100):.1f}"
    V[f'N3_{n.upper()}'] = f"{mean([r['exit_n']['3'] for r in R]):.0f}"
    V[f'EA1_{n.upper()}'] = f"{mean([r['exit_acc']['1'] for r in R],100):.2f}"
    esc = []
    for sd in (1, 2, 3):
        pp = RES + f'{n}_eenet_s{sd}_pred.npz'
        if os.path.exists(pp):
            z = np.load(pp); esc.append(((z['route_exit'] == 3) & (z['route_conf'] < 0.85) & (z['route_prob'] >= 0.5)).mean())
    V[f'ESC_{n.upper()}'] = f"{mean(esc,100):.2f}" 
    if L:
        latf.append(L['path']['3']['ms_mean']); latr.append(L['routed_single']['ms_mean'])
        V[f'LAT1_{n.upper()}'] = f"{L['path']['1']['ms_mean']:.3f}"; V[f'LAT3_{n.upper()}'] = f"{L['path']['3']['ms_mean']:.3f}"
        V[f'LATR_{n.upper()}'] = f"{L['routed_single']['ms_mean']:.3f}"
        V[f'THR_{n.upper()}'] = f"{L['throughput_fps']:,.0f}"; V[f'PRE_{n.upper()}'] = f"{1000*L['preprocess']['ms_mean']:.1f}"
if rows:
    rows[-1] = rows[-1].replace(' \\midrule', '')
write('tab_routing', '\n'.join(rows))
if s1s:
    V['S1_MIN'], V['S1_MAX'] = f'{min(s1s):.1f}', f'{max(s1s):.1f}'
if latf:
    V['LATF_MIN'], V['LATF_MAX'] = f'{min(latf):.2f}', f'{max(latf):.2f}'
    V['LATR_MIN'], V['LATR_MAX'] = f'{min(latr):.2f}', f'{max(latr):.2f}'
    lat1 = [float(V[f'LAT1_{n.upper()}']) for n in DS if f'LAT1_{n.upper()}' in V]
    V['LAT1_MIN'], V['LAT1_MAX'] = f'{min(lat1):.2f}', f'{max(lat1):.2f}'




# The validation sweep summary   ( threshold justification) .........................
SWRAW = {}
for n in DS:
    R = runs(n)
    if not R:
        continue
    sw = R[0]['val_sweep']; full = R[0]['val_full_acc']
    pick = [s for s in sw if s['t2'] == 0.90]
    V[f'SWEEP_{n.upper()}'] = '; '.join(f"$\\tau_1={s['t1']:.2f}$: {100*s['acc']:.2f}\\% / {100*s['s1']:.1f}\\%" for s in pick)
    sel = [s for s in sw if s['t1'] == 0.95 and s['t2'] == 0.90][0]
    V[f'SWGAP_{n.upper()}'] = f"{100*(full-sel['acc']):.2f}"
    loose = [s for s in sw if s['t1'] == 0.80 and s['t2'] == 0.80][0]
    V[f'SWLOOSE_{n.upper()}'] = f"{100*(full-loose['acc']):.2f}"
    V[f'SWMAX_{n.upper()}'] = f"{max(0.0, 100*max(full-x['acc'] for x in sw)):.2f}"
    SWRAW[n] = max(0.0, 100*max(full-x['acc'] for x in sw))

if 'cse' in SWRAW and 'ciciot' in SWRAW:
    V['SWMAX_LARGE'] = f"{max(0.01, np.ceil(100*max(SWRAW['cse'], SWRAW['ciciot']))/100):.2f}"





# --##__+++----- Table: each exit on all of the  flows.............
rows = []
for n in DS:
    R = runs(n)
    if not R:
        continue
    for k, nm in (('1', 'Exit 1 (CNN)'), ('2', 'Exit 2 (BiLSTM)'), ('3', 'Exit 3 (Transformer)')):
        g = lambda m: [r['exits'][k][m] for r in R]
        lead = f"\\multirow{{3}}{{*}}{{{PRETTY[n]}}}" if k == '1' else ''
        rows.append(f"{lead} & {nm} & {ms(g('acc'),100)} & {ms(g('prec'),1,4)} & {ms(g('rec'),1,4)} & "
                    f"{ms(g('f1'),1,4)} & {ms(g('auc'),1,4)} \\\\" + (' \\midrule' if k == '3' else ''))
        V[f'EXACC{k}_{n.upper()}'] = f"{mean(g('acc'),100):.2f}"
if rows:
    rows[-1] = rows[-1].replace(' \\midrule', '')
write('tab_exits', '\n'.join(rows))






# ----- Table: the ablation study  &  the baseline related techniqus.....
rows = []
for n in DS:
    B = J(RES + f'{n}_baselines.json') or {}
    if B:
        V[f'CLS_BEST_{n.upper()}'] = f"{100*max(B[k]['acc'] for k in ('lr', 'rf', 'hgb') if k in B):.2f}"
    items = [('Logistic regression', B.get('lr')), ('Random forest', B.get('rf')), ('Gradient boosting', B.get('hgb'))]
    for kind, nm in (('cnn', 'CNN only'), ('bilstm', 'BiLSTM only'), ('transformer', 'Transformer only'),
                     ('noexit', 'Backbone without early exits')):
        r = J(RES + f'{n}_{kind}_s1.json')
        items.append((nm, r['exits']['1'] if r else None))
    R = runs(n)
    items.append(('EENet-IPS (routed, early exits)', {m: mean([r['routed'][m] for r in R]) for m in ('acc', 'f1', 'auc', 'fpr')} if R else None))
    first = True
    for nm, m in items:
        lead = f"\\multirow{{{len(items)}}}{{*}}{{{PRETTY[n]}}}" if first else ''
        first = False
        cells = (f"{100*m['acc']:.2f} & {m['f1']:.4f} & {m['auc']:.4f} & {100*m['fpr']:.2f}" if m else '-- & -- & -- & --')
        rows.append(f"{lead} & {nm} & {cells} \\\\")
    rows[-1] += ' \\midrule'
    for kind in ('noexit',):
        r = J(RES + f'{n}_{kind}_s1.json')
        if r:
            V[f'NOEXIT_ACC_{n.upper()}'] = f"{100*r['exits']['1']['acc']:.2f}"
if rows:
    rows[-1] = rows[-1].replace(' \\midrule', '')
write('tab_ablation', '\n'.join(rows))






# -----##----- classical models vs the EENet-IPS (See Subsection   4.3.2)............
BL = {n: J(RES + f'{n}_baselines.json') for n in DS}; RR = {n: runs(n) for n in DS}
FLR = {n: J(RES + f'{n}_fl_eenetfl_s1.json') for n in DS}
if all(BL.values()) and all(RR.values()) and all('runs' in BL[n]['hgb'] for n in DS):
    r2 = lambda x: round(100 * x, 2)                    # gap between the values printed in the table
    gaps = [r2(BL[n]['hgb']['acc']) - r2(mean([r['routed']['acc'] for r in RR[n]])) for n in DS]
    V['GB_GAP_MIN'], V['GB_GAP_MAX'] = f'{min(gaps):.2f}', f'{max(gaps):.2f}'
    brun = lambda n, k, m: [r[m] for r in BL[n][k]['runs']]
    V['FPR_EE_CICIOT'] = ms([r['routed']['fpr'] for r in RR['ciciot']], 100)
    V['FPR_GB_CICIOT'] = ms(brun('ciciot', 'hgb', 'fpr'), 100)
    V['FPR_RF_CICIOT'] = ms(brun('ciciot', 'rf', 'fpr'), 100)
    V['TPR_EE_CICIOT'] = f"{mean([r['routed']['rec'] for r in RR['ciciot']], 100):.2f}"
    V['TPR_GB_CICIOT'] = f"{100 * BL['ciciot']['hgb']['rec']:.2f}"
    tr = sorted(100 * BL['unsw'][k]['fpr'] for k in ('rf', 'hgb'))
    V['FPR_TREES_UNSW'] = f'{tr[0]:.2f}--{tr[1]:.2f}'
    V['FPR_EE_UNSW'] = f"{mean([r['routed']['fpr'] for r in RR['unsw']], 100):.2f}"
    c = RR['ciciot'][0]['routed']; V['CIC_BEN_SHARE'] = f"{100 * (c['tn'] + c['fp']) / (c['tn'] + c['fp'] + c['fn'] + c['tp']):.1f}"
    for n in ('unsw', 'ciciot'):                       # benign flows blocked by the graded policy (Section 3.4)
        y = np.load(DATA + f'{n}.npz', allow_pickle=True)['y_test']; blk = []
        for s in (1, 2, 3):
            z = np.load(RES + f'{n}_eenet_s{s}_pred.npz'); att = z['route_prob'] >= 0.5
            b = att & ((z['route_exit'] == 1) | ((z['route_exit'] == 3) & (z['route_conf'] >= 0.85)))
            blk.append(b[y == 0].mean())
        V[f'BLK_BEN_{n.upper()}'] = f'{100 * np.mean(blk):.2f}'
    # deep alternatives (Table tab:ablation): the routed system matches or exceeds all of them on UNSW-NB15 and CICIoT2023
    for n in ('unsw', 'ciciot'):
        deep = [J(RES + f'{n}_{k}_s1.json')['exits']['1'] for k in ('cnn', 'bilstm', 'transformer', 'noexit')]
        ee = {m: mean([r['routed'][m] for r in RR[n]]) for m in ('acc', 'fpr')}
        assert r2(ee['acc']) >= max(r2(d['acc']) for d in deep) and r2(ee['fpr']) <= min(r2(d['fpr']) for d in deep), n
    # histogram-based federated gradient boosting that rebuilds the seed-1 models vs EENet-FL (tree_fl_cost.py)
    hc = [BL[n]['hgb_fl_cost'] for n in DS]
    V['HFL_EXCH_MIN'], V['HFL_EXCH_MAX'] = f"{min(h['exchanges_min'] for h in hc):,}", f"{max(h['exchanges_min'] for h in hc):,}"
    V['HFL_GB_MIN'], V['HFL_GB_MAX'] = f"{min(h['uplink_MB'] for h in hc) / 1e3:.2f}", f"{max(h['uplink_MB'] for h in hc) / 1e3:.2f}"
    if all(FLR.values()):
        up = [FLR[n]['upload_MB_total'] for n in DS]
        V['EEFL_MB_MIN'], V['EEFL_MB_MAX'] = f'{min(up):.1f}', f'{max(up):.1f}'






# #########   Table: XAI cost & quality................

rows = []; XR = []
for n in DS:
    X = J(RES + f'{n}_xai.json'); L = J(RES + f'{n}_latency.json')
    if not X:
        continue
    inf = lambda k: f"{L['path'][k]['ms_mean']:.3f}" if L else '--'
    cf = X['cf']
    rows.append(f"\\multirow{{3}}{{*}}{{{PRETTY[n]}}} & Exit 1: KernelSHAP & {inf('1')} & {X['shap']['ms_mean']:.1f} $\\pm$ {X['shap']['ms_std']:.1f} & local-accuracy gap {X['shap']['local_acc_gap']:.1e} \\\\")
    rows.append(f" & Exit 2: Integrated Gradients & {inf('2')} & {X['ig']['ms_mean']:.1f} $\\pm$ {X['ig']['ms_std']:.1f} & completeness error {100*X['ig']['completeness_rel_err']:.2f}\\% \\\\")
    rows.append(f" & Exit 3: counterfactual & {inf('3')} & {cf['ms_mean']:.1f} $\\pm$ {cf['ms_std']:.1f} & validity {100*cf['validity']:.1f}\\%, {cf['sparsity_mean']:.1f} features, 0 violations \\\\ \\midrule"
                if cf['violations'] == 0 else
                f" & Exit 3: counterfactual & {inf('3')} & {cf['ms_mean']:.1f} $\\pm$ {cf['ms_std']:.1f} & validity {100*cf['validity']:.1f}\\%, {cf['sparsity_mean']:.1f} features, {cf['violations']} violations \\\\ \\midrule")
    V[f'SHAPMS_{n.upper()}'] = f"{X['shap']['ms_mean']:.1f}"; V[f'IGMS_{n.upper()}'] = f"{X['ig']['ms_mean']:.1f}"
    V[f'CFMS_{n.upper()}'] = f"{cf['ms_mean']:.1f}"; V[f'CFVAL_{n.upper()}'] = f"{100*cf['validity']:.1f}"
    V[f'CFSP_{n.upper()}'] = f"{cf['sparsity_mean']:.1f}"; V[f'IGERR_{n.upper()}'] = f"{100*X['ig']['completeness_rel_err']:.2f}"
    V[f'SHAPTOP_{n.upper()}'] = ', '.join(TEX(t) for t in X['shap']['top'][:3])
    if L:                                  # explanation time / inference time of the same exit
        XR.extend([X['shap']['ms_mean'] / L['path']['1']['ms_mean'], X['ig']['ms_mean'] / L['path']['2']['ms_mean'],
                   cf['ms_mean'] / L['path']['3']['ms_mean']])
if rows:
    rows[-1] = rows[-1].replace(' \\midrule', '')
write('tab_xai', '\n'.join(rows))
if XR:
    V['XAIX_MIN'], V['XAIX_MAX'] = f'{np.floor(min(XR)):.0f}', f'{np.ceil(max(XR)):.0f}'
igerr = [100 * J(RES + f'{n}_xai.json')['ig']['completeness_rel_err'] for n in DS if J(RES + f'{n}_xai.json')]
if igerr:
    V['IGERR_MAX'] = f'{np.ceil(max(igerr) * 100) / 100:.2f}'      # 'at most'
shapms = [float(V[f'SHAPMS_{n.upper()}']) for n in DS if f'SHAPMS_{n.upper()}' in V]
if shapms:
    V['SHAPMS_MIN'], V['SHAPMS_MAX'] = f'{min(shapms):.0f}', f'{max(shapms):.0f}'



# -----------======------------ Table: per-attack-family detection ====
FAM = lambda c: ('Benign (false alarms)' if c.upper() in ('BENIGN', 'NORMAL') else
                 'DDoS' if c.upper().startswith('DDOS') else 'DoS' if c.upper().startswith('DOS') else
                 'Mirai' if c.upper().startswith('MIRAI') else
                 'Reconnaissance' if c.upper().startswith('RECON') or c.upper() == 'VULNERABILITYSCAN' else
                 'Spoofing' if 'SPOOF' in c.upper() else 'Brute force' if 'BRUTE' in c.upper() else
                 'Web-based' if c.upper() in ('BROWSERHIJACKING', 'SQLINJECTION', 'COMMANDINJECTION', 'XSS',
                                              'BACKDOOR_MALWARE', 'UPLOADING_ATTACK') else c)
rows = []; FAMRATE = {}
for n in DS:
    R = runs(n)
    if not R:
        continue
    agg = {}
    for r in R:
        for c, v in r['per_class'].items():
            f = FAM(c) if n == 'ciciot' else ('Benign (false alarms)' if c in ('Benign', 'Normal') else c)
            a = agg.setdefault(f, [[], 0])
            a[0].append((v['flagged'], v['n'])); a[1] = v['n']
    cells = []
    for f in sorted(agg, key=lambda x: (x.startswith('Benign'), x)):
        per_run = {}
        for r in R:
            num = den = 0
            for c, v in r['per_class'].items():
                ff = FAM(c) if n == 'ciciot' else ('Benign (false alarms)' if c in ('Benign', 'Normal') else c)
                if ff == f:
                    num += v['flagged'] * v['n']; den += v['n']
            per_run.setdefault(f, []).append(num / den)
        nn = sum(v['n'] for c, v in R[0]['per_class'].items()
                 if (FAM(c) if n == 'ciciot' else ('Benign (false alarms)' if c in ('Benign', 'Normal') else c)) == f)
        cells.append((f, nn, mean(per_run[f], 100)))
    FAMRATE[n] = {f: (nn, rate) for f, nn, rate in cells}
    for i, (f, nn, rate) in enumerate(cells):
        lead = f"\\multirow{{{len(cells)}}}{{*}}{{{PRETTY[n]}}}" if i == 0 else ''
        rows.append(f"{lead} & {TEX(f)} & {nn:,} & {rate:.2f} \\\\" + (' \\midrule' if i == len(cells) - 1 else ''))
if rows:
    rows[-1] = rows[-1].replace(' \\midrule', '')
write('tab_perclass', '\n'.join(rows))
if 'cse' in FAMRATE and 'FTP-BruteForce' in FAMRATE['cse']:
    V['FTP_N'] = str(FAMRATE['cse']['FTP-BruteForce'][0]); V['FTP_RATE'] = f"{FAMRATE['cse']['FTP-BruteForce'][1]:.1f}"
    _n, _r = FAMRATE['cse']['FTP-BruteForce']
    V['FTP_TEXT'] = (f'none of the {_n} in the test partition is flagged as malicious,' if _r == 0 else
                     f'all {_n} in the test partition are flagged as malicious,' if _r >= 99.95 else
                     f'only {_r:.1f}\\% of the {_n} in the test partition are flagged as malicious,')
if 'ciciot' in FAMRATE:
    weak = [FAMRATE['ciciot'][f][1] for f in ('Brute force', 'Reconnaissance', 'Spoofing', 'Web-based') if f in FAMRATE['ciciot']]
    flood = [FAMRATE['ciciot'][f][1] for f in ('DDoS', 'DoS', 'Mirai') if f in FAMRATE['ciciot']]
    if weak:
        V['CIC_WEAK_MIN'], V['CIC_WEAK_MAX'] = f'{min(weak):.1f}', f'{max(weak):.1f}'
    if flood:
        V['CIC_FLOOD_MIN'] = f'{np.floor(min(flood)*100)/100:.2f}'




#@@@@@@-------- Table: the federated .............

rows = []; gaps = []; upx = []
for n in DS:
    R = runs(n)
    F = {c: J(RES + f'{n}_fl_{c}_s1.json') for c in ('fedavg', 'fedprox', 'eenetfl')}
    items = [('Centralized (upper bound)', {m: mean([r['routed'][m] for r in R]) for m in ('acc', 'f1', 'auc', 'fpr')} if R else None,
              mean([r['share']['1'] for r in R]) if R else None, None)]
    for c, nm in (('fedavg', 'FedAvg (dense)'), ('fedprox', 'FedProx (dense)'), ('eenetfl', 'EENet-FL (clip + noise + top-$k$)')):
        f = F[c]
        items.append((nm, f['routed'] if f else None, f['share']['1'] if f else None, f['upload_MB_total'] if f else None))
    for i, (nm, m, s1, up) in enumerate(items):
        lead = f"\\multirow{{4}}{{*}}{{{PRETTY[n]}}}" if i == 0 else ''
        cells = (f"{100*m['acc']:.2f} & {m['f1']:.4f} & {m['auc']:.4f} & {100*m['fpr']:.2f} & {100*s1:.1f}" if m else '-- & -- & -- & -- & --')
        upc = f"{up:.1f}" if up is not None else '--'
        rows.append(f"{lead} & {nm} & {cells} & {upc} \\\\" + (' \\midrule' if i == 3 else ''))
    if R and F['eenetfl']:
        gap = mean([r['routed']['acc'] for r in R], 100) - 100 * F['eenetfl']['routed']['acc']
        gaps.append(gap); V[f'FLGAP_{n.upper()}'] = f'{gap:.2f}'
        V[f'FLACC_{n.upper()}'] = f"{100*F['eenetfl']['routed']['acc']:.2f}"
    if F['fedavg'] and F['eenetfl']:
        upx.append(F['fedavg']['upload_MB_total'] / F['eenetfl']['upload_MB_total'])
        V[f'FAACC_{n.upper()}'] = f"{100*F['fedavg']['routed']['acc']:.2f}"
    if F['fedprox']:
        V[f'FPACC_{n.upper()}'] = f"{100*F['fedprox']['routed']['acc']:.2f}"
if rows:
    rows[-1] = rows[-1].replace(' \\midrule', '')
write('tab_federated', '\n'.join(rows))
if gaps:
    V['FLGAP_MAX'] = f'{max(gaps):.2f}'
if upx:
    V['UPX'] = f'{np.mean(upx):.1f}'





# ------- Table: flow volume  =___--

Vv = J(RES + 'cse_volume.json')
if Vv:
    rows = []
    for N in sorted({r['N'] for r in Vv['runs']}):
        rr = [r for r in Vv['runs'] if r['N'] == N]
        rows.append(f"{N:,} & {ms([r['fps'] for r in rr],1,0)} & {ms([r['ms_per_flow'] for r in rr],1,4)} & "
                    f"{ms([r['peak_MB'] for r in rr],1,0)} & {ms([r['acc'] for r in rr],100,2)} & {ms([r['share1'] for r in rr],100,1)} \\\\")
    write('tab_volume', '\n'.join(rows))
    fps = [r['fps'] for r in Vv['runs']]
    V['VOL_FPS_MIN'], V['VOL_FPS_MAX'] = f'{min(fps):,.0f}', f'{max(fps):,.0f}'

json.dump(V, open(OUT + 'values.json', 'w'), indent=1)






# =============----================ THE  FIGURES,...
def save(fig, name):
    fig.savefig(FIG + name); plt.close(fig)

# Fig 2(a) validation accuracy (final exit) per epoch, (b) ROC of the routed system (seed 1)
fig, ax = plt.subplots(figsize=(5, 4))
for c, n in zip(C3, DS):
    R = runs(n, seeds=(1,))
    if R:
        h = R[0]['history']; ax.plot([e['epoch'] for e in h], [e['val_acc'][2] for e in h], color=c, marker='o', ms=4, label=PRETTY[n])
ax.set_xlabel('Epoch'); ax.set_ylabel('Validation accuracy (Exit 3)'); ax.legend(loc='lower right')
save(fig, 'fig2a_valacc.png')
fig, ax = plt.subplots(figsize=(5, 4))
for c, n in zip(C3, DS):
    p = RES + f'{n}_eenet_s1_pred.npz'
    if os.path.exists(p):
        z = np.load(p); y = np.load(DATA + f'{n}.npz', allow_pickle=True)['y_test']
        fpr, tpr, _ = roc_curve(y, z['route_prob']); ax.plot(fpr, tpr, color=c, label=f'{PRETTY[n]} (AUC = {sk_auc(fpr, tpr):.4f})')
ax.plot([0, 1], [0, 1], ls=':', color=INK2, lw=1, label='Random guess')
ax.set_xscale('symlog', linthresh=0.01); ax.set_xlim(0, 1)
ax.set_xlabel('False positive rate (symlog scale)'); ax.set_ylabel('True positive rate'); ax.legend(loc='lower right', fontsize=9)
save(fig, 'fig2b_roc.png')

# Fig 3: training / validation loss per dataset (seed 1)
for n in DS:
    R = runs(n, seeds=(1,))
    if not R:
        continue
    h = R[0]['history']; e = [x['epoch'] for x in h]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot(e, [x['train_loss'] for x in h], color=C3[0], marker='o', ms=4, label='Training loss')
    ax.plot(e, [x['val_loss'] for x in h], color=C3[1], marker='s', ms=4, label='Validation loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Mean loss of the three exits'); ax.set_title(PRETTY[n], fontsize=11); ax.legend()
    save(fig, f'fig3_loss_{n}.png')



#   Fig 4(a): calibrated confidence at each exit for the flows that reach it (seed 1)
fig, ax = plt.subplots(figsize=(5.4, 4))
pos = 0; ticks = []
for j, n in enumerate(DS):
    p = RES + f'{n}_eenet_s1_pred.npz'
    if not os.path.exists(p):
        continue
    z = np.load(p); P = z['probs']; ex = z['route_exit']
    conf = [P[0].max(1), P[1].max(1)[ex >= 2], P[2].max(1)[ex == 3]]
    bp = ax.boxplot(conf, positions=[j * 4 + 1, j * 4 + 2, j * 4 + 3], widths=0.7, patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set(facecolor=C3[j] + '55', edgecolor=C3[j])
    for el in ('whiskers', 'caps', 'medians'):
        for l in bp[el]:
            l.set(color=C3[j])
    ticks += [(j * 4 + 1, 'E1'), (j * 4 + 2, 'E2'), (j * 4 + 3, 'E3')]
    ax.text(j * 4 + 2, 0.405, PRETTY[n], ha='center', fontsize=9, color=INK)
ax.axhline(0.95, color=INK2, ls='--', lw=1); ax.text(11.6, 0.953, r'$\tau_1$', fontsize=9, color=INK2)
ax.axhline(0.90, color=INK2, ls=':', lw=1); ax.text(11.6, 0.903, r'$\tau_2$', fontsize=9, color=INK2)
ax.set_xticks([t for t, _ in ticks]); ax.set_xticklabels([l for _, l in ticks]); ax.set_ylim(0.4, 1.02)
ax.set_ylabel('Calibrated confidence $c_k$'); ax.set_xlabel('Exit (flows that reach it)')
save(fig, 'fig4a_conf.png')


#   Fig 4(b): flow-volume test (throughput) for CSE-CIC-IDS2018
if Vv:
    Ns = sorted({r['N'] for r in Vv['runs']})
    m = [np.mean([r['fps'] for r in Vv['runs'] if r['N'] == N]) for N in Ns]
    s = [np.std([r['fps'] for r in Vv['runs'] if r['N'] == N], ddof=1) for N in Ns]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(Ns, m, yerr=s, color=C3[0], marker='o', ms=6, capsize=3)
    ax.set_xscale('log'); ax.set_xlabel('Number of replayed flows'); ax.set_ylabel('Throughput (flows/s, one CPU thread)')
    ax.set_ylim(0, max(m) * 1.25)
    save(fig, 'fig4b_volume.png')

#    Fig 5(a): per-exit validation accuracy per epoch (UNSW-NB15, seed 1); (b) per-exit test accuracy per dataset
R = runs('unsw', seeds=(1,))
if R:
    h = R[0]['history']; e = [x['epoch'] for x in h]
    fig, ax = plt.subplots(figsize=(5, 4))
    for k, (c, nm) in enumerate(zip(C3, ('Exit 1 (CNN)', 'Exit 2 (BiLSTM)', 'Exit 3 (Transformer)'))):
        ax.plot(e, [x['val_acc'][k] for x in h], color=c, marker='os^'[k], ms=4, label=nm)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Validation accuracy (UNSW-NB15)'); ax.legend(loc='lower right')
    save(fig, 'fig5a_exitacc.png')
fig, ax = plt.subplots(figsize=(5.4, 4))
w = 0.26
for k, (c, nm) in enumerate(zip(C3, ('Exit 1 (CNN)', 'Exit 2 (BiLSTM)', 'Exit 3 (Transformer)'))):
    vals, errs = [], []
    for n in DS:
        Rn = runs(n); a = [100 * r['exits'][str(k + 1)]['acc'] for r in Rn]
        vals.append(np.mean(a) if a else 0); errs.append(np.std(a, ddof=1) if len(a) > 1 else 0)
    x = np.arange(3) + (k - 1) * w
    ax.bar(x, vals, w * 0.92, yerr=errs, color=c, label=nm, capsize=2)
lo = min([100 * r['exits'][k]['acc'] for n in DS for r in runs(n) for k in ('1', '2', '3')] or [90])
ax.set_ylim(max(0, np.floor(lo) - 2), 100.5); ax.set_xticks(range(3)); ax.set_xticklabels([PRETTY[n] for n in DS], fontsize=9)
ax.set_ylabel('Test accuracy on all flows (%)'); ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8.5)
save(fig, 'fig5b_exitbars.png')



# Fig 8(a): federated convergence (validation accuracy of Exit 3 per round); (b) uplink traffic per configuration
fig, axs = plt.subplots(1, 3, figsize=(12, 3.9), sharey=False, constrained_layout=True)
for ax, n in zip(axs, DS):
    for c, (cfg, nm) in zip(C3, (('fedavg', 'FedAvg'), ('fedprox', 'FedProx'), ('eenetfl', 'EENet-FL'))):
        f = J(RES + f'{n}_fl_{cfg}_s1.json')
        if f:
            ax.plot([h['round'] for h in f['history']], [h['val_acc'][2] for h in f['history']], color=c, marker='o', ms=3, label=nm)
            b = f.get('best_round')
            if b:                                   # selected round (lowest validation loss)
                ax.plot(b, f['history'][b - 1]['val_acc'][2], marker='*', ms=12, color=c, mec='white', mew=0.8, ls='none')
    R = runs(n)
    if R:
        ax.axhline(mean([r['val_full_acc'] for r in R]), color=INK2, ls='--', lw=1, label='Centralized')
    ax.set_title(PRETTY[n], fontsize=11); ax.set_xlabel('Round')
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
axs[0].set_ylabel('Validation accuracy (Exit 3)')
hl, lb = axs[0].get_legend_handles_labels()
fig.legend(hl, lb, loc='outside upper center', ncol=4, fontsize=10)
save(fig, 'fig8a_flconv.png')
fig, ax = plt.subplots(figsize=(5.4, 4))
for k, (c, (cfg, nm)) in enumerate(zip(C3, (('fedavg', 'FedAvg'), ('fedprox', 'FedProx'), ('eenetfl', 'EENet-FL')))):
    vals = [(J(RES + f'{n}_fl_{cfg}_s1.json') or {}).get('upload_MB_total', 0) for n in DS]
    ax.bar(np.arange(3) + (k - 1) * w, vals, w * 0.92, color=c, label=nm)
ax.set_xticks(range(3)); ax.set_xticklabels([PRETTY[n] for n in DS], fontsize=9)
ax.set_ylabel('Total uplink traffic (MB)'); ax.legend(fontsize=9)
save(fig, 'fig8b_uplink.png')

print(json.dumps(V, indent=0)[:3000])




