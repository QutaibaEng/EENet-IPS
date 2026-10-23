"""The EENet-IPS multi-exit backbone and the single-model baselines (PyTorch).


One sequential backbone: The  CNN stage ->   BiLSTM stage ->   Transformer stage, with a
classifier (exit) after each stage.    A flow's d features are read as a length-d,
one-channel sequence; the CNN stage (two conv+pool blocks) reduces it to
L = ceil(d/4) tokens, which the BiLSTM and Transformer stages then process......


"""


import math, torch, torch.nn as nn, torch.nn.functional as F




def head(dim, n_cls=2):
    return nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, n_cls))





def n_tokens(d):
    return math.ceil(math.ceil(d / 2) / 2)




class Stage1CNN(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.c1 = nn.Conv1d(1, ch, 3, padding=1)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.pool = nn.MaxPool1d(2, ceil_mode=True)
        self.norm = nn.LayerNorm(ch)

    def forward(self, x):                      # x: [B, d]
        h = self.pool(F.relu(self.c1(x.unsqueeze(1))))   # [B, ch, d/2]
        h = self.pool(F.relu(self.c2(h)))                # [B, ch, d/4]
        return self.norm(h.transpose(1, 2))              # [B, L, ch]






class Stage2BiLSTM(nn.Module):
    def __init__(self, inp=32, hid=32):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, batch_first=True, bidirectional=True)

    def forward(self, h):                      # [B, L, inp]
        return self.lstm(h)[0]                 # [B, L, 2*hid]


class Stage3Transformer(nn.Module):
    def __init__(self, L, dm=64, heads=4, ff=128, layers=2, drop=0.1):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, L, dm))
        layer = nn.TransformerEncoderLayer(dm, heads, ff, drop, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)






    def forward(self, h):                      # [B, L, dm] ..........
        return self.enc(h + self.pos[:, :h.shape[1]])





class EENet(nn.Module):
    def __init__(self, d, n_cls=2):
        super().__init__()
        self.d, self.L = d, n_tokens(d)
        self.s1, self.s2, self.s3 = Stage1CNN(32), Stage2BiLSTM(32, 32), Stage3Transformer(self.L, 64)
        self.h1, self.h2, self.h3 = head(32, n_cls), head(64, n_cls), head(64, n_cls)

    @staticmethod
    def pool(h):
        return h.mean(1)



    def forward(self, x):                      # all exits (training)
        a1 = self.s1(x); z1 = self.h1(self.pool(a1))
        a2 = self.s2(a1); z2 = self.h2(self.pool(a2))
        a3 = self.s3(a2); z3 = self.h3(self.pool(a3))
        return z1, z2, z3

    # The   sub-networks ending at each exit (used by the explainers)
    def exit1(self, x):
        return self.h1(self.pool(self.s1(x)))



    def exit2(self, x):
        return self.h2(self.pool(self.s2(self.s1(x))))



    def exit3(self, x):
        return self.h3(self.pool(self.s3(self.s2(self.s1(x)))))

    @torch.no_grad()
    def route(self, x, T=(1., 1., 1.), tau=(0.95, 0.90)):
        """Confidence-based early exit: compute the next stage only for flows whose
        calibrated confidence at the current exit is below that exit's threshold."""
        B = x.shape[0]
        prob = torch.empty(B, 2); ex = torch.empty(B, dtype=torch.long); conf = torch.empty(B)
        a1 = self.s1(x)
        p = F.softmax(self.h1(self.pool(a1)) / T[0], 1); c = p.max(1).values
        m = c >= tau[0]
        prob[m], ex[m], conf[m] = p[m], 1, c[m]
        idx = (~m).nonzero().squeeze(1)
        if idx.numel():
            a2 = self.s2(a1[idx])
            p = F.softmax(self.h2(self.pool(a2)) / T[1], 1); c = p.max(1).values
            m2 = c >= tau[1]
            j = idx[m2]; prob[j], ex[j], conf[j] = p[m2], 2, c[m2]
            idx2, a2 = idx[~m2], a2[~m2]
            if idx2.numel():
                p = F.softmax(self.h3(self.pool(self.s3(a2))) / T[2], 1); c = p.max(1).values
                prob[idx2], ex[idx2], conf[idx2] = p, 3, c
        return prob, ex, conf




    @torch.no_grad()
    def route_single(self, x, T=(1., 1., 1.), tau=(0.95, 0.90)):
        """Same routing rule for one flow (x: [1, d]); used for single-flow latency."""
        a1 = self.s1(x); p = F.softmax(self.h1(a1.mean(1)) / T[0], 1); c = p.max().item()
        if c >= tau[0]:
            return p, 1, c
        a2 = self.s2(a1); p = F.softmax(self.h2(a2.mean(1)) / T[1], 1); c = p.max().item()
        if c >= tau[1]:
            return p, 2, c
        p = F.softmax(self.h3(self.s3(a2).mean(1)) / T[2], 1)
        return p, 3, p.max().item()




class SingleModel(nn.Module):
    """The stand-alone CNN,    BiLSTM or Transformer (ablation). The BiLSTM and Transformer
    read the same number of tokens as the backbone (L = ceil(d/4)), each token being a
    linear embedding of four consecutive features....    """


    def __init__(self, d, kind):
        super().__init__()
        self.kind, self.d, self.L = kind, d, n_tokens(d)
        if kind == 'cnn':
            self.body, dim = Stage1CNN(32), 32
        elif kind == 'bilstm':
            self.emb = nn.Linear(4, 32); self.body, dim = Stage2BiLSTM(32, 32), 64
        else:
            self.emb = nn.Linear(4, 64); self.body, dim = Stage3Transformer(self.L, 64), 64
        self.h = head(dim)

    def forward(self, x):
        if self.kind == 'cnn':
            a = self.body(x)
        else:
            x = F.pad(x, (0, 4 * self.L - self.d)).view(x.shape[0], self.L, 4)
            a = self.body(self.emb(x))
        return (self.h(a.mean(1)),)






class NoExitNet(EENet):
    """The same backbone,   trained and evaluated with the final exit only (no early exits).,.."""
    def forward(self, x):
        return (self.exit3(x),)





def add_prior(model, lp):
    """Copy of a model whose exit logits are shifted by lp (log class-prior correction)."""
    import copy
    m = copy.deepcopy(model)
    with torch.no_grad():
        for a in ('h1', 'h2', 'h3', 'h'):
            if hasattr(m, a):
                getattr(m, a)[-1].bias += lp
    return m


