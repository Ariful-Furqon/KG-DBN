# Deep Belief Network: greedy layer-wise RBM pretraining (CD-k) + supervised fine-tuning.

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RBM(nn.Module):
    # Restricted Boltzmann Machine with binary hidden units.
    #
    # visible="bernoulli" expects inputs in [0, 1]; visible="gaussian" expects
    # standardized inputs (zero mean, unit variance).

    def __init__(self, n_visible: int, n_hidden: int, visible: str = "bernoulli"):
        super().__init__()
        if visible not in ("bernoulli", "gaussian"):
            raise ValueError(f"visible must be 'bernoulli' or 'gaussian', got {visible!r}")
        self.visible = visible
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def p_h(self, v):
        return torch.sigmoid(v @ self.W + self.h_bias)

    def mean_v(self, h):
        activation = h @ self.W.t() + self.v_bias
        return torch.sigmoid(activation) if self.visible == "bernoulli" else activation

    def free_energy(self, v):
        hidden = F.softplus(v @ self.W + self.h_bias).sum(1)
        if self.visible == "bernoulli":
            return -(v @ self.v_bias) - hidden
        return 0.5 * ((v - self.v_bias) ** 2).sum(1) - hidden

    def contrastive_divergence(self, v0, k: int = 1):
        # Loss whose gradient is the CD-k update, plus the reconstruction error.
        v = v0
        for _ in range(k):
            h = torch.bernoulli(self.p_h(v))
            v = self.mean_v(h)
        v = v.detach()
        loss = self.free_energy(v0).mean() - self.free_energy(v).mean()
        return loss, F.mse_loss(v, v0).item()

    def pseudo_likelihood(self, v):
        # Estimate average log pseudo-likelihood per sample for Bernoulli visible.
        if self.visible != "bernoulli":
            return None
        with torch.no_grad():
            i = torch.randint(0, v.shape[1], (len(v),), device=v.device)
            v_corrupt = v.clone()
            v_corrupt.scatter_(1, i.unsqueeze(1), 1.0 - v.gather(1, i.unsqueeze(1)))
            fe_orig = self.free_energy(v)
            fe_corrupt = self.free_energy(v_corrupt)
            pl = -v.shape[1] * F.softplus(fe_orig - fe_corrupt)
            return pl.mean().item()

    def forward(self, v):
        return self.p_h(v)


class DBN:
    # Stacked RBMs whose weights initialize a sigmoid MLP classifier.
    #
    # With pretrain=False the same network is trained from random weights, which
    # serves as the "no pretraining" baseline.

    def __init__(
        self,
        hidden_layers=(128, 64),
        visible: str = "bernoulli",
        k: int = 1,
        pretrain: bool = True,
        pretrain_epochs: int = 30,
        pretrain_lr: float | None = None,
        finetune_epochs: int = 200,
        finetune_lr: float = 5e-3,
        batch_size: int = 64,
        dropout: float = 0.0,
        weight_decay: float = 1e-4,
        patience: int = 15,
        val_fraction: float = 0.1,
        seed: int = 42,
        device: str | None = None,
        verbose: bool = False,
    ):
        self.hidden_layers = tuple(hidden_layers)
        self.visible = visible
        self.k = k
        self.pretrain = pretrain
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_lr = pretrain_lr
        self.finetune_epochs = finetune_epochs
        self.finetune_lr = finetune_lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_fraction = val_fraction
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose

    # ------------------------------------------------------------------ utils
    def _tensor(self, X):
        return torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)

    def _batches(self, n, shuffle=True):
        order = torch.randperm(n, device=self.device) if shuffle else torch.arange(n, device=self.device)
        return order.split(self.batch_size)

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # --------------------------------------------------------------- training
    def _pretrain(self, X):
        self.rbms_ = []
        self.history_["pretrain"] = []
        self.history_["pseudo_likelihood"] = []
        v = X
        for depth, n_hidden in enumerate(self.hidden_layers):
            vis = self.visible if depth == 0 else "bernoulli"
            rbm = RBM(v.shape[1], n_hidden, vis).to(self.device)
            if self.pretrain_lr is not None:
                lr = self.pretrain_lr
            else:
                # Bernoulli visible units benefit from higher lr (~0.1) than Gaussian (~0.01)
                lr = 0.01 if vis == "gaussian" else 0.1
            opt = torch.optim.SGD(rbm.parameters(), lr=lr, momentum=0.9, weight_decay=self.weight_decay)
            errors = []

            # PL hanya bermakna jika input benar-benar biner ({0, 1}).
            # Layer 0 dengan visible="bernoulli" menerima masukan biner asli.
            # Layer >= 1 menerima probabilitas kontinu dari rbm(v) layer sebelumnya,
            # sehingga bit-flip PL tidak bermakna secara matematis dan dilewati (None)
            # daripada melakukan sampling Bernoulli buatan yang menambah variansi acak.
            track_pl = (depth == 0 and vis == "bernoulli")
            pls = [] if track_pl else None

            for epoch in range(self.pretrain_epochs):
                total = 0.0
                for idx in self._batches(len(v)):
                    loss, recon = rbm.contrastive_divergence(v[idx], self.k)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total += recon * len(idx)
                errors.append(total / len(v))
                if track_pl:
                    pls.append(rbm.pseudo_likelihood(v))
            self._log(f"RBM {depth + 1} ({v.shape[1]}->{n_hidden}): recon error {errors[0]:.4f} -> {errors[-1]:.4f}")
            self.history_["pretrain"].append(errors)
            self.history_["pseudo_likelihood"].append(pls)
            self.rbms_.append(rbm)
            with torch.no_grad():
                v = rbm(v)

    def _build_network(self, n_features, n_classes):
        layers, sizes = [], (n_features, *self.hidden_layers)
        for depth, (n_in, n_out) in enumerate(zip(sizes, sizes[1:])):
            linear = nn.Linear(n_in, n_out)
            if self.pretrain:
                with torch.no_grad():
                    linear.weight.copy_(self.rbms_[depth].W.t())
                    linear.bias.copy_(self.rbms_[depth].h_bias)
            layers += [linear, nn.Sigmoid()]
            if self.dropout:
                layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(sizes[-1], n_classes))
        return nn.Sequential(*layers).to(self.device)

    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        X = np.asarray(X, dtype=np.float32)

        if X_val is None and self.val_fraction:
            perm = rng.permutation(len(X))
            n_val = max(1, int(len(X) * self.val_fraction))
            val, train = perm[:n_val], perm[n_val:]
            X, X_val, y_idx, y_val_idx = X[train], X[val], y_idx[train], y_idx[val]
        elif X_val is not None:
            y_val_idx = np.searchsorted(self.classes_, np.asarray(y_val))

        Xt, yt = self._tensor(X), torch.as_tensor(y_idx, device=self.device)
        self.history_ = {"train_loss": [], "val_loss": []}

        if self.pretrain:
            self._pretrain(Xt)
        self.network_ = self._build_network(X.shape[1], len(self.classes_))

        opt = torch.optim.Adam(self.network_.parameters(), lr=self.finetune_lr, weight_decay=self.weight_decay)
        has_val = X_val is not None
        if has_val:
            Xv, yv = self._tensor(X_val), torch.as_tensor(y_val_idx, device=self.device)
        best, best_state, wait = float("inf"), None, 0

        for epoch in range(self.finetune_epochs):
            self.network_.train()
            total = 0.0
            for idx in self._batches(len(Xt)):
                loss = F.cross_entropy(self.network_(Xt[idx]), yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            self.history_["train_loss"].append(total / len(Xt))

            if not has_val:
                continue
            self.network_.eval()
            with torch.no_grad():
                val_loss = F.cross_entropy(self.network_(Xv), yv).item()
            self.history_["val_loss"].append(val_loss)
            if val_loss < best - 1e-4:
                best, best_state, wait = val_loss, copy.deepcopy(self.network_.state_dict()), 0
            else:
                wait += 1
                if wait >= self.patience:
                    self._log(f"early stop at epoch {epoch + 1}, best val loss {best:.4f}")
                    break

        if best_state is not None:
            self.network_.load_state_dict(best_state)
        return self

    # -------------------------------------------------------------- inference
    def predict_proba(self, X):
        self.network_.eval()
        with torch.no_grad():
            return torch.softmax(self.network_(self._tensor(X)), dim=1).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
