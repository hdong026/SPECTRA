import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import scipy.sparse as sp


class GraphSpectralUtils(nn.Module):
    """
    Minimal SPECTRA tokenizer utilities:
    - cache Laplacian eigenvectors U (N,k) and U_T (k,N) as buffers
    - deterministic sign canonicalization (for reproducibility)
    - correct GFT/IGFT einsum
    - inference-time sign flip (robustness test)
    - basis-robust total energy summary (for diagnostics/conditioning)
    """
    def __init__(self, adj_mx_path, device=None, top_k=None, node_size=None):
        super().__init__()
        self.top_k = top_k
        self.node_size = node_size

        U, U_T, eigvals = self._compute_eigenvectors(adj_mx_path)

        # register as buffers: moved automatically by .to(device), included in state_dict
        self.register_buffer("U", U)
        self.register_buffer("U_T", U_T)
        self.register_buffer("eigvals", eigvals)

    def _compute_eigenvectors(self, adj_mx_path):
        try:
            with open(adj_mx_path, 'rb') as f:
                pickle_data = pickle.load(f)

            # Robust loading
            adj = None
            if isinstance(pickle_data, (list, tuple)):
                adj = pickle_data[2]
            elif isinstance(pickle_data, dict):
                adj = pickle_data.get('adj_mx', None)
            else:
                adj = pickle_data
            if adj is None:
                raise ValueError("adj_mx not found in pickle")

            if sp.issparse(adj):
                adj = adj.toarray()
            adj = np.array(adj, dtype=np.float32)

            # Force symmetrization (PEMS04 has directed edges)
            adj = np.maximum(adj, adj.T)

            # Add self-loops
            adj = adj + np.eye(adj.shape[0], dtype=np.float32)

            # Normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
            d = np.array(adj.sum(1)).flatten()
            d_inv_sqrt = np.power(d, -0.5)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            D_inv_sqrt = sp.diags(d_inv_sqrt)

            normalized_adj = D_inv_sqrt @ adj @ D_inv_sqrt
            L = np.eye(adj.shape[0], dtype=np.float32) - normalized_adj

            # eigen decomposition for symmetric matrix
            vals, vecs = np.linalg.eigh(L)

            # sort ascending by eigenvalues
            idx = vals.argsort()
            vals = vals[idx]
            vecs = vecs[:, idx]

            # deterministic sign canonicalization (fix eigenvector sign ambiguity)
            for i in range(vecs.shape[1]):
                j = np.argmax(np.abs(vecs[:, i]))
                if vecs[j, i] < 0:
                    vecs[:, i] *= -1.0

            if self.top_k is not None:
                vecs = vecs[:, :self.top_k]
                vals = vals[:self.top_k]

            U = torch.from_numpy(vecs).float()         # [N,k]
            U_T = U.transpose(0, 1).contiguous()       # [k,N]
            eigvals = torch.from_numpy(vals).float()   # [k] (or [N] if top_k None)

            return U, U_T, eigvals

        except Exception as e:
            print(f"Warning: Failed to load graph from {adj_mx_path} ({e}). using Identity.")
            N = int(self.node_size) if self.node_size else 307
            eye = torch.eye(N).float()
            eigvals = torch.zeros(N).float()
            return eye, eye, eigvals

    def gft(self, x):
        """
        x: [B,T,N,C] -> [B,T,k,C]
        """
        return torch.einsum('kn, btnc -> btkc', self.U_T, x)

    def igft(self, x):
        """
        x: [B,T,k,C] -> [B,T,N,C]
        """
        return torch.einsum('nk, btkc -> btnc', self.U, x)

    def total_energy(self, spec_tokens):
        """
        Basis-robust summary:
        spec_tokens: [B,T,k,C] -> [B,T,1,C]
        """
        return (spec_tokens ** 2).sum(dim=2, keepdim=True)

    def sign_flip_(self):
        """
        Inference-time robustness test:
        random sign flip on eigenvectors (columns of U).
        """
        k = self.U.shape[1]
        flips = (torch.randint(0, 2, (k,), device=self.U.device) * 2 - 1).to(self.U.dtype)  # ±1
        self.U.mul_(flips.view(1, -1))
        self.U_T = self.U.transpose(0, 1).contiguous()


# ===== KAN components (kept as-is, except formatting) =====

class SimpleKANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5):
        super().__init__()
        self.grid_size = grid_size
        self.base_linear = nn.Linear(in_features, out_features)
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size))
        self.grid = nn.Parameter(torch.linspace(-3, 3, grid_size), requires_grad=False)
        nn.init.kaiming_uniform_(self.base_linear.weight, nonlinearity='linear')
        nn.init.uniform_(self.spline_weight, -0.1, 0.1)

    def forward(self, x):
        base = self.base_linear(F.silu(x))
        x_uns = x.unsqueeze(-1)
        basis = torch.exp(-((x_uns - self.grid) / (2 / (self.grid_size - 1) + 1e-5)) ** 2)
        spline = torch.einsum("...ig,oig->...o", basis, self.spline_weight)
        return base + spline


class SpectralKANVectorField(nn.Module):
    def __init__(self, input_dim, hidden_dim, cond_dim, grid_size=5):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.kan_layers = nn.Sequential(
            SimpleKANLinear(hidden_dim, hidden_dim, grid_size), nn.SiLU(),
            SimpleKANLinear(hidden_dim, hidden_dim, grid_size), nn.SiLU(),
            SimpleKANLinear(hidden_dim, hidden_dim, grid_size)
        )
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t, cond):
        if t.ndim == 1:
            t_emb = self.time_mlp(t.view(-1, 1)).view(t.shape[0], 1, 1, -1)
        else:
            t_emb = self.time_mlp(t.unsqueeze(-1))
        x_emb = self.input_proj(x)
        cond_emb = self.cond_proj(cond)
        h = x_emb + cond_emb + t_emb
        h = self.kan_layers(h)
        return self.output_proj(h)