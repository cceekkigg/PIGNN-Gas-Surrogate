

# ============================

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CONFIG, CONSTANTS
from src.dataset import get_or_compute_norm_state
from .projection import MassBalanceProjection, PressureDropIntegration


class MessagePassingLayer(nn.Module):
    """
    Node embedding trunk (no direct p2 regression here).

    Uses GINEConv to incorporate edge attributes (pipe features).
    """
    def __init__(self, node_in_dim, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        aggr = str(CONFIG.get("gnn_aggr", "add")).lower()
        self.share_edge_encoder = bool(CONFIG.get("share_edge_encoder", False))

        self.enc = nn.Linear(node_in_dim, hidden_dim)
        if not self.share_edge_encoder:
            self.edge_enc = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.edge_enc = None

        GINEConv = __import__("torch_geometric.nn", fromlist=["GINEConv"]).GINEConv

        self.convs = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                aggr=aggr,
            )
            for _ in range(num_layers)
        ])
        self.dropout = float(CONFIG.get("dropout", 0.0))

    def forward(self, x, edge_index, edge_attr_hidden):
        h = self.enc(x)
        if not self.share_edge_encoder:
            edge_attr_hidden = self.edge_enc(edge_attr_hidden)
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr_hidden)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class FlowPredictionLayer(nn.Module):
    """Predicts normalized flow latent Qtilde_norm per directed edge."""
    def __init__(
        self,
        hidden_dim,
        edge_dim=3,
        use_edge_context=False,
        use_dp2_context=False,
    ):
        super().__init__()
        self.use_edge_context = bool(use_edge_context)
        self.use_dp2_context = bool(use_dp2_context)
        self.share_edge_encoder = bool(CONFIG.get("share_edge_encoder", False))

        input_dim = 2 * hidden_dim
        if self.use_edge_context:
            if not self.share_edge_encoder:
                self.edge_enc = nn.Sequential(
                    nn.Linear(edge_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            else:
                self.edge_enc = None
            input_dim += hidden_dim
        else:
            self.edge_enc = None

        if self.use_dp2_context:
            self.dp2_enc = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            input_dim += hidden_dim
        else:
            self.dp2_enc = None

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_embeddings,
        edge_index,
        edge_attr_hidden=None,
        dp2_phys_norm=None,
    ):
        u, v = edge_index
        h_u = node_embeddings[u]
        h_v = node_embeddings[v]
        features = [h_u, h_v]

        if self.use_edge_context:
            assert edge_attr_hidden is not None, "edge_attr_hidden required for flow edge context"
            if self.share_edge_encoder:
                features.append(edge_attr_hidden)
            else:
                features.append(self.edge_enc(edge_attr_hidden))

        if self.use_dp2_context:
            assert dp2_phys_norm is not None, "dp2_phys_norm required for flow dp2 context"
            features.append(self.dp2_enc(dp2_phys_norm))

        h_e = torch.cat(features, dim=-1)
        return self.mlp(h_e)  # [E,1]


class EdgeDP2PredictionLayer(nn.Module):
    """
    Predicts normalized dp2 per directed edge:
        dp2_norm = (p2[v]-p2[u]) / c_p2
    """
    def __init__(self, hidden_dim, edge_dim=3):
        super().__init__()
        self.share_edge_encoder = bool(CONFIG.get("share_edge_encoder", False))
        if not self.share_edge_encoder:
            self.edge_enc = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.edge_enc = None
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embeddings, edge_index, edge_attr_hidden):
        u, v = edge_index
        h_u = node_embeddings[u]
        h_v = node_embeddings[v]
        if self.share_edge_encoder:
            h_e = edge_attr_hidden
        else:
            h_e = self.edge_enc(edge_attr_hidden)
        h = torch.cat([h_u, h_v, h_e], dim=-1)
        return self.mlp(h)  # [E,1] dp2_norm


def _blended_friction_factor_torch(Re, eps_over_D):
    Re_safe = torch.clamp(Re, min=1.0)
    f_lam = 64.0 / Re_safe

    term = torch.clamp((eps_over_D / 3.7) ** 1.11 + 6.9 / Re_safe, min=1e-12)
    inv_sqrt_f = -1.8 * torch.log10(term)
    f_turb = 1.0 / torch.clamp(inv_sqrt_f ** 2, min=1e-8)

    w = torch.clamp((Re_safe - 2000.0) / 2000.0, 0.0, 1.0)
    f = torch.where(
        Re_safe <= 2000.0,
        f_lam,
        torch.where(Re_safe >= 4000.0, f_turb, (1.0 - w) * f_lam + w * f_turb),
    )
    return torch.clamp(f, min=1e-6, max=1.0)


def _dp2_from_flow_phys_torch(q_phys, edge_attr_phys):
    """
    Compute signed directed dp2 = p2[v] - p2[u] from canonical directed q_{u->v}.
    """
    bar_to_pa = 1e5
    rho_n = (
        (float(CONSTANTS["P_ATM_BAR"]) * bar_to_pa)
        * float(CONSTANTS["GAS_MOLAR_MASS"])
        / (float(CONSTANTS["R_UNIVERSAL"]) * float(CONSTANTS["TEMP_NORMAL_K"]))
    )
    r_spec = float(CONSTANTS["R_UNIVERSAL"]) / float(CONSTANTS["GAS_MOLAR_MASS"])
    mu = float(CONSTANTS["VISCOSITY"])

    q = q_phys.view(-1)
    L = edge_attr_phys[:, 0].clamp_min(1e-12)
    D = edge_attr_phys[:, 1].clamp_min(1e-12)
    roughness = edge_attr_phys[:, 2].clamp_min(1e-12)

    q_abs = torch.abs(q)
    m_dot = q_abs * rho_n
    Re = (4.0 * m_dot) / (torch.pi * D * mu)
    eps_over_D = roughness / D
    friction = _blended_friction_factor_torch(Re, eps_over_D)

    coeff = (
        friction
        * 16.0
        * r_spec
        * float(CONSTANTS["TEMP_K"])
        * L
        * (rho_n ** 2)
        / ((torch.pi ** 2) * (D ** 5) * (bar_to_pa ** 2))
    )
    dp2 = -coeff * q * q_abs
    return dp2.unsqueeze(1)


class GasGNN(nn.Module):
    """
    Edge-first dp2 prediction + constrained Laplacian integration to nodal p2.

    Outputs:
        p2_pred (bar^2), q_proj (Nm3/s)
    Optionally returns q_unproj and dp2_hat for training losses.
    """
    def __init__(self, node_in_dim, hidden_dim, num_layers, **kwargs):
        super().__init__()

        norm_state = get_or_compute_norm_state(CONFIG['train_file'])
        self.c_p2 = float(norm_state['c_p2'])
        self.q0 = float(norm_state['q0'])

        self.share_edge_encoder = bool(CONFIG.get("share_edge_encoder", False))
        if self.share_edge_encoder:
            self.edge_encoder = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.edge_encoder = None
        self.mp = MessagePassingLayer(node_in_dim, hidden_dim, num_layers)

        self.dp2_pred = EdgeDP2PredictionLayer(hidden_dim, edge_dim=3)
        self.flow_pred = FlowPredictionLayer(
            hidden_dim,
            edge_dim=3,
            use_edge_context=bool(CONFIG.get("flow_use_edge_context", False)),
            use_dp2_context=bool(CONFIG.get("flow_condition_on_dp2", False)),
        )

        self.projection = MassBalanceProjection(
            max_iter=int(CONFIG.get("projection_max_iter", 100)),
            tolerance=float(CONFIG.get("projection_tolerance", 1e-6)),
            solve_dtype=CONFIG.get("projection_dtype", "float64"),
            use_direct_cache=bool(CONFIG.get("use_direct_solver_cache", True)),
        )

        self.pressure_integrator = PressureDropIntegration(
            max_iter=int(CONFIG.get("pressure_cg_max_iter", 2000)),
            tolerance=float(CONFIG.get("pressure_cg_tolerance", 1e-10)),
            damping=float(CONFIG.get("pressure_laplacian_damping", 0.0)),
            solve_dtype=CONFIG.get("pressure_solver_dtype", "float64"),
            use_direct_cache=bool(CONFIG.get("use_direct_solver_cache", True)),
        )

    def forward(self, data, return_unprojected: bool = False, return_dp2: bool = False):
        """
        Forward pass (Option A, robust):
        - DO NOT assume forward edges are in the first half.
        - Use rev_edge_id to select unique physical edges, predict only once per pair.
        - Reconstruct directed quantities by antisymmetry.
        - Integrate dp2 -> p2 with slack clamp.
        - Predict flow + antisymmetry + hard mass-balance projection.
        """

        # ============================================================
        # 0) Basic checks
        # ============================================================
        assert hasattr(data, "rev_edge_id"), "Data object must contain rev_edge_id"
        assert data.edge_index.size(1) == data.rev_edge_id.numel(), \
            "edge_index / rev_edge_id length mismatch"
        edge_index = data.edge_index
        rev = data.rev_edge_id.to(edge_index.device)

        E = int(edge_index.size(1))
        assert rev.numel() == E, f"rev_edge_id length {rev.numel()} != E {E}"

        # pick one representative per (u,v)<->(v,u) pair
        if hasattr(data, "phys_edge_mask"):
            phys_mask = data.phys_edge_mask.view(-1).bool()
        else:
            eid = torch.arange(E, device=edge_index.device)
            phys_mask = eid < rev  # [E] bool, ~E/2 true
        phys_idx = phys_mask.nonzero(as_tuple=False).view(-1)  # [E_phys]
        E_phys = int(phys_idx.numel())
        assert E_phys > 0, "No physical edges selected (check rev_edge_id correctness)"

        edge_index_phys = edge_index[:, phys_idx]           # [2, E_phys]
        edge_attr_phys_norm = data.edge_attr[phys_idx]      # [E_phys, 3]
        if self.share_edge_encoder:
            edge_attr_hidden = self.edge_encoder(data.edge_attr)
            edge_attr_hidden_phys = edge_attr_hidden[phys_idx]
        else:
            edge_attr_hidden = data.edge_attr
            edge_attr_hidden_phys = edge_attr_phys_norm

        # ============================================================
        # 1) Node embedding trunk
        # ============================================================
        node_embeddings = self.mp(data.x, edge_index, edge_attr_hidden)

        # ============================================================
        # 2) Predict dp2 ONLY on physical edges, reconstruct directed
        # ============================================================
        dp2_phys_norm = self.dp2_pred(node_embeddings, edge_index_phys, edge_attr_hidden_phys)  # [E_phys,1]
        dp2_phys = dp2_phys_norm * self.c_p2                                                  # [E_phys,1]

        dp2_hat = torch.zeros((E, 1), dtype=dp2_phys.dtype, device=dp2_phys.device)
        dp2_hat[phys_idx] = dp2_phys
        dp2_hat[rev[phys_idx]] = -dp2_phys

        # (optional safety) enforce exact antisymmetry again
        dp2_hat = 0.5 * (dp2_hat - dp2_hat[rev])

        # ============================================================
        # 3) Integrate to nodal p2 with slack clamp
        # ============================================================
        pset2_graph = data.pset2_bar2
        if torch.is_tensor(pset2_graph) and pset2_graph.dim() == 0:
            pset2_graph = pset2_graph.view(1)

        pset2_per_node = pset2_graph[data.batch].unsqueeze(1)  # [N,1]

        p2_pred, _delta = self.pressure_integrator(
            dp2_hat=dp2_hat,
            edge_index=edge_index,
            is_slack_mask=data.is_slack,
            pset2_per_node=pset2_per_node,
            graph_ptr=getattr(data, "ptr", None),
        )

        # ============================================================
        # 4) Predict flow ONLY on physical edges, reconstruct directed
        # ============================================================
        # Prevent sinh overflow when older checkpoints emit large latent flow logits.
        flow_dp2_phys = dp2_phys_norm
        if bool(CONFIG.get("flow_dp2_detach", True)):
            flow_dp2_phys = flow_dp2_phys.detach()

        Qtilde_phys = self.flow_pred(
            node_embeddings,
            edge_index_phys,
            edge_attr_hidden=edge_attr_hidden_phys,
            dp2_phys_norm=flow_dp2_phys,
        )  # [E_phys,1]
        Qtilde_phys = torch.clamp(Qtilde_phys, min=-15.0, max=15.0)
        q_phys = self.q0 * torch.sinh(Qtilde_phys)                      # [E_phys,1]

        q_unprojected_pred = torch.zeros((E, 1), dtype=q_phys.dtype, device=q_phys.device)
        q_unprojected_pred[phys_idx] = q_phys
        q_unprojected_pred[rev[phys_idx]] = -q_phys

        # safety antisym
        q_unprojected_pred = 0.5 * (q_unprojected_pred - q_unprojected_pred[rev])

        # ============================================================
        # 5) Hard mass-balance projection
        # ============================================================
        q_projected, residual_pre, residual_post = self.projection(
            q_unprojected_pred,
            edge_index,
            data.s_Nm3s,
            data.is_slack,
            graph_ptr=getattr(data, "ptr", None),
        )

        if bool(CONFIG.get("final_pressure_from_flow", False)):
            edge_attr_phys = data.edge_attr_phys[phys_idx]
            dp2_phys_from_q = _dp2_from_flow_phys_torch(q_projected[phys_idx], edge_attr_phys)

            dp2_from_q = torch.zeros((E, 1), dtype=dp2_phys_from_q.dtype, device=dp2_phys_from_q.device)
            dp2_from_q[phys_idx] = dp2_phys_from_q
            dp2_from_q[rev[phys_idx]] = -dp2_phys_from_q
            dp2_from_q = 0.5 * (dp2_from_q - dp2_from_q[rev])

            p2_pred, _delta = self.pressure_integrator(
                dp2_hat=dp2_from_q,
                edge_index=edge_index,
                is_slack_mask=data.is_slack,
                pset2_per_node=pset2_per_node,
                graph_ptr=getattr(data, "ptr", None),
            )

        p2_floor = float(CONFIG.get("final_p2_floor_bar2", 0.0))
        if p2_floor > 0.0:
            p2_pred = torch.clamp(p2_pred, min=p2_floor)

        # ============================================================
        # 6) Optional returns
        # ============================================================
        if return_unprojected and return_dp2:
            return p2_pred, q_projected, q_unprojected_pred, dp2_hat
        if return_unprojected:
            return p2_pred, q_projected, q_unprojected_pred
        if return_dp2:
            return p2_pred, q_projected, dp2_hat
        return p2_pred, q_projected



