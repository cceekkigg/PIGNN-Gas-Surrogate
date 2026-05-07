

import json
import torch
import numpy as np
from tqdm import tqdm
import networkx as nx
from pathlib import Path
from collections import defaultdict
from torch_geometric.data import InMemoryDataset, Data

from src.config import PROCESSED_DIR, DATA_DIR, CONFIG
from .projection import validate_mass_balance


def build_active_edge_mask_single_graph(dp2_abs: torch.Tensor, q: float, topk_min: int) -> torch.Tensor:
    mask = torch.zeros_like(dp2_abs, dtype=torch.bool)
    if dp2_abs.numel() == 0:
        return mask
    if dp2_abs.numel() <= topk_min:
        mask[:] = True
        return mask

    thr = torch.quantile(dp2_abs, q)
    selected = dp2_abs >= thr
    if int(selected.sum().item()) < topk_min:
        topk = torch.topk(dp2_abs, k=topk_min, largest=True).indices
        mask[topk] = True
    else:
        mask[selected] = True
    return mask


def build_rev_edge_id(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Robust reverse-edge mapper supporting parallel edges.
    Ensures rev[rev[i]] == i and (u_i, v_i) == (v_rev, u_rev).
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must be [2,E], got {tuple(edge_index.shape)}")

    E = int(edge_index.size(1))
    device = edge_index.device

    # ---- FAST PATH: your dataset construction is [fwd, bwd] in identical order ----
    if E % 2 == 0:
        M = E // 2
        u0, v0 = edge_index[0, :M], edge_index[1, :M]
        u1, v1 = edge_index[0, M:], edge_index[1, M:]
        if torch.equal(u0, v1) and torch.equal(v0, u1):
            rev = torch.empty(E, dtype=torch.long, device=device)
            rev[:M] = torch.arange(M, E, device=device)
            rev[M:] = torch.arange(0, M, device=device)
            return rev

    # ---- GENERAL PATH: multi-map pairing for arbitrary ordering / parallel edges ----
    u = edge_index[0].detach().cpu().tolist()
    v = edge_index[1].detach().cpu().tolist()

    pair_to_idxs = defaultdict(list)
    for i, (a, b) in enumerate(zip(u, v)):
        pair_to_idxs[(a, b)].append(i)

    rev = [-1] * E
    visited = set()

    for (a, b), idxs_ab in pair_to_idxs.items():
        if (a, b) in visited:
            continue
        idxs_ba = pair_to_idxs.get((b, a), None)
        if idxs_ba is None:
            raise RuntimeError(f"Missing reverse edges for pair ({a},{b}).")

        if len(idxs_ab) != len(idxs_ba):
            raise RuntimeError(
                f"Unbalanced parallel edges for ({a},{b}) vs ({b},{a}): "
                f"{len(idxs_ab)} vs {len(idxs_ba)}"
            )

        idxs_ab = sorted(idxs_ab)
        idxs_ba = sorted(idxs_ba)

        for i, j in zip(idxs_ab, idxs_ba):
            rev[i] = j
            rev[j] = i

        visited.add((a, b))
        visited.add((b, a))

    rev = torch.tensor(rev, dtype=torch.long, device=device)

    # ---- SANITY: involution check ----
    eid = torch.arange(E, device=device)
    if not torch.equal(rev[rev], eid):
        bad = (rev[rev] != eid).nonzero(as_tuple=False).view(-1)[:10].tolist()
        raise RuntimeError(f"rev_edge_id is not an involution. First bad edge ids: {bad}")

    # ---- SANITY: endpoint reversal check ----
    uu, vv = edge_index[0], edge_index[1]
    if not (torch.equal(uu, vv[rev]) and torch.equal(vv, uu[rev])):
        raise RuntimeError("rev_edge_id does not map edges to exact endpoint reversals.")

    return rev


def get_or_compute_norm_state(train_file_path: Path):
    """
    Computes, tests, and saves/loads normalization constants for a given training dataset.
    Includes cache invalidation based on key config knobs.
    """
    dataset_name_stem = train_file_path.stem
    output_path = PROCESSED_DIR / f"norm_state_{dataset_name_stem}.json"

    # cache key for invalidation
    q_active = float(CONFIG.get("dp2_active_quantile", 0.99))
    percentile = q_active * 100.0
    norm_version = {
        "p2_scale_percentile": percentile,
    }

    if output_path.exists():
        with open(output_path, "r") as f:
            norm_state = json.load(f)

        cached_ver = norm_state.get("_version", {})
        loaded_ok = (cached_ver == norm_version)

        if loaded_ok:
            return norm_state

    class AdHocRawDataLoader:
        def __init__(self, npz_path):
            self.npz_path = npz_path
            with np.load(self.npz_path, allow_pickle=True) as raw:
                self.edge_static = raw["edge_static"]
                self.demands = raw["demand"]
                self.pressures = raw["pressure"]
                self.flows = raw["flow"]

            L_m = np.maximum(self.edge_static[:, 2] * 1000.0, 1e-6)
            D_m = self.edge_static[:, 3] / 1000.0
            eps_m = self.edge_static[:, 4] / 1000.0

            self.edge_attr_phys_fwd = torch.tensor(
                np.stack([L_m, D_m, eps_m], axis=1), dtype=torch.float32
            )
            self.edge_attr_phys = torch.cat([self.edge_attr_phys_fwd, self.edge_attr_phys_fwd])

            self.num_scenarios, _ = self.demands.shape

            # fixed slack node id based on minimal variance across scenarios
            p_variance = np.var(self.pressures, axis=0)
            self.slack_node_id = int(np.argmin(p_variance))

        def __len__(self):
            return self.num_scenarios

        def __getitem__(self, idx):
            p_true_bar = torch.tensor(self.pressures[idx], dtype=torch.float32)
            p2_true_bar2 = p_true_bar ** 2

            pset2_bar2 = p2_true_bar2[self.slack_node_id]

            Q_true_Nm3s_fwd_scenario = torch.tensor(self.flows[idx], dtype=torch.float32)

            return {
                "edge_attr_phys": self.edge_attr_phys,
                "Q_true_Nm3s": Q_true_Nm3s_fwd_scenario,
                "p2_true_bar2": p2_true_bar2,
                "pset2_bar2": pset2_bar2,
            }

    adhoc_loader = AdHocRawDataLoader(npz_path=train_file_path)

    all_edge_attr_phys = []
    all_Q_true_Nm3s = []
    all_p2_delta_bar2 = []

    for i in tqdm(range(len(adhoc_loader)), desc="Collecting physical values"):
        data = adhoc_loader[i]
        all_edge_attr_phys.append(data["edge_attr_phys"])
        all_Q_true_Nm3s.append(data["Q_true_Nm3s"])
        dp2 = data["p2_true_bar2"] - data["pset2_bar2"]
        all_p2_delta_bar2.append(dp2)

    all_edge_attr_phys = torch.cat(all_edge_attr_phys, dim=0).numpy()
    all_Q_true_Nm3s = torch.cat(all_Q_true_Nm3s, dim=0).numpy().flatten()
    all_p2_delta_bar2 = torch.cat(all_p2_delta_bar2, dim=0).numpy().flatten()

    # edge normalization (log)
    edge_log_eps = 1e-6
    log_edge_attrs = np.log(all_edge_attr_phys + edge_log_eps)
    edge_feat_mean = np.mean(log_edge_attrs, axis=0)
    edge_feat_std = np.std(log_edge_attrs, axis=0)

    # pressure scale: tail-safe percentile
    abs_dp2 = np.abs(all_p2_delta_bar2)
    abs_dp2 = abs_dp2[abs_dp2 > 1e-6]  # avoid zeros dominating
    c_p2 = np.percentile(abs_dp2, percentile)
    c_p2 = float(max(c_p2, 1.0))

    # flow scale
    q0 = np.median(np.abs(all_Q_true_Nm3s[np.abs(all_Q_true_Nm3s) > 1e-3]))
    q0 = float(max(q0, 1e-6))

    norm_state = {
        "_version": norm_version,
        "edge_log_eps": float(edge_log_eps),
        "edge_feat_mean": edge_feat_mean.tolist(),
        "edge_feat_std": edge_feat_std.tolist(),
        "c_p2": float(c_p2),
        "q0": float(q0),
    }

    # round-trip sanity checks
    x_orig = all_edge_attr_phys[0]
    x_norm = (np.log(x_orig + edge_log_eps) - edge_feat_mean) / (edge_feat_std + 1e-12)
    x_denorm = np.exp(x_norm * edge_feat_std + edge_feat_mean) - edge_log_eps
    edge_rel_error = np.max(np.abs(x_orig - x_denorm) / (np.abs(x_orig) + 1e-9))
    assert edge_rel_error < 1e-6, f"Edge norm round-trip failed! Error: {edge_rel_error}"

    q_candidates = all_Q_true_Nm3s[np.abs(all_Q_true_Nm3s) > 1.0]
    if q_candidates.size > 0:
        q_orig = float(q_candidates[0])
        q_norm = np.arcsinh(q_orig / q0)
        q_denorm = q0 * np.sinh(q_norm)
        flow_rel_error = np.abs(q_orig - q_denorm) / (np.abs(q_orig) + 1e-9)
        assert flow_rel_error < 1e-6, f"Flow norm round-trip failed! Error: {flow_rel_error}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(norm_state, f, indent=2)

    return norm_state


def _processed_cache_name(npz_path: Path) -> str:
    dataset_stem = npz_path.stem
    bps = int(CONFIG.get("broadcast_pset", False))
    k = int(CONFIG.get("lap_pe_k", 0))
    aq = int(round(float(CONFIG.get("dp2_active_quantile", 0.95)) * 100))
    tk = int(CONFIG.get("dp2_active_topk_min", 20))
    am = int(bool(CONFIG.get("use_precomputed_active_mask", True)))
    return f"graph_cache_{dataset_stem}_bps{bps}_lap{k}_aq{aq}_tk{tk}_am{am}.pt"


def cleanup_processed_dir():
    keep_names = {
        f"norm_state_{Path(CONFIG['train_file']).stem}.json",
        _processed_cache_name(Path(CONFIG["train_file"])),
    }
    for test_file in CONFIG.get("test_files", []):
        keep_names.add(_processed_cache_name(Path(test_file)))

    for path in PROCESSED_DIR.iterdir():
        if path.name in keep_names:
            continue
        if path.is_file():
            path.unlink()


class GasData(Data):
    """
    Custom Data so that rev_edge_id (edge-local indices) are incremented properly when batching.
    """
    def __inc__(self, key, value, *args, **kwargs):
        if key == "rev_edge_id":
            # rev_edge_id indexes edges, so offset by number of edges in this graph
            return self.edge_index.size(1)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "rev_edge_id":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


class GasDataProcessor(InMemoryDataset):
    """
    Loads raw .npz simulation data and prepares p^2 and Q targets.
    """

    def __init__(self, npz_path, norm_state, transform=None, pre_transform=None):
        self.npz_path = Path(npz_path)
        self.norm_state = norm_state

        if not self.npz_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.npz_path}")

        super().__init__(str(DATA_DIR), transform, pre_transform)
        cleanup_processed_dir()
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [self.npz_path.name]

    @property
    def processed_file_names(self):
        return [_processed_cache_name(self.npz_path)]

    def process(self):
        """Loads and validates raw .npz data, creating per-scenario graphs."""

        with np.load(self.npz_path, allow_pickle=True) as raw:
            edge_static = raw["edge_static"]
            demands = raw["demand"]
            pressures = raw["pressure"]
            flows = raw["flow"]

        num_scenarios, num_nodes = demands.shape

        edge_log_eps = float(self.norm_state["edge_log_eps"])
        edge_feat_mean = np.array(self.norm_state["edge_feat_mean"], dtype=np.float64)
        edge_feat_std = np.array(self.norm_state["edge_feat_std"], dtype=np.float64)

        # slack node id is fixed across scenarios via minimal pressure variance
        p_variance = np.var(pressures, axis=0)
        slack_node_id = int(np.argmin(p_variance))

        u_ids, v_ids = edge_static[:, 0].astype(int), edge_static[:, 1].astype(int)

        edge_index_fwd = np.stack([u_ids, v_ids], axis=0)
        edge_index_bwd = np.stack([v_ids, u_ids], axis=0)
        edge_index = torch.tensor(
            np.concatenate([edge_index_fwd, edge_index_bwd], axis=1), dtype=torch.long
        )
        # =========================
        # CHECKPOINT C1: physical edge_index sanity
        # =========================
        E = edge_index.size(1)
        assert E % 2 == 0, "Physical edges must be bidirectional (E even)"

        u, v = edge_index
        assert not (u == v).any(), "Self-loop detected in physical graph"

        # degree check (physical network)
        from torch_geometric.utils import degree
        deg = degree(u, num_nodes=num_nodes)
        assert deg.max() <= 10, f"Unphysical node degree detected: max_deg={deg.max().item()}"

        # undirected duplicate check
        u0 = torch.minimum(u, v)
        v0 = torch.maximum(u, v)
        key = u0 * num_nodes + v0
        _, counts = torch.unique(key, return_counts=True)
        assert counts.max() == 2, "Duplicated or missing reverse physical edges detected"


        # physical edge attrs
        L_m = np.maximum(edge_static[:, 2] * 1000.0, 1e-6)
        D_m = edge_static[:, 3] / 1000.0
        eps_m = edge_static[:, 4] / 1000.0

        edge_attr_phys_fwd = np.stack([L_m, D_m, eps_m], axis=1)
        edge_attr_phys = torch.tensor(
            np.concatenate([edge_attr_phys_fwd, edge_attr_phys_fwd]), dtype=torch.float32
        )

        # normalized edge attrs (log)
        log_edge_attrs = np.log(edge_attr_phys_fwd + edge_log_eps)
        edge_attr_norm_fwd = (log_edge_attrs - edge_feat_mean) / (edge_feat_std + 1e-8)
        edge_attr_norm = torch.tensor(
            np.concatenate([edge_attr_norm_fwd, edge_attr_norm_fwd]),
            dtype=torch.float32,
        )

        # topo distance to slack
        topology_graph = nx.Graph(list(zip(u_ids, v_ids)))
        try:
            distances = nx.single_source_shortest_path_length(topology_graph, slack_node_id)
            max_dist = max(distances.values()) if distances else 1.0
        except nx.NetworkXNoPath:
            distances = {i: 0 for i in range(num_nodes)}
            max_dist = 1.0
        topo_distance = torch.tensor(
            [distances.get(i, max_dist) / max_dist for i in range(num_nodes)],
            dtype=torch.float32,
        )

        # LapPE: compute once (shared across scenarios because topology fixed)
        lap_pe = None
        lap_pe_k_cfg = int(CONFIG.get("lap_pe_k", 0))
        if lap_pe_k_cfg > 0 and num_nodes > 1:
            adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
            for u, v in zip(u_ids, v_ids):
                adj[u, v] = 1.0
                adj[v, u] = 1.0

            deg = adj.sum(dim=1)
            inv_sqrt_deg = torch.pow(deg.clamp(min=1e-12), -0.5)
            D_inv_sqrt = torch.diag(inv_sqrt_deg)
            L = torch.eye(num_nodes, dtype=torch.float64) - D_inv_sqrt @ adj @ D_inv_sqrt

            eig_vals, eig_vecs = torch.linalg.eigh(L)

            max_k = min(lap_pe_k_cfg, num_nodes - 1)
            if max_k > 0:
                lap_pe = eig_vecs[:, 1:1 + max_k].to(torch.float32)  # [N, max_k]

        data_list = []
        q0 = float(self.norm_state["q0"])
        c_p2 = float(self.norm_state["c_p2"])
        q_act = float(CONFIG.get("dp2_active_quantile", 0.99))
        topk_min = int(CONFIG.get("dp2_active_topk_min", 8))
        use_precomputed_active_mask = bool(CONFIG.get("use_precomputed_active_mask", True))

        for scenario_index in tqdm(range(num_scenarios), desc="Building graphs"):
            s_Nm3s = torch.tensor(demands[scenario_index], dtype=torch.float32)  # [N]
            p_true_bar = torch.tensor(pressures[scenario_index], dtype=torch.float32)  # [N]
            p2_true_bar2 = p_true_bar ** 2  # [N]

            Q_true_Nm3s_fwd = torch.tensor(flows[scenario_index], dtype=torch.float32)  # [E_fwd]
            Q_true_Nm3s = torch.cat([Q_true_Nm3s_fwd, -Q_true_Nm3s_fwd], dim=0)  # [E]

            pset_bar = p_true_bar[slack_node_id]
            pset2_bar2 = p2_true_bar2[slack_node_id]

            is_slack = torch.zeros(num_nodes, dtype=torch.float32)
            is_slack[slack_node_id] = 1.0

            validate_mass_balance(s_Nm3s)

            # node features
            s_feat = torch.asinh(s_Nm3s / (q0 + 1e-12))  # [N]

            pset_feature = torch.zeros(num_nodes, dtype=torch.float32)
            pset_feature[slack_node_id] = 1.0  # marker

            node_features = [
                s_feat.unsqueeze(1),
                is_slack.unsqueeze(1),
                pset_feature.unsqueeze(1),
                topo_distance.unsqueeze(1),
            ]

            if bool(CONFIG.get("broadcast_pset", False)):
                pset2_val = float(pset2_bar2)
                pset2_log = float(np.log(pset2_val + 1e-6))
                denom = float(CONFIG.get("pset2_log_denom", np.log(1e4)))
                pset2_log_norm = float(pset2_log / (denom + 1e-12))
                pset2_global = torch.full((num_nodes, 1), pset2_log_norm, dtype=torch.float32)
                node_features.append(pset2_global)

            # append LapPE to x
            if lap_pe is not None:
                node_features.append(lap_pe)

            x = torch.cat(node_features, dim=1)

            data = GasData(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr_norm,               # normalized edge features for GNN
                edge_attr_phys=edge_attr_phys,          # physical (debug)
                s_Nm3s=s_Nm3s.unsqueeze(1),             # [N,1]
                is_slack=is_slack.unsqueeze(1),         # [N,1]
                p2_true_bar2=p2_true_bar2.unsqueeze(1), # [N,1]
                Q_true_Nm3s=Q_true_Nm3s.unsqueeze(1),    # [E,1]
                pset_bar=float(pset_bar),
                pset2_bar2=torch.tensor(float(pset2_bar2), dtype=torch.float32),  # scalar per graph
                slack_node_id=int(slack_node_id),
                scenario_id=int(scenario_index),
            )

            rev_edge_id = build_rev_edge_id(edge_index)   # [E]
            data.rev_edge_id = rev_edge_id
            data.phys_edge_mask = torch.arange(E, dtype=torch.long) < rev_edge_id

            if use_precomputed_active_mask:
                u_dir, v_dir = edge_index[0], edge_index[1]
                dp2_true = (p2_true_bar2[v_dir] - p2_true_bar2[u_dir]).view(-1)
                dp2_true_norm = torch.abs(dp2_true / c_p2)
                data.dp2_active_mask = build_active_edge_mask_single_graph(
                    dp2_true_norm,
                    q=q_act,
                    topk_min=topk_min,
                )

            data_list.append(data)

        self._validate_data_integrity(data_list)
        torch.save(self.collate(data_list), self.processed_paths[0])
        return

    def _validate_data_integrity(self, data_list):
        """Sanity checks on processed data."""
        sample = data_list[0]

        # =========================
        # CHECKPOINT C2: physical topology invariant
        # =========================
        ei = sample.edge_index
        N = sample.num_nodes

        u, v = ei
        from torch_geometric.utils import degree
        deg = degree(u, num_nodes=N)

        assert deg.max() <= 10, (
            f"Physical topology corrupted in processed data: "
            f"max_deg={deg.max().item()}"
        )

        # ensure bidirectional pairing
        rev = sample.rev_edge_id
        eid = torch.arange(ei.size(1))
        
        mask = eid != rev
        assert torch.all((eid[mask] < rev[mask]) | (eid[mask] > rev[mask])), \
            "rev_edge_id pairing inconsistent"

        assert sample.x.dim() == 2, "Node features must be 2D"
        bps = 1 if bool(CONFIG.get("broadcast_pset", False)) else 0
        k_cfg = int(CONFIG.get("lap_pe_k", 0))
        k_eff = min(k_cfg, int(sample.num_nodes) - 1) if k_cfg > 0 else 0
        expected_node_feats = 4 + bps + k_eff
        assert sample.x.size(1) == expected_node_feats, \
            f"Expected {expected_node_feats} node features, got {sample.x.size(1)}"
        assert sample.edge_attr.size(1) == 3, f"Expected 3 edge features, got {sample.edge_attr.size(1)}"
        assert sample.edge_attr_phys.size(1) == 3, f"Expected 3 physical edge features, got {sample.edge_attr_phys.size(1)}"

        for data in data_list[:10]:
            for key, tensor in data:
                if torch.is_tensor(tensor):
                    assert not torch.isnan(tensor).any(), f"NaN in tensor '{key}'"
                    assert not torch.isinf(tensor).any(), f"Inf in tensor '{key}'"

        rev = data.rev_edge_id
        E = rev.numel()
        eid = torch.arange(E, device=rev.device)
        assert torch.equal(rev[rev], eid), "rev_edge_id must satisfy rev[rev[i]] = i"
        u, v = data.edge_index[0], data.edge_index[1]
        assert torch.equal(u, v[rev]) and torch.equal(v, u[rev]), "rev_edge_id must flip endpoints"


def create_dataloaders(npz_path, norm_state, batch_size, train_ratio=0.8, shuffle_train=True):
    """Factory function for train/val dataloaders."""
    from torch_geometric.loader import DataLoader

    dataset = GasDataProcessor(npz_path, norm_state)
    split_file = CONFIG.get("split_file")
    if split_file:
        split_path = Path(split_file)
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        with np.load(split_path, allow_pickle=True) as split:
            if "train_idx" not in split or "val_idx" not in split:
                raise ValueError(f"Split file must contain train_idx and val_idx: {split_path}")
            train_idx = np.asarray(split["train_idx"], dtype=np.int64).tolist()
            val_idx = np.asarray(split["val_idx"], dtype=np.int64).tolist()
        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
    else:
        dataset = dataset.shuffle()
        train_size = int(len(dataset) * train_ratio)
        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size:]

    num_workers = int(CONFIG.get("num_workers", 0))
    loader_kwargs = {
        "batch_size": batch_size,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": bool(CONFIG.get("pin_memory", False)),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=shuffle_train,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )


    return train_loader, val_loader


