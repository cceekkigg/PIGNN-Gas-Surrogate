import json
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm

from src.dataset import create_dataloaders, get_or_compute_norm_state
from src.model import GasGNN
from src.config import CONFIG, CHECKPOINT_DIR, CONSTANTS, LOG_DIR, save_config
from src.utils.visualizer import plot_loss_curves


def _edge_graph_id(batch, edge_index):
    """
    Map each edge to its graph-id in a PyG Batch.
    Assumes edges do not cross graphs (true for InMemoryDataset batching).
    """
    u = edge_index[0]
    return batch.batch[u]  # [E]


def _select_active_edges_per_graph(dp2_abs, edge_gid, q=0.99, topk_min=8):
    """
    Select active edges per graph using per-graph quantile threshold on |dp2_true|.
    Fallback: ensure at least topk_min edges supervised per graph.

    Args:
        dp2_abs:  [E] absolute dp2_true (normalized)
        edge_gid: [E] graph id per edge
    Returns:
        mask: [E] boolean
    """
    device = dp2_abs.device
    E = dp2_abs.numel()
    mask = torch.zeros(E, dtype=torch.bool, device=device)
    if E == 0:
        return mask

    num_graphs = int(edge_gid.max().item()) + 1
    for g in range(num_graphs):
        idx = (edge_gid == g).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue

        vals = dp2_abs[idx]
        if idx.numel() <= topk_min:
            mask[idx] = True
            continue

        thr = torch.quantile(vals, q).item()
        m = vals >= thr

        if int(m.sum().item()) < topk_min:
            topk = torch.topk(vals, k=topk_min, largest=True).indices
            mask[idx[topk]] = True
        else:
            mask[idx[m]] = True

    return mask


def _robust_relative_l1(err_abs, denom_abs, floor=1e-2, clip=2.0):
    """
    Robust relative error:
        rel = |err| / max(|target|, floor)
        rel = clamp(rel, 0, clip)
    Returns mean(rel).
    """
    floor_t = torch.tensor(float(floor), device=denom_abs.device, dtype=denom_abs.dtype)
    denom = torch.maximum(denom_abs, floor_t)
    rel = err_abs / denom
    rel = torch.clamp(rel, 0.0, float(clip))
    return rel.mean()


def mono_weight(epoch, start=10, end=30, max_w=1.0):
    if epoch < start:
        return 0.0
    if epoch > end:
        return max_w
    return max_w * (epoch - start) / (end - start)


def _blended_friction_factor_torch(Re, eps_over_D):
    """
    Cheap differentiable Darcy-friction proxy for training-time pipe-law coupling.
    Uses laminar 64/Re, Haaland in turbulence, and a linear blend in transition.
    """
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


def final_state_pipe_loss(
    q_proj: torch.Tensor,
    p2_pred: torch.Tensor,
    edge_index: torch.Tensor,
    rev_edge_id: torch.Tensor,
    edge_attr_phys: torch.Tensor,
    c_p2: float,
    *,
    q_quantile: float = 0.20,
    dp2_quantile: float = 0.20,
    rel_floor: float = 5e-2,
    rel_clip: float = 2.0,
    re_min: float = 2000.0,
):
    """
    Final-state constitutive consistency on physical edges:
        p2[v] - p2[u] ~= dp2_from(q_proj, geometry)
    """
    u, v = edge_index[0], edge_index[1]

    E = int(q_proj.size(0))
    eid = torch.arange(E, device=q_proj.device)
    phys_mask = eid < rev_edge_id.view(-1)
    if phys_mask.sum().item() == 0:
        return q_proj.new_tensor(0.0)

    q_phys = q_proj.view(-1)[phys_mask]
    dp2_from_p = (p2_pred[v] - p2_pred[u]).view(-1)[phys_mask]

    L = edge_attr_phys[phys_mask, 0]
    D = edge_attr_phys[phys_mask, 1].clamp_min(1e-12)
    roughness = edge_attr_phys[phys_mask, 2].clamp_min(1e-12)

    bar_to_pa = 1e5
    rho_n = (
        (float(CONSTANTS["P_ATM_BAR"]) * bar_to_pa)
        * float(CONSTANTS["GAS_MOLAR_MASS"])
        / (float(CONSTANTS["R_UNIVERSAL"]) * float(CONSTANTS["TEMP_NORMAL_K"]))
    )
    r_spec = float(CONSTANTS["R_UNIVERSAL"]) / float(CONSTANTS["GAS_MOLAR_MASS"])
    mu = float(CONSTANTS["VISCOSITY"])

    q_abs = torch.abs(q_phys)
    m_dot = q_abs * rho_n
    Re = (4.0 * m_dot) / (3.141592653589793 * D * mu)
    eps_over_D = roughness / D
    friction = _blended_friction_factor_torch(Re, eps_over_D)

    coeff = (
        friction
        * 16.0
        * r_spec
        * float(CONSTANTS["TEMP_K"])
        * L
        * (rho_n ** 2)
        / ((3.141592653589793 ** 2) * (D ** 5) * (bar_to_pa ** 2))
    )
    dp2_from_q = -coeff * q_phys * q_abs

    q_abs_detached = q_abs.detach()
    dp_abs_detached = torch.maximum(
        torch.abs(dp2_from_p.detach()),
        torch.abs(dp2_from_q.detach()),
    )

    q_th = (
        torch.quantile(q_abs_detached, q_quantile)
        if q_abs_detached.numel() > 0
        else q_abs_detached.new_tensor(0.0)
    )
    dp_th = (
        torch.quantile(dp_abs_detached, dp2_quantile)
        if dp_abs_detached.numel() > 0
        else dp_abs_detached.new_tensor(0.0)
    )

    mask = (
        (q_abs_detached > q_th)
        & (dp_abs_detached > dp_th)
        & torch.isfinite(dp2_from_p)
        & torch.isfinite(dp2_from_q)
        & (Re >= re_min)
    )
    if mask.sum().item() == 0:
        return q_proj.new_tensor(0.0)

    err_abs = torch.abs((dp2_from_p - dp2_from_q)[mask]) / float(c_p2)
    denom_abs = (
        torch.maximum(torch.abs(dp2_from_p[mask]), torch.abs(dp2_from_q[mask]))
        / float(c_p2)
    )
    return _robust_relative_l1(err_abs, denom_abs, floor=rel_floor, clip=rel_clip)

def monotonicity_loss_p2(
    q_used: torch.Tensor,          # [E,1]
    p2_pred: torch.Tensor,         # [N,1]  (bar^2)
    edge_index: torch.Tensor,      # [2,E]
    rev_edge_id: torch.Tensor,     # [E]
    q_quantile: float = 0.20,
    dp2_quantile: float = 0.20,
    eps: float = 1e-2,
    loss_mode: str = "softplus_sign",
    accept_tol: float = 7e-2,
    topk_frac: float = 1.0,
    topk_min: int = 0,
):
    """
    Enforce monotonicity on physical edges:
        violation when q_{u->v} * (p_u^2 - p_v^2) < 0

    Uses differentiable penalty on the signed product.
    """
    u, v = edge_index[0], edge_index[1]
    dp2_drop = (p2_pred[u] - p2_pred[v])  # [E,1]

    E = int(q_used.size(0))
    eid = torch.arange(E, device=q_used.device)
    phys_mask = eid < rev_edge_id  # one direction per undirected pipe

    q = q_used[phys_mask].view(-1)
    dp_p2 = dp2_drop[phys_mask].view(-1)
    q_abs = q.abs()
    dp_abs = dp_p2.abs()

    # adaptive thresholds (detach to avoid backprop through quantiles)
    q_th = torch.quantile(q_abs.detach().view(-1), q_quantile) if q.numel() > 0 else q_abs.new_tensor(0.0)
    dp_th = torch.quantile(dp_abs.detach().view(-1), dp2_quantile) if dp_p2.numel() > 0 else dp_abs.new_tensor(0.0)

    mask = (q_abs > q_th) & (dp_abs > dp_th)
    if mask.sum().item() == 0:
        return q_used.new_tensor(0.0)

    if loss_mode == "tol_excess":
        p_bar = torch.sqrt(torch.clamp(p2_pred.view(-1), min=0.0))
        dp_bar = (p_bar[u] - p_bar[v]).view(-1)[phys_mask]
        dp_bar_m = dp_bar[mask]
        prod_m = (q[mask] * dp_bar_m)
        excess = torch.relu(-(prod_m + float(accept_tol)))
        if excess.numel() == 0:
            return q_used.new_tensor(0.0)
        if topk_frac < 1.0 or topk_min > 0:
            k = max(int(topk_min), int(torch.ceil(torch.tensor(float(topk_frac) * excess.numel())).item()))
            k = max(1, min(k, int(excess.numel())))
            excess = torch.topk(excess, k=k, largest=True).values
        return excess.mean()

    prod = (q * dp_p2)  # want prod >= 0
    prod_m = prod[mask]

    # scale to make argument ~O(1)
    scale = (q_abs[mask] * dp_abs[mask]).median().detach()
    scale = scale.clamp_min(eps)   # prevents huge gradients early
    z = prod_m / scale  # dimensionless signed product

    # penalize negative z (smooth hinge)
    return F.softplus(-z).mean()

def run_epoch(
    model,
    data_loader,
    optimizer,
    device,
    c_p2: float,
    q0: float,
    epoch: int, 
    is_trainable_model: bool = True,
):
    """
    Edge-first training epoch.

    Loss:
      L = lam_dp2 * L_dp2_active + lam_p2 * L_p2_nodes + lam_q * L_flow

    - L_dp2_active: robust relative L1 on active edges (dp2_hat vs dp2_true)
    - L_p2_nodes:   L1 on free nodes of integrated p2 (delta-to-pset2, normalized)
    - L_flow:       original mixed unproj/proj in asinh space
    """
    is_train = (optimizer is not None) and is_trainable_model
    model.train(is_train)
    if not is_train:
        model.eval()

    # weights
    lam_p2 = float(CONFIG.get("lambda_pressure_p2", 1.0))
    lam_dp2 = float(CONFIG.get("lambda_dp2", 1.0))
    lam_q = float(CONFIG.get("lambda_flow", 1.0))
    # lam_mono = float(CONFIG.get("lambda_mono", 0.1))

    # flow mix coefficient (same semantics as your original train.py)
    alpha = float(CONFIG.get("flow_unproj_mix_alpha", 1.0))
    alpha = max(0.0, min(1.0, alpha))

    # active edge selection + robust relative loss params
    q_act = float(CONFIG.get("dp2_active_quantile", 0.99))
    topk_min = int(CONFIG.get("dp2_active_topk_min", 8))
    rel_floor = float(CONFIG.get("dp2_rel_floor", 1e-2))
    rel_clip = float(CONFIG.get("dp2_rel_clip", 2.0))

    total = {
        "loss": 0.0,
        "p2": 0.0,
        "dp2": 0.0,
        "q": 0.0,
        "pipe": 0.0,
    }
    num_graphs_seen = 0
    batch_to_kwargs = {"non_blocking": True} if device.type == "cuda" else {}
    grad_context = torch.enable_grad() if is_train else torch.inference_mode()

    with grad_context:
        for batch in data_loader:
            batch = batch.to(device, **batch_to_kwargs)
            
            # model must return dp2_hat and q_unproj for losses
            p2_pred, q_proj, q_unproj, dp2_hat = model(
                batch, return_unprojected=True, return_dp2=True
            )

            # dp2 target from p2_true
            u, v = batch.edge_index[0], batch.edge_index[1]
            p2_true = batch.p2_true_bar2  # [N,1]
            dp2_true = (p2_true[v] - p2_true[u]).view(-1)  # [E]
            dp2_hat = dp2_hat.view(-1)                     # [E]

            E = int(batch.edge_index.size(1))
            if hasattr(batch, "phys_edge_mask"):
                phys_mask = batch.phys_edge_mask.view(-1).bool()
            else:
                eid = torch.arange(E, device=batch.edge_index.device)
                phys_mask = eid < batch.rev_edge_id.view(-1)

            # normalized dp2
            dp2_true_norm = dp2_true / float(c_p2)
            dp2_hat_norm = dp2_hat / float(c_p2)

            if bool(CONFIG.get("use_precomputed_active_mask", True)) and hasattr(batch, "dp2_active_mask"):
                active_mask = batch.dp2_active_mask.view(-1).bool()
            else:
                edge_gid = _edge_graph_id(batch, batch.edge_index)  # [E]
                dp2_abs = torch.abs(dp2_true_norm)
                active_mask = _select_active_edges_per_graph(
                    dp2_abs, edge_gid, q=q_act, topk_min=topk_min
                )

            if active_mask.any():
                err = torch.abs(dp2_hat_norm[active_mask] - dp2_true_norm[active_mask])
                denom = torch.abs(dp2_true_norm[active_mask])
                dp2_loss = _robust_relative_l1(err, denom, floor=rel_floor, clip=rel_clip)
            else:
                dp2_loss = F.l1_loss(dp2_hat_norm, dp2_true_norm)

            inactive_w = float(CONFIG.get("dp2_inactive_weight", 0.05))
            inactive_mask = (~active_mask) & phys_mask
            if inactive_w > 0.0 and inactive_mask.any():
                dp2_inactive = F.smooth_l1_loss(
                    dp2_hat_norm[inactive_mask],
                    dp2_true_norm[inactive_mask],
                    beta=0.5
                )
                dp2_loss = dp2_loss + inactive_w * dp2_inactive

            # node p2 loss (after integration). compare delta-to-pset2, normalized, free nodes only
            pset2_per_node = batch.pset2_bar2[batch.batch].unsqueeze(1)  # [N,1]

            p2_pred_norm = (p2_pred - pset2_per_node) / float(c_p2)
            p2_true_norm = (batch.p2_true_bar2 - pset2_per_node) / float(c_p2)

            slack_mask = batch.is_slack.view(-1).bool()
            free_mask = ~slack_mask
            if free_mask.any():
                p2_loss = F.l1_loss(p2_pred_norm[free_mask], p2_true_norm[free_mask])
            else:
                p2_loss = F.l1_loss(p2_pred_norm, p2_true_norm)

            # ------ flow loss -------
            q_true = batch.Q_true_Nm3s  # [E,1]

            def asinh_norm(q):
                return torch.asinh(q / (float(q0) + 1e-12))

            q_true_t = asinh_norm(q_true)
            q_unproj_t = asinh_norm(q_unproj)
            q_proj_t = asinh_norm(q_proj)

            q_loss_unproj = F.l1_loss(q_unproj_t, q_true_t)
            q_loss_proj = F.l1_loss(q_proj_t, q_true_t)
            q_loss = alpha * q_loss_unproj + (1.0 - alpha) * q_loss_proj

            scale_start = int(CONFIG.get("scale_start_epoch", 0))
            if epoch >= scale_start:
                t_eps = 1e-8
                q_true_e = q_true.view(-1)
                RHO_STD = 1.01 * 1e5 * CONSTANTS["GAS_MOLAR_MASS"] / (CONSTANTS["R_UNIVERSAL"] * CONSTANTS["TEMP_NORMAL_K"])
                m_dot_true = torch.abs(q_true_e) * RHO_STD
                D = batch.edge_attr_phys[:, 1]
                Re_true = (4.0 * m_dot_true) / (3.14159 * D * CONSTANTS["VISCOSITY"])
                dp2_pred_from_p2 = (p2_pred[v] - p2_pred[u]).view(-1) / float(c_p2)
                q_abs = torch.abs(q_true_e)
                dp_abs = torch.abs(dp2_true_norm)
                k = max(1, int(0.20 * q_abs.numel()))
                q_th = q_abs.detach().kthvalue(k).values
                dp_th = torch.quantile(dp_abs.detach(), float(CONFIG.get("scale_dp2_quantile", 0.20)))
                mask = phys_mask & (q_abs > q_th) & (dp_abs > dp_th) & (Re_true > 4000)
                if mask.any():
                    log_dp2_pred = torch.log(torch.abs(dp2_pred_from_p2[mask]) + t_eps)
                    log_dp2_true = torch.log(torch.abs(dp2_true_norm[mask]) + t_eps)
                    log_q_hat = torch.log(torch.abs(q_proj.view(-1)[mask]) + t_eps)
                    log_q_true = torch.log(torch.abs(q_true.view(-1)[mask]) + t_eps)
                    scale_loss = F.l1_loss(
                        log_dp2_pred - 2.0 * log_q_hat,
                        log_dp2_true - 2.0 * log_q_true
                    )
                else:
                    scale_loss = dp2_loss * 0.0
            else:
                scale_loss = dp2_loss * 0.0
            

            # ---- monotonicity loss ----
            mono_start = int(CONFIG.get("mono_start_epoch", 0))
            if epoch >= mono_start:
                mono_loss = monotonicity_loss_p2(
                    q_proj, p2_pred, batch.edge_index, batch.rev_edge_id,
                    q_quantile=float(CONFIG.get("mono_q_quantile", 0.20)),
                    dp2_quantile=float(CONFIG.get("mono_dp2_quantile", 0.20)),
                    loss_mode=str(CONFIG.get("mono_loss_mode", "softplus_sign")),
                    accept_tol=float(CONFIG.get("mono_accept_tol", 7e-2)),
                    topk_frac=float(CONFIG.get("mono_topk_frac", 1.0)),
                    topk_min=int(CONFIG.get("mono_topk_min", 0)),
                )
            else:
                mono_loss = dp2_loss * 0.0
            lam_mono = mono_weight(
                epoch,
                start=mono_start,
                end=max(mono_start + 1, mono_start + 10),
                max_w=CONFIG["lambda_mono"],
            )

            pipe_loss = final_state_pipe_loss(
                q_proj=q_proj,
                p2_pred=p2_pred,
                edge_index=batch.edge_index,
                rev_edge_id=batch.rev_edge_id,
                edge_attr_phys=batch.edge_attr_phys,
                c_p2=c_p2,
                q_quantile=float(CONFIG.get("final_pipe_q_quantile", 0.20)),
                dp2_quantile=float(CONFIG.get("final_pipe_dp_quantile", 0.20)),
                rel_floor=float(CONFIG.get("final_pipe_rel_floor", 5e-2)),
                rel_clip=float(CONFIG.get("final_pipe_rel_clip", 2.0)),
                re_min=float(CONFIG.get("final_pipe_re_min", 2000.0)),
            )
            lam_pipe = mono_weight(
                epoch,
                start=int(CONFIG.get("final_pipe_start_epoch", 5)),
                end=int(CONFIG.get("final_pipe_end_epoch", 20)),
                max_w=float(CONFIG.get("lambda_final_pipe", 0.2)),
            )

            # ----- moody loss ------
            lam_scale = mono_weight(
                epoch,
                start=int(CONFIG.get("scale_start_epoch", 0)),
                end=int(CONFIG.get("scale_end_epoch", 10)),
                max_w=float(CONFIG.get("lambda_scale", 0.05)),
            )

            # ----- total loss ------
            loss = (
                lam_dp2 * dp2_loss
                + lam_p2 * p2_loss
                + lam_q * q_loss
                + lam_mono * mono_loss
                + lam_pipe * pipe_loss
                + lam_scale * scale_loss
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_clip = float(CONFIG.get("grad_clip_norm", 0.0))
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            bs = int(batch.num_graphs)
            total["loss"] += float(loss.item()) * bs
            total["p2"] += float(p2_loss.item()) * bs
            total["dp2"] += float(dp2_loss.item()) * bs
            total["q"] += float(q_loss.item()) * bs
            total["pipe"] += float(pipe_loss.item()) * bs
            num_graphs_seen += bs

    denom = max(1, num_graphs_seen)
    return (
        total["loss"] / denom,
        total["p2"] / denom,
        total["dp2"] / denom,
        total["q"] / denom,
        total["pipe"] / denom,
    )


def main():
    device = torch.device(CONFIG["device"])
    torch.manual_seed(int(CONFIG.get("seed", 42)))
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    norm_state = get_or_compute_norm_state(CONFIG["train_file"])
    c_p2 = float(norm_state["c_p2"])
    q0 = float(norm_state["q0"])

    train_loader, val_loader = create_dataloaders(
        npz_path=CONFIG["train_file"],
        norm_state=norm_state,
        batch_size=int(CONFIG.get("batch_size", 64)),
        train_ratio=float(CONFIG.get("train_ratio", 0.8)),
        shuffle_train=True,
    )

    sample = next(iter(train_loader))

    model = GasGNN(
        node_in_dim=sample.x.shape[-1],
        hidden_dim=int(CONFIG.get("hidden_dim", 256)),
        num_layers=int(CONFIG.get("num_layers", 6)),
    ).to(device)

    init_checkpoint_path = CONFIG.get("init_checkpoint_path")
    if init_checkpoint_path:
        init_ckpt = Path(init_checkpoint_path)
        if not init_ckpt.exists():
            raise FileNotFoundError(f"Initial checkpoint not found: {init_ckpt}")
        state = torch.load(init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    has_trainable_params = len(trainable_params) > 0

    if has_trainable_params:
        optimizer = torch.optim.Adam(trainable_params, lr=float(CONFIG.get("lr", 1e-3)))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.5, patience=max(1, int(CONFIG.get("patience", 50)) // 2)
        )
    else:
        optimizer = None
        scheduler = None

    exp_name = CONFIG["experiment_name"]
    save_config()
    best_val = float("inf")
    best_epoch = -1
    bad = 0
    patience = int(CONFIG.get("patience", 50))
    num_epochs = int(CONFIG.get("epochs", 2))

    history = {
        "train_loss": [], "val_loss": [],
        "train_pressure_loss": [], "val_pressure_loss": [],
        "train_dp2": [], "val_dp2": [],
        "train_flow_loss": [], "val_flow_loss": [],
        "train_pipe": [], "val_pipe": [],
    }
    epoch_bar = tqdm(range(num_epochs), desc="Training epochs", unit="epoch")
    for epoch in epoch_bar:
        tr_loss, tr_p2, tr_dp2, tr_q, tr_pipe = run_epoch(
            model, train_loader, optimizer, device, c_p2=c_p2, q0=q0, epoch=epoch,
            is_trainable_model=has_trainable_params
        )
        va_loss, va_p2, va_dp2, va_q, va_pipe = run_epoch(
            model, val_loader, None, device, c_p2=c_p2, q0=q0, epoch=epoch,
            is_trainable_model=has_trainable_params
        )

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_pressure_loss"].append(tr_p2)
        history["val_pressure_loss"].append(va_p2)
        history["train_dp2"].append(tr_dp2)
        history["val_dp2"].append(va_dp2)
        history["train_flow_loss"].append(tr_q)
        history["val_flow_loss"].append(va_q)
        history["train_pipe"].append(tr_pipe)
        history["val_pipe"].append(va_pipe)

        epoch_bar.set_postfix({
            "train_loss": f"{tr_loss:.4f}",
            "val_loss": f"{va_loss:.4f}",
            "best_val": f"{best_val:.4f}" if best_val != float("inf") else "inf",
        })

        if scheduler is not None:
            scheduler.step(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            bad = 0
            ckpt_path = CHECKPOINT_DIR / f"{exp_name}_best.pt"
            torch.save({"model": model.state_dict(), "config": CONFIG}, ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch+1}. best_val={best_val:.6f} @ epoch {best_epoch+1}.")
                break

    try:
        plot_loss_curves(history, exp_name)
    except Exception as e:
        print(f"[WARN] plot_loss_curves failed: {e}")

    try:
        history_path = LOG_DIR / f"{exp_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save history: {e}")

    return


if __name__ == "__main__":
    main()
