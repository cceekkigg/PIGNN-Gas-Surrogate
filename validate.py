import json
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.config import CHECKPOINT_DIR, CONFIG, LOG_DIR, save_config
from src.dataset import GasDataProcessor, get_or_compute_norm_state
from src.model import GasGNN
from src.projection import MonotonicityCorrection
from src.utils.physics import compute_edge_monotonicity, compute_nodal_mass_balance
from src.utils.visualizer import plot_validation_parity


def r_squared(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
    if mask.sum() == 0:
        return 0.0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return float((1 - ss_res / (ss_tot + 1e-8)).item())


def summarize_distribution(values) -> dict[str, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def build_model(device: torch.device, checkpoint_path: Path) -> GasGNN:
    norm_state = get_or_compute_norm_state(CONFIG["train_file"])
    dataset = GasDataProcessor(CONFIG["train_file"], norm_state)
    sample = dataset[0]
    model = GasGNN(
        node_in_dim=sample.x.size(1),
        hidden_dim=int(CONFIG["hidden_dim"]),
        num_layers=int(CONFIG["num_layers"]),
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def build_mono_corrector(model: GasGNN) -> MonotonicityCorrection | None:
    if not CONFIG.get("enable_mono_correction", False):
        return None
    return MonotonicityCorrection(
        projection=model.projection,
        steps=int(CONFIG["mono_corr_steps"]),
        step_size=float(CONFIG["mono_corr_step_size"]),
        beta=float(CONFIG["mono_corr_beta"]),
        q_quantile=float(CONFIG["mono_corr_q_quantile"]),
        dp_quantile=float(CONFIG["mono_corr_dp_quantile"]),
        scale_floor=float(CONFIG["mono_corr_scale_floor"]),
        accept_tol=float(CONFIG.get("mono_corr_accept_tol", 7e-2)),
        backtracking_steps=int(CONFIG.get("mono_corr_backtracking_steps", 4)),
        require_improvement=bool(CONFIG.get("mono_corr_require_improvement", True)),
    )


def evaluate_file(
    model: GasGNN,
    mono_corrector: MonotonicityCorrection | None,
    norm_state: dict,
    test_npz: Path,
    device: torch.device,
) -> tuple[dict, dict]:
    dataset = GasDataProcessor(test_npz, norm_state)
    loader = DataLoader(dataset, batch_size=int(CONFIG["batch_size"]), shuffle=False)

    p2_true_all, p2_pred_all = [], []
    q_true_all, q_proj_all, q_unproj_all = [], [], []
    demand_all, pset2_all, scenario_id_all = [], [], []

    pressure_mae_all = []
    flow_mae_all = []
    mass_balance_pre_max_all, mass_balance_post_max_all = [], []
    mass_balance_pre_l2_all, mass_balance_post_l2_all = [], []
    mono_pre_all, mono_post_all = [], []
    runtime_total_all, runtime_forward_all, runtime_mono_all = [], [], []

    total_graphs = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            t0 = time.time()
            p2_pred, q_proj, q_unproj = model(batch, return_unprojected=True)
            t1 = time.time()

            if mono_corrector is not None:
                q_proj = mono_corrector(
                    q_proj,
                    p2_pred,
                    batch.edge_index,
                    batch.rev_edge_id,
                    batch.s_Nm3s,
                    batch.is_slack,
                )
            t2 = time.time()

            batch_list = batch.to_data_list()
            node_cursor = 0
            edge_cursor = 0

            for graph_data in batch_list:
                n = graph_data.num_nodes
                e = graph_data.num_edges

                p2_pred_graph = p2_pred[node_cursor : node_cursor + n]
                q_proj_graph = q_proj[edge_cursor : edge_cursor + e]
                q_unproj_graph = q_unproj[edge_cursor : edge_cursor + e]
                node_cursor += n
                edge_cursor += e
                total_graphs += 1

                rev = graph_data.rev_edge_id
                phys_mask = torch.arange(rev.size(0), device=device) < rev

                q_true_phys = graph_data.Q_true_Nm3s[phys_mask]
                q_proj_phys = q_proj_graph[phys_mask]
                q_unproj_phys = q_unproj_graph[phys_mask]
                p2_true_graph = graph_data.p2_true_bar2

                p2_true_all.append(p2_true_graph.cpu())
                p2_pred_all.append(p2_pred_graph.cpu())
                q_true_all.append(q_true_phys.cpu())
                q_proj_all.append(q_proj_phys.cpu())
                q_unproj_all.append(q_unproj_phys.cpu())
                demand_all.append(graph_data.s_Nm3s.view(-1).cpu())
                pset2_all.append(graph_data.pset2_bar2.view(1).cpu())
                scenario_id_all.append(torch.tensor([int(graph_data.scenario_id)], dtype=torch.long))

                q_true_np = q_true_phys.cpu().numpy().reshape(-1)
                q_proj_np = q_proj_phys.cpu().numpy().reshape(-1)
                q_unproj_np = q_unproj_phys.cpu().numpy().reshape(-1)
                p_true_np = torch.sqrt(torch.clamp(p2_true_graph, min=0.0)).cpu().numpy().reshape(-1)
                p_pred_np = torch.sqrt(torch.clamp(p2_pred_graph, min=0.0)).cpu().numpy().reshape(-1)

                _, mb_pre_max, mb_pre_l2 = compute_nodal_mass_balance(graph_data, q_unproj_np)
                _, mb_post_max, mb_post_l2 = compute_nodal_mass_balance(graph_data, q_proj_np)
                _, mono_pre = compute_edge_monotonicity(graph_data, p_pred_np, q_unproj_np)
                _, mono_post = compute_edge_monotonicity(graph_data, p_pred_np, q_proj_np)

                pressure_mae_all.append(float(np.mean(np.abs(p_true_np - p_pred_np))))
                flow_mae_all.append(float(np.mean(np.abs(q_true_np - q_proj_np))))
                mass_balance_pre_max_all.append(float(mb_pre_max))
                mass_balance_post_max_all.append(float(mb_post_max))
                mass_balance_pre_l2_all.append(float(mb_pre_l2))
                mass_balance_post_l2_all.append(float(mb_post_l2))
                mono_pre_all.append(float(mono_pre))
                mono_post_all.append(float(mono_post))

                batch_graphs = max(1, len(batch_list))
                runtime_total_all.append(float((t2 - t0) / batch_graphs))
                runtime_forward_all.append(float((t1 - t0) / batch_graphs))
                runtime_mono_all.append(float((t2 - t1) / batch_graphs))

    p2_true_cat = torch.cat(p2_true_all)
    p2_pred_cat = torch.cat(p2_pred_all)
    p_true_cat = torch.sqrt(torch.clamp(p2_true_cat, min=0.0))
    p_pred_cat = torch.sqrt(torch.clamp(p2_pred_cat, min=0.0))
    q_true_cat = torch.cat(q_true_all)
    q_proj_cat = torch.cat(q_proj_all)

    metrics = {
        "pressure_mae_bar": float(torch.mean(torch.abs(p_true_cat - p_pred_cat))),
        "pressure_r2": r_squared(p_true_cat, p_pred_cat),
        "flow_mae_nm3s": float(torch.mean(torch.abs(q_true_cat - q_proj_cat))),
        "flow_r2": r_squared(q_true_cat, q_proj_cat),
        "pressure_mae_distribution": summarize_distribution(pressure_mae_all),
        "flow_mae_distribution": summarize_distribution(flow_mae_all),
        "mass_balance_pre_max": summarize_distribution(mass_balance_pre_max_all),
        "mass_balance_post_max": summarize_distribution(mass_balance_post_max_all),
        "mass_balance_pre_l2": summarize_distribution(mass_balance_pre_l2_all),
        "mass_balance_post_l2": summarize_distribution(mass_balance_post_l2_all),
        "monotonicity_pre": summarize_distribution(mono_pre_all),
        "monotonicity_post": summarize_distribution(mono_post_all),
        "runtime_total_sec": summarize_distribution(runtime_total_all),
        "runtime_forward_sec": summarize_distribution(runtime_forward_all),
        "runtime_mono_sec": summarize_distribution(runtime_mono_all),
        "num_graphs": int(total_graphs),
    }

    arrays = {
        "p2_val_true_bar2": torch.stack(p2_true_all).numpy(),
        "p2_val_pred_bar2": torch.stack(p2_pred_all).numpy(),
        "q_val_true_Nm3s": torch.stack(q_true_all).numpy(),
        "q_val_pred_Nm3s": torch.stack(q_proj_all).numpy(),
        "q_val_pred_unproj_Nm3s": torch.stack(q_unproj_all).numpy(),
        "pset2_val_bar2": torch.cat(pset2_all).numpy(),
        "demand_val_Nm3s": torch.stack(demand_all).numpy(),
        "scenario_id_val": torch.cat(scenario_id_all).numpy(),
        "mass_balance_pre_max": np.asarray(mass_balance_pre_max_all, dtype=np.float64),
        "mass_balance_post_max": np.asarray(mass_balance_post_max_all, dtype=np.float64),
        "mass_balance_pre_l2": np.asarray(mass_balance_pre_l2_all, dtype=np.float64),
        "mass_balance_post_l2": np.asarray(mass_balance_post_l2_all, dtype=np.float64),
        "monotonicity_violation_rate_pre": np.asarray(mono_pre_all, dtype=np.float64),
        "monotonicity_violation_rate_post": np.asarray(mono_post_all, dtype=np.float64),
    }
    return metrics, arrays


def main():
    device = torch.device(CONFIG["device"])
    exp_name = CONFIG["experiment_name"]
    checkpoint_override = CONFIG.get("checkpoint_path_override")
    checkpoint_path = Path(checkpoint_override) if checkpoint_override else (CHECKPOINT_DIR / f"{exp_name}_best.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    norm_state = get_or_compute_norm_state(CONFIG["train_file"])
    model = build_model(device, checkpoint_path)
    mono_corrector = build_mono_corrector(model)
    save_config()

    for test_npz in CONFIG["test_files"]:
        test_npz = Path(test_npz)
        file_suffix = test_npz.stem
        metrics, arrays = evaluate_file(model, mono_corrector, norm_state, test_npz, device)

        metrics_path = LOG_DIR / f"{exp_name}_{file_suffix}_metrics.json"
        predictions_path = LOG_DIR / f"{exp_name}_{file_suffix}_predictions.npz"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        np.savez_compressed(predictions_path, **arrays)
        plot_validation_parity(
            p_true=np.sqrt(np.clip(arrays["p2_val_true_bar2"].reshape(-1), a_min=0.0, a_max=None)),
            p_pred=np.sqrt(np.clip(arrays["p2_val_pred_bar2"].reshape(-1), a_min=0.0, a_max=None)),
            q_true=arrays["q_val_true_Nm3s"].reshape(-1),
            q_pred=arrays["q_val_pred_Nm3s"].reshape(-1),
            exp_name=exp_name,
        )

        print(
            f"{test_npz.name}: pressure_mae={metrics['pressure_mae_bar']:.4f} bar, "
            f"flow_mae={metrics['flow_mae_nm3s']:.4f} Nm3/s"
        )


if __name__ == "__main__":
    main()
