import json
import re
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
PLOT_DIR = RESULTS_DIR / "plots"
LOG_DIR = RESULTS_DIR / "logs"

for directory in [RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR, PLOT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


NODE_FEAT_DEMAND_ASINH = 0
NODE_FEAT_IS_SLACK = 1
NODE_FEAT_SLACK_MARKER = 2
NODE_FEAT_TOPO_DIST = 3
NODE_FEAT_PSET_LOG = 4
NODE_FEAT_LAP_PE_START = 5

EDGE_FEAT_LOG_LENGTH = 0
EDGE_FEAT_LOG_DIAMETER = 1
EDGE_FEAT_LOG_ROUGHNESS = 2
EDGE_FEAT_DIM = 3

BROADCAST_PSET = True
LAP_PE_K = 4

CONSTANTS = {
    "CV_KWH_PER_NM3": 10.4839,
    "P_ATM_BAR": 1.01325,
    "TEMP_K": 283.15,
    "TEMP_NORMAL_K": 273.15,
    "R_UNIVERSAL": 8.314,
    "GAS_MOLAR_MASS": 0.01604,
    "VISCOSITY": 1.1e-5,
}

UNITS = {
    "flow_internal": "Nm3/s",
    "flow_display": "MW",
    "pressure": "bar",
    "velocity": "m/s",
    "temperature": "K",
    "length": "m",
    "diameter": "m",
    "roughness": "m",
}

VALIDATION = {
    "mass_balance_tolerance": 1e-1,
    "projection_residual_target": 1e-6,
    "min_pressure_bar": 1.0,
    "max_flow_nm3s": 500.0,
}

MASS_BALANCE_MW_TOL = VALIDATION["mass_balance_tolerance"]

CONFIG = {
    "network_name": "gaslib-582",
    "experiment_tag": "mynts",
    "train_file": RAW_DIR / "gaslib582_mynts_demo_train400.npz",
    "test_files": [RAW_DIR / "gaslib582_mynts_demo_val100.npz"],
    "split_file": None,
    "hidden_dim": 256,
    "num_layers": 6,
    "dropout": 0.05,
    "gnn_aggr": "add",
    "flow_use_edge_context": False,
    "flow_condition_on_dp2": False,
    "flow_dp2_detach": True,
    "share_edge_encoder": False,
    "final_pressure_from_flow": False,
    "final_p2_floor_bar2": 0.0,
    "resistance_mode": "physics",
    "broadcast_pset": BROADCAST_PSET,
    "pset2_log_denom": 9.21034,
    "lap_pe_k": LAP_PE_K,
    "projection_max_iter": 1000,
    "projection_tolerance": 1e-6,
    "projection_dtype": "float64",
    "use_direct_solver_cache": True,
    "pressure_cg_max_iter": 2000,
    "pressure_cg_tolerance": 1e-10,
    "pressure_laplacian_damping": 0.0,
    "pressure_solver_dtype": "float64",
    "num_workers": 0,
    "pin_memory": torch.cuda.is_available(),
    "batch_size": 32,
    "epochs": 50,
    "lr": 0.0003,
    "grad_clip_norm": 1.0,
    "patience": 35,
    "train_ratio": 0.8,
    "flow_unproj_mix_alpha": 0.5,
    "lambda_pressure_p2": 6.0,
    "lambda_dp2": 1.5,
    "lambda_flow": 4.0,
    "lambda_mono": 1.0,
    "lambda_scale": 0.0,
    "lambda_final_pipe": 0.0,
    "dp2_inactive_weight": 0.05,
    "mono_loss_mode": "softplus_sign",
    "mono_q_quantile": 0.20,
    "mono_dp2_quantile": 0.20,
    "mono_accept_tol": 7e-2,
    "mono_topk_frac": 1.0,
    "mono_topk_min": 0,
    "dp2_active_quantile": 0.95,
    "dp2_active_topk_min": 16,
    "use_precomputed_active_mask": True,
    "dp2_rel_floor": 1e-2,
    "dp2_rel_clip": 2.0,
    "final_pipe_start_epoch": 5,
    "final_pipe_end_epoch": 20,
    "final_pipe_q_quantile": 0.20,
    "final_pipe_dp_quantile": 0.20,
    "final_pipe_rel_floor": 5e-2,
    "final_pipe_rel_clip": 2.0,
    "final_pipe_re_min": 2000.0,
    "mono_start_epoch": 5,
    "scale_start_epoch": 0,
    "scale_end_epoch": 10,
    "scale_dp2_quantile": 0.20,
    "seed": 42,
    "enable_mono_correction": False,
    "mono_corr_steps": 3,
    "mono_corr_step_size": 0.15,
    "mono_corr_beta": 0.3,
    "mono_corr_q_quantile": 0.20,
    "mono_corr_dp_quantile": 0.20,
    "mono_corr_scale_floor": 1e-2,
    "mono_corr_accept_tol": 7e-2,
    "mono_corr_backtracking_steps": 4,
    "mono_corr_require_improvement": True,
    "checkpoint_path_override": None,
    "init_checkpoint_path": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def get_experiment_name(train_file: Path) -> str:
    match = re.search(r"(\d+)(?=\.npz$)", str(train_file))
    db_size = match.group(1) if match else "unknown"
    return (
        f"{CONFIG['experiment_tag']}_"
        f"{CONFIG['network_name']}_"
        f"{CONFIG['resistance_mode']}_"
        f"{db_size}"
    )


CONFIG["experiment_name"] = get_experiment_name(CONFIG["train_file"])


def save_config():
    path = LOG_DIR / f"{CONFIG['experiment_name']}_config.json"

    def serialize(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (list, tuple)):
            return [serialize(item) for item in obj]
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        return obj

    serializable = {
        "config": serialize(CONFIG),
        "units": UNITS,
        "constants": CONSTANTS,
        "validation": VALIDATION,
        "feature_indices": {
            "IDX_LENGTH": EDGE_FEAT_LOG_LENGTH,
            "IDX_DIAMETER": EDGE_FEAT_LOG_DIAMETER,
            "IDX_ROUGHNESS": EDGE_FEAT_LOG_ROUGHNESS,
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
