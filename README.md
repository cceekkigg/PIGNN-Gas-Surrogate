# Physics-Informed Graph Neural Surrogate for Gas Networks

This repository contains a cleaned implementation of a physics-informed graph neural network surrogate for steady-state gas-network simulation and feasibility-oriented evaluation. From this package, users can reproduce the included GasLib-582 demonstration training and validation workflow.

## Repository contents

```text
.
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- requirements.txt
|-- src/
|   |-- config.py
|   |-- dataset.py
|   |-- model.py
|   |-- projection.py
|   |-- train.py
|   |-- validate.py
|   `-- utils/
|       |-- physics.py
|       `-- visualizer.py
|-- data/
|   |-- gaslib-582.gml
|   `-- raw/
```

The `src/` directory contains the training, validation, graph construction, model, and physics-operator code. The `data/` directory contains the GasLib-582 topology and bundled demonstration arrays. 

## Installation

Use a dedicated Python environment. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux or macOS, activate the environment with the corresponding shell command:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dependency file lists `numpy`, `pandas`, `matplotlib`, `networkx`, `tqdm`, `torch`, `torch-geometric`, and `scikit-learn`. For GPU use, install PyTorch and PyTorch Geometric builds compatible with the local CUDA environment before running the package.

## Data

The repository includes the raw GasLib-582 topology file:

- `data/gaslib-582.gml`

The repository includes labelled MYNTS-demonstration data used by the packaged experiment:

- `data/raw/gaslib582_mynts_demo_train400.npz`: 400 labelled scenarios
- `data/raw/gaslib582_mynts_demo_val100.npz`: 100 labelled scenarios

The `.npz` files contain arrays for node identifiers, node names, nodal demand, nodal pressure, average edge pressure, edge flow, node static data, and edge static data. The model code uses the `demand`, `pressure`, `flow`, and `edge_static` arrays during graph construction.

The original MYNTS scenario-generation workflow is not included in this release. No MYNTS execution script or interface is provided. To use additional data, place compatible `.npz` files under `data/raw/` and update `src/config.py`.

## Usage

Run commands from the repository root.

### Train model

```bash
python -m src.train
```

This command reads the training file configured in `src/config.py`, constructs or loads processed graph caches, trains `GasGNN`, and writes outputs under `results/`. With the default configuration, the command uses `data/raw/gaslib582_mynts_demo_train400.npz`, applies the internal `train_ratio=0.8`, and saves:

- a checkpoint in `results/checkpoints/`
- a configuration file and training history in `results/logs/`
- loss curves in `results/plots/<experiment_name>/`

The default training configuration is intended for the bundled GasLib-582 demonstration package.

### Evaluate model

```bash
python -m src.validate
```

This command loads the checkpoint specified by the current configuration. By default, it expects:

- `results/checkpoints/mynts_gaslib-582_physics_400_best.pt`
- `data/raw/gaslib582_mynts_demo_val100.npz`

The validation script writes:

- metrics JSON under `results/logs/`
- prediction arrays under `results/logs/`
- a pressure/flow parity plot under `results/plots/<experiment_name>/`


### Generate data

The manuscript uses MYNTS to generate labelled steady-state gas-network simulation data. The cleaned package does not include scripts or an interface for MYNTS-based scenario generation.

Users may generate additional training or validation data with MYNTS or another steady-state gas-network simulator, provided the resulting files are converted to the `.npz` schema expected by `src/dataset.py`. At minimum, compatible files should include:

- `demand`: scenario-wise nodal demand/injection array with shape `[num_scenarios, num_nodes]`
- `pressure`: scenario-wise nodal pressure array with shape `[num_scenarios, num_nodes]`
- `flow`: scenario-wise physical-edge flow array with shape `[num_scenarios, num_edges]`
- `edge_static`: physical-edge attributes containing source node, target node, length, diameter, and roughness

Place compatible files under `data/raw/` and update `train_file` and `test_files` in `src/config.py`.


## Configuration

Model, data, and training settings are configured in `src/config.py`. Important default settings include:

- `network_name`: `gaslib-582`
- `train_file`: `data/raw/gaslib582_mynts_demo_train400.npz`
- `test_files`: `data/raw/gaslib582_mynts_demo_val100.npz`
- `hidden_dim`: 256
- `num_layers`: 6
- `dropout`: 0.05
- `batch_size`: 32
- `epochs`: 50
- `lr`: 0.0003
- `patience`: 35
- `train_ratio`: 0.8
- `seed`: 42

The physics-operator tolerances, pressure reconstruction settings, mass-balance projection settings, loss weights, and feature-normalization controls are also defined in `src/config.py`.


## Citation

If you use this code, cite the associated manuscript and this repository. The repository archive DOI is [10.5281/zenodo.20075820](https://doi.org/10.5281/zenodo.20075820). A `CITATION.cff` file is included for GitHub citation metadata.

```bibtex
@software{jiang_gasgnn_2026,
  author    = {Jiang, Dongrui and Garcke, Jochen and Akca, Okan and Hollnagel, Jeremias and Klaassen, Bernhard and Anvari, Mehrnaz and M{\"u}ller-Kirchenbauer, Joachim},
  title     = {Physics-Informed Graph Neural Surrogate for Steady-State Gas Network Simulation and Feasibility Screening: Code and Reproducibility Package},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20075820},
  url       = {https://doi.org/10.5281/zenodo.20075820}
}
```

## License and Terms of Use

This project is released under the [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/). See `LICENSE` for the repository terms.

**Academic Use:** You are free to use, modify, and distribute this software for non-commercial academic research purposes, provided that you properly cite our paper.

**Commercial Use:** Commercial use of this software, including but not limited to integrating it into commercial products, using it for internal company operations, or offering it as a paid service, is strictly prohibited without explicit written permission.

## Contact

jdr_maggiea@hotmail.com
