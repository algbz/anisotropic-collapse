# Anisotropic Collapse

Reproducible figure-generation code for the manuscript:

> **Fisher Collapse and Directional Concealment Near Bifurcation**  
> Aldo Alberto Aguilar Bermúdez

The project studies a model-based failure mode in which an observation channel becomes locally insensitive along a dynamically important direction near a bifurcation. The numerical examples separate latent-state fluctuations from observation-map sensitivity and illustrate directional Fisher-information loss, masking by preserved directions, alignment effects, and local sensor-selection diagnostics.

## Repository contents

```text
.
├── generate_figures.py   # self-contained generator for all manuscript figures
├── README.md
└── LICENSE               # MIT License
```

The script does not require external datasets or project-specific helper modules.

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- Matplotlib

Install the dependencies with:

```bash
python -m pip install numpy scipy matplotlib
```

A virtual environment is recommended:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then install the dependencies:

```bash
python -m pip install numpy scipy matplotlib
```

## Reproduce all figures

Clone the repository and run:

```bash
git clone https://github.com/algbz/anisotropic-collapse.git
cd anisotropic-collapse
python generate_figures.py
```

By default, the 13 PNG files are written to `figures/`.

Use a custom output directory:

```bash
python generate_figures.py --output-dir results
```

Generate one figure only:

```bash
python generate_figures.py --figure 8
```

List the available figures:

```bash
python generate_figures.py --list
```

Full generation can take several minutes. Each figure is produced in a separate Python process to keep peak memory use bounded.

## Generated outputs

| Figure | Output file | Main role |
|---:|---|---|
| 1 | `fig1_engineering_schematic.png` | Conceptual sensing and policy schematic |
| 2 | `fig2_engineering_policy.png` | Simulated sensor-response and switching illustration |
| 3 | `fig3_fisher_field.png` | Scalar potential, observation map, and Fisher field |
| 4 | `fig4_hierarchy.png` | Scalar threshold hierarchy |
| 5 | `fig5_channel.png` | Channel-strength and observed-variance dynamics |
| 6 | `fig6_scalar_occupation_v2.png` | Low-information occupation in the scalar model |
| 7 | `fig7_ablation.png` | Matched observation-map ablation |
| 8 | `fig8_channel_geometry.png` | Anisotropic channel geometry |
| 9 | `fig9_directional_collapse.png` | Coordinate-specific Fisher-information degradation |
| 10 | `fig10_alignment.png` | Alignment experiment |
| 11 | `fig11_anisotropic_occupation_v2.png` | Anisotropic low-information occupation |
| 12 | `fig12_masking.png` | Directional masking by a preserved channel |
| 13 | `fig13_diagnostics.png` | Directional, overall, and aggregate diagnostics |

## Reproducibility notes

- All numerical data are generated from the equations implemented in `generate_figures.py`.
- No external empirical data are used.
- Stochastic ensemble panels use fixed random seeds and 24 replicates.
- Comparisons across observation channels use matched latent trajectories and, where applicable, matched observation-noise draws.
- Analytic, deterministic, schematic, and simulation-based panels are identified in the manuscript captions and in the source-code comments.
- Model parameters and plotting settings are defined near the top of the script.

The simulations illustrate the behavior of the specified models and parameter choices. They are not presented as universal empirical validation.

## Citation

Until a version of record is available, please cite the manuscript and repository as:

> Aguilar Bermúdez, Aldo Alberto. *Fisher Collapse and Directional Concealment Near Bifurcation*. Manuscript, 2026. Code: `https://github.com/algbz/anisotropic-collapse`.

BibTeX:

```bibtex
@unpublished{aguilarbermudez2026fisher,
  author = {Aguilar Berm{\'u}dez, Aldo Alberto},
  title  = {Fisher Collapse and Directional Concealment Near Bifurcation},
  year   = {2026},
  note   = {Manuscript. Code available at https://github.com/algbz/anisotropic-collapse}
}
```

## Author

**Aldo Alberto Aguilar Bermúdez**  
Email: [aldgbz@pm.me](mailto:aldgbz@pm.me)

## License

The code is released under the MIT License. See `LICENSE` for the full terms.
