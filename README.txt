# Swarm-Optimized Multilevel Image Thresholding

A Python library and benchmarking suite that implements metaheuristic swarm intelligence (**Particle Swarm Optimization**) and genetic algorithms (**Genetic Algorithm**) to automate multi-level image segmentation using **Otsu’s Between-Class Variance** and **Kapur’s Entropy** fitness functions.

---

## Key Features

* **Metaheuristic Optimization:** Leverages PSO and GA to rapidly find optimal intensity thresholds without brute-force computation.
* **Dual Fitness Objectives:** Supports both variance-based (Otsu) and information-entropy (Kapur) fitness evaluation functions.
* **Image Preprocessing & Reconstruction:** Pre-smoothing with Gaussian filtering to mitigate over-segmentation caused by background noise or bokeh.
* **Evaluation Metrics:** Automatic computation of **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index) for quantitative performance comparison.
* **Visual Diagnostics:** Automated exporting of multi-level thresholded comparison grids and convergence plot histories.

---

## Repository Structure

```text
swarm-image-thresholding/
├── data/                       # Input images directory
├── outputs/
│   ├── experiment_result/      # Convergence plots & results_summary.csv
│   └── threshold_result/       # Sample segmented comparisons & plots
├── src/
│   ├── __init__.py
│   ├── fitness_functions.py    # Otsu & Kapur fitness algorithms
│   ├── image_utils.py          # Preprocessing & segmentation logic
│   ├── metrics.py              # PSNR & SSIM evaluation routines
│   └── optimizers.py           # PSO and GA implementations
├── venv/                       # Local virtual environment
├── experiments.py              # Batch benchmark execution across multiple images
├── main.py                     # Single-image demo entry-point
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation