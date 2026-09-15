# Swarm-Optimized Multilevel Image Thresholding

## Overview

<img width="1400" height="1000" alt="sample_comparison" src="https://github.com/user-attachments/assets/053beac7-9fb2-4e0f-a057-f08fd8322f95" />


TThis project is about **image segmentation using optimization techniques**.

The goal is to take a **grayscale image**, find the best **threshold values**, and divide the image into clear regions using **Particle Swarm Optimization (PSO)** and **Genetic Algorithm (GA)**.

Both PSO and GA are implemented **from scratch**, without using ready-made optimization libraries.

The project then compares both methods based on **image quality, accuracy, and how quickly they find the best solution**.

---

## Tools Used

* Python
* NumPy, SciPy
* OpenCV
* Matplotlib
* Particle Swarm Optimization (from scratch)
* Genetic Algorithm (from scratch)

---

### Optimization Pipeline Architecture

                    ┌─────────────────────┐
                    │    Raw Grayscale    │
                    │        Image        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Image Histogram   │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │     Fitness Function     │
                 │                          │
                 │    • Otsu Variance       │
                 │    • Kapur Entropy       │
                 └────────────┬─────────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
       ┌──────────────────┐      ┌──────────────────┐
       │       PSO        │      │        GA        │
       │  From Scratch    │      │   From Scratch   │
       └────────┬─────────┘      └────────┬─────────┘
                │                         │
                └────────────┬────────────┘
                             ▼
                  ┌──────────────────────┐
                  │ Optimal Threshold Set│
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Segmented Image      │
                  └──────────┬───────────┘
                             │
                             ▼
             ┌───────────────────────────────┐
             │ Quantitative Evaluation      │
             │                               │
             │ • PSNR                       │
             │ • SSIM                       │
             │ • Convergence                │
             │ • Sensitivity Analysis       │
             └───────────────────────────────┘




---



## Features

* From-scratch **PSO** and **GA** implementations (no external optimization libraries)
* **Otsu's variance** and **Kapur's entropy** fitness functions
* Multilevel thresholding across configurable threshold counts
* Convergence tracking with plateau annotations
* PSNR / SSIM quantitative image-quality scoring
* Swarm/population size sensitivity analysis
* Quantitative metrics dashboard

---

## Dataset
<img width="485" height="266" alt="sample" src="https://github.com/user-attachments/assets/80d4d20d-e5f1-467d-a35f-a0884c598b22" />


The pipeline runs on standard grayscale test images commonly used in segmentation research, including:

* 1 standard benchmark test images `sample.png`
* Any custom grayscale image (via the `data/` folder)



---

## Analysis Performed

* **PSO vs. GA Convergence Analysis**
* **Threshold Quality Scoring (PSNR / SSIM)**
* **Swarm/Population Size Sensitivity Analysis**
* **Fitness Landscape Comparison (Otsu vs. Kapur)**

### Key Finding

PSO and GA reach comparable final segmentation quality — up to **43.98 dB PSNR** and **0.97 SSIM** — but **PSO converges in 12–17 iterations**, compared to **20–44 iterations for GA**, on the same fitness landscape.

---

## Project Execution
<img width="1000" height="400" alt="sample_convergence" src="https://github.com/user-attachments/assets/6dde9946-1241-4b52-a44d-3b8f8225be11" />


This section shows the actual execution of the optimization pipeline, where threshold overlays, convergence curves, and the metrics dashboard are automatically generated after running the script.




## Key Insights

* PSO reaches near-optimal thresholds significantly faster than GA
* Otsu and Kapur fitness functions produce comparable final image quality
* Swarm/population size directly trades off accuracy against compute cost
* From-scratch implementation confirms understanding of the underlying optimization math, not just library usage

---

## Project Structure

```bash
swarm-optimized-thresholding/
│── data/
│   ├── sample.png
│
│── outputs/
│   ├── figures/
│   │   ├── threshold_overlay.png
│   │   ├── convergence_curves.png
│   │   └── metrics_dashboard.png
│   └── benchmark_results.csv
│
│── src/
│   ├── config.py
│   ├── fitness.py
│   ├── optimizers.py
│   ├── thresholding.py
│   └── evaluate.py
│
│── tests/
│── requirements.txt
│── main.py
│── README.md
```

---

## How to Use

1. Install dependencies from `requirements.txt` by writing in terminal `pip install -r requirements.txt`.
2. Place grayscale test images in `data/`
3. Run `python main.py` to optimize thresholds and generate reports
4. Check `outputs/figures/` for visualizations and `outputs/benchmark_results.csv` for scores

---


## Author

**Zia Ur Rehman**

**AI Engineer**

