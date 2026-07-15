import argparse
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src import (
    ParticleSwarmOptimizer, 
    GeneticAlgorithm, 
    otsu_fitness, 
    kapur_fitness, 
    compute_histogram, 
    preprocess_image, 
    apply_multilevel_threshold, 
    compute_psnr, 
    compute_ssim
)

def process_single_image(image_path="data/sample.png", num_thresholds=4, max_iter=50, output_dir="outputs/threshold_result"):
    os.makedirs(output_dir, exist_ok=True)
    
    raw_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw_image is None:
        print(f"Error: Could not read image at '{image_path}'. Ensure the file exists in 'data/'.")
        return

    image = preprocess_image(raw_image)
    probabilities = compute_histogram(image)
    filename = os.path.basename(image_path).split('.')[0]

    print(f"\n--- Running Optimization Pipeline ---")
    print(f"Target Image: {image_path}")
    print(f"Threshold Cuts: {num_thresholds} | Iterations: {max_iter}\n")
    
    t0 = time.time()
    pso = ParticleSwarmOptimizer(30, num_thresholds, max_iter, probabilities, otsu_fitness)
    t_pso, hist_pso = pso.optimize()
    time_pso = time.time() - t0
    seg_pso = apply_multilevel_threshold(image, t_pso)
    psnr_pso = compute_psnr(image, seg_pso)
    ssim_pso = compute_ssim(image, seg_pso)

    t0 = time.time()
    ga = GeneticAlgorithm(30, num_thresholds, max_iter, probabilities, otsu_fitness)
    t_ga, hist_ga = ga.optimize()
    time_ga = time.time() - t0
    seg_ga = apply_multilevel_threshold(image, t_ga)
    psnr_ga = compute_psnr(image, seg_ga)
    ssim_ga = compute_ssim(image, seg_ga)


    fig1 = plt.figure(figsize=(14, 10))
    grid = fig1.add_gridspec(3, 3, height_ratios=[1, 0.8, 0.3])

    ax1 = fig1.add_subplot(grid[0, 0])
    ax1.imshow(image, cmap='gray')
    ax1.set_title("Input Image (Pre-smoothed)")
    ax1.axis('off')

    ax2 = fig1.add_subplot(grid[0, 1])
    ax2.imshow(seg_pso, cmap='gray')
    ax2.set_title(f"PSO Otsu ({num_thresholds}-Thresholds)")
    ax2.axis('off')

    ax3 = fig1.add_subplot(grid[0, 2])
    ax3.imshow(seg_ga, cmap='gray')
    ax3.set_title(f"GA Otsu ({num_thresholds}-Thresholds)")
    ax3.axis('off')

    ax4 = fig1.add_subplot(grid[1, :])
    counts, bins = np.histogram(image, bins=256, range=(0, 256))
    ax4.bar(bins[:-1], counts, width=1.0, color='gray', alpha=0.6, label='Pixel Distribution')
    for t in t_pso:
        ax4.axvline(t, color='blue', linestyle='--', linewidth=1.5, label=f'PSO T={t}')
    for t in t_ga:
        ax4.axvline(t, color='orange', linestyle=':', linewidth=1.5, label=f'GA T={t}')
    ax4.set_title("Histogram & Threshold Position Overlay")
    ax4.set_xlim([0, 255])
    ax4.legend(loc='upper right')

    ax5 = fig1.add_subplot(grid[2, :])
    ax5.axis('off')
    table_data = [
        ["Algorithm", "Optimal Thresholds", "Fitness Score", "PSNR (dB)", "SSIM", "Runtime"],
        ["Particle Swarm (PSO)", str(t_pso.tolist()), f"{hist_pso[-1]:.2f}", f"{psnr_pso:.2f}", f"{ssim_pso:.4f}", f"{time_pso:.3f} s"],
        ["Genetic Algorithm (GA)", str(t_ga.tolist()), f"{hist_ga[-1]:.2f}", f"{psnr_ga:.2f}", f"{ssim_ga:.4f}", f"{time_ga:.3f} s"]
    ]
    table = ax5.table(cellText=table_data, loc='center', cellLoc='center')
    table.scale(1, 1.4)
    table.set_fontsize(10)

    plt.tight_layout()
    
    fig1_path = os.path.join(output_dir, f"{filename}_comparison.png")
    plt.savefig(fig1_path)
    print(f"Saved comparison figure to '{fig1_path}'")

    fig2 = plt.figure(figsize=(10, 4))
    plt.plot(hist_pso, label='PSO Convergence', color='blue', linewidth=2)
    plt.plot(hist_ga, label='GA Convergence', color='orange', linewidth=2)
    plt.title(f"Fitness Convergence Curves ({filename.capitalize()})")
    plt.xlabel("Iteration")
    plt.ylabel("Otsu Between-Class Variance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig2_path = os.path.join(output_dir, f"{filename}_convergence.png")
    plt.savefig(fig2_path)
    print(f"Saved convergence figure to '{fig2_path}'")

    print("Opening interactive result windows...")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilevel Image Thresholding Pipeline")
    parser.add_argument("--image", type=str, default="data/sample.png", help="Target image path")
    parser.add_argument("--levels", type=int, default=4, help="Number of thresholds")
    parser.add_argument("--iterations", type=int, default=50, help="Max iterations")
    
    args = parser.parse_args()
    process_single_image(args.image, args.levels, args.iterations)