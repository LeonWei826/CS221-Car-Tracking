# Multi-Car Tracking System (Bayesian Inference & Particle Filter)

This repository contains my personal implementation of exact inference and particle filter algorithms for tracking vehicles on a grid. This was originally developed as part of my study of the Stanford CS221 course assignment.

**Note:** To comply with academic integrity policies, this repository only includes my core algorithmic implementations (`inference.py`) and performance evaluation script (`evaluate.py`). The official course framework and graders are **not** included.

## 🚀 Key Features
* **Exact Inference (HMM):** Implemented exact Bayesian filtering to track vehicle positions, including observation updates and time elapse prediction.
* **Particle Filter:** Implemented approximate inference using particle weighting and resampling.
* **Performance Evaluation:** Built a custom evaluation script to quantify the trade-off between execution time and localization accuracy under high-noise conditions.

## 📊 Performance & Benchmarking

*Tested on a standard 31x31 grid (100 steps). Note: The sensors in this simulation have a very high noise level (Gaussian STD = 20.0), which significantly impacts accuracy.*

| Method | Average Time (ms/step) | Average Error (grids) | Speedup |
| :--- | :--- | :--- | :--- |
| Exact Inference | 1.31 ms | 14.49 | 1.0x |
| Particle Filter (200) | 2.39 ms | 14.50 | 0.5x |
| Particle Filter (50) | **0.70 ms** | 14.53 | **1.9x** |

### 💡 Engineering Insights
* **Time-Accuracy Trade-off:** In extreme noise environments, the exact inference and particle filter converge to similar error rates (around 14.5 grids). However, reducing the particle count from 200 to 50 achieves a **1.9x speedup** without sacrificing accuracy.
* **Algorithm Selection:** While exact inference is faster in small grids due to Python's overhead for dictionary operations in particle filters, the particle filter's time complexity is independent of the grid size. It is the preferred choice for large-scale environments where traversing the entire grid becomes computationally infeasible.

## 📂 File Structure
* `inference.py`: Core implementation of `ExactInference` and `ParticleFilter`.
* `evaluate.py`: Custom script to benchmark time and accuracy.

## 👤 Author
[LeonWei826](https://github.com/LeonWei826)
