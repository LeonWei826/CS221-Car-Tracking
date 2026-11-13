# 🧪 Independent Algorithm Benchmark Report

## Objective
To evaluate the performance (execution time) and accuracy (localization error) of the Exact Inference and Particle Filter algorithms independently of the official course framework. I built a standalone 1D circular grid simulator to act as a testbed.

## Experimental Setup
* **Environment:** Custom 1D circular grid.
* **Noise Level:** Gaussian noise with STD = 5.0.
* **Metrics:** Average execution time per step (ms) and average position error (grid cells).

## 📊 Key Findings

### 1. Grid Size Scaling (100 vs 1000 cells)
* **Exact Inference:** Execution time increases linearly with grid size. It went from **35.25 ms** (100 cells) to over **350 ms** (1000 cells), making it computationally infeasible for large-scale environments.
* **Particle Filter:** Execution time remains constant regardless of grid size. It took roughly **44 ms** for both 100 and 1000 cells.

### 2. Particle Count & Accuracy Trade-off (200 vs 50 particles)
* In a **100-cell** grid, reducing particles from 200 to 50 achieved a **13.8x speedup** (from 44.47ms to 3.20ms) with **no loss in accuracy** (error remained ~2.5 cells). This proves that in small spaces, 50 particles are sufficient.
* In a **1000-cell** grid, 50 particles are too sparse. The particle filter fails to track the car, and the error spikes significantly. 200 particles are required to maintain tracking accuracy.

## 🖼️ Visualization
*(Assuming you upload the image to the repo)*
![3D Circular Grid Tracking](circular_track_3d.png)
*This 3D visualization shows the true trajectory (green), noisy observations (gray dots), exact inference (blue), and particle filter with 50 particles (red) on the 1000-cell circular track.*

## 💡 Engineering Takeaway
There is no "one-size-fits-all" algorithm. A robust tracking system must dynamically adjust its particle count based on the size of the state space and sensor noise, balancing computational cost and tracking accuracy.
