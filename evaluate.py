# -*- coding: utf-8 -*-
"""
Benchmark Evaluation Script for HMM Inference Algorithms

This script compares the performance (execution time and accuracy) of:
1. Exact Inference (Full grid traversal)
2. Particle Filter (200 particles)
3. Particle Filter (50 particles)

It simulates a moving vehicle and evaluates the average step time and Manhattan error.
"""

import time
import math
import random
import util
from engine.const import Const
from submission import ExactInference, ParticleFilter


def run_evaluation():
    # Standard 31x31 grid map
    numRows, numCols = 31, 31
    num_steps = 100

    # Agent (observing car) position
    agentX, agentY = 15, 15
    # True target car starting position
    true_car_row, true_car_col = 25, 25

    # Initialize inference models
    exact = ExactInference(numRows, numCols)
    particle_200 = ParticleFilter(numRows, numCols)

    # Test the extreme case of 50 particles
    particle_50 = ParticleFilter(numRows, numCols)
    particle_50.NUM_PARTICLES = 50

    # Initialize distribution for 50 particles
    particle_50.particles = {}
    potentialParticles = list(particle_50.transProbDict.keys())
    for _ in range(50):
        p = potentialParticles[int(random.random() * len(potentialParticles))]
        particle_50.particles[p] = particle_50.particles.get(p, 0) + 1
    particle_50.updateBelief()

    # Track performance metrics
    times = {"exact": 0, "pf200": 0, "pf50": 0}
    errors = {"exact": 0, "pf200": 0, "pf50": 0}

    print(f"Starting simulation for {num_steps} steps...")
    for step in range(num_steps):
        # Generate true distance and Gaussian noise observation
        true_x = util.colToX(true_car_col)
        true_y = util.rowToY(true_car_row)
        true_dist = math.sqrt((agentX - true_x) ** 2 + (agentY - true_y) ** 2)
        observedDist = random.gauss(true_dist, Const.SONAR_STD)

        # 1. Exact Inference
        start = time.perf_counter()
        exact.observe(agentX, agentY, observedDist)
        exact.elapseTime()
        times["exact"] += (time.perf_counter() - start)

        # 2. Particle Filter (200)
        start = time.perf_counter()
        particle_200.observe(agentX, agentY, observedDist)
        particle_200.elapseTime()
        times["pf200"] += (time.perf_counter() - start)

        # 3. Particle Filter (50)
        start = time.perf_counter()
        particle_50.observe(agentX, agentY, observedDist)
        particle_50.elapseTime()
        times["pf50"] += (time.perf_counter() - start)

        # Simulate random movement of the target car
        true_car_row += 1 if random.random() > 0.5 else -1
        true_car_col += 1 if random.random() > 0.5 else -1
        true_car_row = max(10, min(20, true_car_row))
        true_car_col = max(10, min(20, true_car_col))

        # Calculate Manhattan distance error (Peak probability vs True position)
        def get_error(belief):
            max_prob, max_r, max_c = -1, -1, -1
            for r in range(numRows):
                for c in range(numCols):
                    p = belief.getProb(r, c)
                    if p > max_prob:
                        max_prob, max_r, max_c = p, r, c
            return abs(max_r - true_car_row) + abs(max_c - true_car_col)

        errors["exact"] += get_error(exact.getBelief())
        errors["pf200"] += get_error(particle_200.getBelief())
        errors["pf50"] += get_error(particle_50.getBelief())

    print("\n========== Benchmark Results ==========")
    for key, name in [("exact", "Exact Inference"), ("pf200", "Particle Filter (200)"), ("pf50", "Particle Filter (50)")]:
        print(f"{name}: Avg Step Time {times[key] / num_steps * 1000:.2f} ms, Avg Error {errors[key] / num_steps:.2f} tiles")
    print("=======================================")


if __name__ == "__main__":
    run_evaluation()