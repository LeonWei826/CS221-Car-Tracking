import time
import math
import random
import util
from engine.const import Const
from submission import ExactInference, ParticleFilter


def run_evaluation():
    # 回归官方标准的 31x31 地图
    numRows, numCols = 31, 31
    num_steps = 100

    agentX, agentY = 15, 15
    true_car_row, true_car_col = 25, 25

    exact = ExactInference(numRows, numCols)
    particle_200 = ParticleFilter(numRows, numCols)

    # 测试 50 个粒子的极限情况
    particle_50 = ParticleFilter(numRows, numCols)
    particle_50.NUM_PARTICLES = 50

    # 初始化 50 粒子的分布
    particle_50.particles = {}
    potentialParticles = list(particle_50.transProbDict.keys())
    for _ in range(50):
        p = potentialParticles[int(random.random() * len(potentialParticles))]
        particle_50.particles[p] = particle_50.particles.get(p, 0) + 1
    particle_50.updateBelief()

    times = {"exact": 0, "pf200": 0, "pf50": 0}
    errors = {"exact": 0, "pf200": 0, "pf50": 0}

    print(f"开始模拟 {num_steps} 步...")
    for step in range(num_steps):
        true_x = util.colToX(true_car_col)
        true_y = util.rowToY(true_car_row)
        true_dist = math.sqrt((agentX - true_x) ** 2 + (agentY - true_y) ** 2)
        observedDist = random.gauss(true_dist, Const.SONAR_STD)

        # 1. 精确推断
        start = time.perf_counter()
        exact.observe(agentX, agentY, observedDist)
        exact.elapseTime()
        times["exact"] += (time.perf_counter() - start)

        # 2. 粒子滤波 (200)
        start = time.perf_counter()
        particle_200.observe(agentX, agentY, observedDist)
        particle_200.elapseTime()
        times["pf200"] += (time.perf_counter() - start)

        # 3. 粒子滤波 (50)
        start = time.perf_counter()
        particle_50.observe(agentX, agentY, observedDist)
        particle_50.elapseTime()
        times["pf50"] += (time.perf_counter() - start)

        # 模拟车辆运动
        true_car_row += 1 if random.random() > 0.5 else -1
        true_car_col += 1 if random.random() > 0.5 else -1
        true_car_row = max(10, min(20, true_car_row))
        true_car_col = max(10, min(20, true_car_col))

        # 计算误差
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

    print("\n========== 测试结果 ==========")
    for key, name in [("exact", "精确推断"), ("pf200", "粒子滤波(200)"), ("pf50", "粒子滤波(50)")]:
        print(f"{name}: 平均单步 {times[key] / num_steps * 1000:.2f} ms, 平均误差 {errors[key] / num_steps:.2f} 格")
    print("==============================")


if __name__ == "__main__":
    run_evaluation()