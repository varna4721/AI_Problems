import numpy as np
import heapq
import random
import math
import time
import matplotlib.pyplot as plt


def a_star(grid, start, goal, heuristic_func):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0 + heuristic_func(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    expanded = 0
    while open_set:
        _, cost, current = heapq.heappop(open_set)
        expanded += 1
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, expanded
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0:
                tentative_g = cost + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic_func(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                    came_from[neighbor] = current
    return None, expanded  

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def diagonal(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def create_grid(size=20, obstacle_prob=0.2):
    grid = np.zeros((size, size), dtype=int)
    # Place obstacles
    for i in range(size):
        for j in range(size):
            if random.random() < obstacle_prob:
                grid[i, j] = 1
    while True:
        start = (random.randint(0, size-1), random.randint(0, size-1))
        goal = (random.randint(0, size-1), random.randint(0, size-1))
        if grid[start] == 0 and grid[goal] == 0 and start != goal:
            break
    return grid, start, goal

def evaluate_heuristics(runs=20, grid_size=20):
    heuristics = {
        "Manhattan": manhattan,
        "Euclidean": euclidean,
        "Diagonal": diagonal
    }

    performance = {h: {"time": [], "expanded": [], "path_len": []} for h in heuristics}

    for _ in range(runs):
        grid, start, goal = create_grid(grid_size)

        for name, h_func in heuristics.items():
            start_time = time.time()
            path, expanded = a_star(grid, start, goal, h_func)
            end_time = time.time()

            duration = end_time - start_time
            path_len = len(path) if path else np.inf

            performance[name]["time"].append(duration)
            performance[name]["expanded"].append(expanded)
            performance[name]["path_len"].append(path_len)

    return performance



def plot_results(results):
    methods = ["Manhattan", "Euclidean", "Diagonal"]
    metrics = ["time", "expanded", "path_len"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for i, metric in enumerate(metrics):
        avg_vals = [sum(results[m][metric]) / len(results[m][metric]) for m in methods]
        axs[i].bar(methods, avg_vals, color=["#1f77b4", "#2ca02c"])
        axs[i].set_title(f"Average {metric.capitalize()}")
        axs[i].set_ylabel(metric.capitalize())
        axs[i].grid(True, linestyle="--", alpha=0.6)
    plt.suptitle("Timetable CSP: Backtracking Heuristics vs Forward Checking")
    plt.show()
    
if __name__ == "__main__":
    results = evaluate_heuristics(runs=30, grid_size=25)
    plot_results(results)
