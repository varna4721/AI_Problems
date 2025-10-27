#ROBOT-FINDING_PROBLEM#
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


#CSP-Timetable_Genaration#
import itertools
import random
import time
from kiwisolver import Constraint
import matplotlib.pyplot as plt


courses = ["C1", "C2", "C3", "C4"]
teachers = ["T1", "T2", "T3", "T4"]
timeslots = ["Morning", "Afternoon", "Evening"]
rooms = ["R1", "R2"]

domain = list(itertools.product(timeslots, rooms, teachers))
domains = {c: domain.copy() for c in courses}

def is_consistent(assignments, var, value):
    for other_var, other_val in assignments.items():
        if other_val is None:
            continue
        if value[0] == other_val[0]:
            if value[1] == other_val[1]:
                return False
            if value[2] == other_val[2]:
                return False
    return True


def select_unassigned_variable(assignments, domains):
    unassigned = [v for v in assignments if assignments[v] is None]
    return min(unassigned, key=lambda var: len(domains[var]))


def order_domain_values(var, domains, assignments):
    
    def count_constraints(value):
        count = 0
        for other_var in assignments:
            if assignments[other_var] is None:
                for other_val in domains[other_var]:
                    if not is_consistent({var: value}, other_var, other_val):
                        count += 1
        return count
    return sorted(domains[var], key=count_constraints)






def backtrack_with_heuristics(assignment, domains, metrics):
# Create empty assignment ONCE
    assignment = {var: None for var in domains}
    metrics = {"assignments": 0, "backtracks": 0, "time": 0}
# Call solver — no need to copy() every time
    result = backtrack_with_heuristics(assignment, domains, metrics)
    # Base case: all variables are assigned
    if all(assignment[v] is not None for v in assignment):
        return assignment

    # Select variable using heuristics
    var = select_unassigned_variable(assignment, domains)

    # Try each value in order
    for value in order_domain_values(var, domains, assignment):
        if is_consistent(assignment, var, value):
            assignment[var] = value
            metrics["assignments"] += 1

            # Recursive call with updated assignment
            result = backtrack_with_heuristics(assignment, domains, metrics)
            if result is not None:
                return result

            # Undo the assignment and count backtrack
            assignment[var] = None
            metrics["backtracks"] += 1

    return None




def forward_checking(assignments, domains, metrics):
    metrics = {"assignments": 0, "backtracks": 0, "time": 0}
    result = forward_checking(assignments.copy(), domains.copy(), metrics)
    # Base case: if all variables are assigned
    if all(assignments[v] is not None for v in assignments):
        return assignments

    var = select_unassigned_variable(assignments, domains)

    for value in order_domain_values(var, domains, assignments):
        if is_consistent(assignments, var, value):
            assignments[var] = value
            metrics["assignments"] += 1

            # Copy domains to simulate forward checking
            new_domains = {v: domains[v][:] for v in domains}

            consistent = True
            for other_var in domains:
                if assignments[other_var] is None:
                    new_domains[other_var] = [
                        val for val in new_domains[other_var]
                        if is_consistent(assignments, other_var, val)
                    ]
                    # If any domain becomes empty, backtrack
                    if not new_domains[other_var]:
                        consistent = False
                        break

            if consistent:
                result = forward_checking(assignments, new_domains, metrics)
                if result is not None:
                    return result

            # Undo assignment and count backtrack
            assignments[var] = None
            metrics["backtracks"] += 1

    return None



def run_experiment(runs=10):
    results = {"heuristics": {"time": [], "assignments": [], "backtracks": []},
                "forward": {"time": [], "assignments": [], "backtracks": []}}
    for _ in range(runs):
        assignments = {v: None for v in courses}
        metrics = {"assignments": 0, "backtracks": 0}
        start = time.time()
        backtrack_with_heuristics(assignments, domains, metrics)
        _extracted_from_run_experiment_9(results, "heuristics", start, metrics)
        assignments = {v: None for v in courses}
        metrics = {"assignments": 0, "backtracks": 0}
        start = time.time()
        forward_checking(assignments, domains, metrics)
        _extracted_from_run_experiment_9(results, "forward", start, metrics)
    return results



def _extracted_from_run_experiment_9(results, arg1, start, metrics):
    results[arg1]["time"].append(time.time() - start)
    results[arg1]["assignments"].append(metrics["assignments"])
    results[arg1]["backtracks"].append(metrics["backtracks"])


def plot_results(results):
    methods = ["heuristics", "forward"]
    metrics = ["time", "assignments", "backtracks"]
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
    results = run_experiment(runs=15)
    plot_results(results)

