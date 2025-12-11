import os
from pysat_solver import run_pysat_solver
from astar_solver import run_astar_solver
from bruteforce_solver import run_bruteforce_solver
from backtracking_solver import run_backtracking_solver

INPUT_FOLDER = "input"; OUTPUT_FOLDER = "output"

### Thay đổi tên file tại đây 
TARGET_FILE = "input-02.txt" # Để None nếu muốn chạy tất cả. Điền tên file nếu muốn chạy lẻ (VD: "input-01.txt")

def process_file(filename):
    input_path = os.path.join(INPUT_FOLDER, filename)
    if not os.path.exists(input_path): return
    output_filename = filename.replace("input", "output")
    print(f"{'='*10} Processing: {filename} {'='*10}")

    # 1. PySAT
    t = run_pysat_solver(input_path, os.path.join(OUTPUT_FOLDER, "PySAT", output_filename))
    print(f"1. PySAT       : {t:.4f}s" if t is not None else "Error")

    # 2. AStar
    t = run_astar_solver(input_path, os.path.join(OUTPUT_FOLDER, "AStar", output_filename))
    print(f"2. AStar       : {t:.4f}s" if t is not None else "Error")

    # 3. BruteForce
    t = run_bruteforce_solver(input_path, os.path.join(OUTPUT_FOLDER, "BruteForce", output_filename))
    print(f"3. BruteForce  : {t:.4f}s" if t is not None else "3. BruteForce  : Skipped/Error")

    # 4. Backtracking
    t = run_backtracking_solver(input_path, os.path.join(OUTPUT_FOLDER, "Backtracking", output_filename))
    print(f"4. Backtracking: {t:.4f}s" if t is not None else "Error")
    print("-" * 40)

def main():
    for m in ["PySAT", "AStar", "BruteForce", "Backtracking"]:
        os.makedirs(os.path.join(OUTPUT_FOLDER, m), exist_ok=True)
    
    if TARGET_FILE: process_file(TARGET_FILE)
    else:
        files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")])
        for f in files: process_file(f)

if __name__ == "__main__": main()