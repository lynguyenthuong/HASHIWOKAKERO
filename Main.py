import os
from PySAT_AStar import run_pysat_solver, run_astar_solver
from BruteForce_Backtracking import run_bruteforce_solver, run_backtracking_solver

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

# ==========================================
# CẤU HÌNH CHẠY
# ==========================================
# Để None nếu muốn chạy tất cả. Điền tên file nếu muốn chạy lẻ (VD: "input-01.txt")
TARGET_FILE = "input-02.txt" 

def process_file(filename):
    """Hàm xử lý chạy toàn bộ 4 thuật toán cho 1 file cụ thể"""
    input_path = os.path.join(INPUT_FOLDER, filename)
    
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy file '{filename}' trong thư mục input.")
        return

    # --- TẠO TÊN FILE OUTPUT (Sửa đổi theo yêu cầu) ---
    # Ví dụ: input-01.txt -> output-01.txt
    #        input_01.txt -> output_01.txt
    output_filename = filename.replace("input", "output")
    
    print(f"{'='*10} Đang xử lý: {filename} {'='*10}")
    
    # --- 1. PySAT ---
    out_pysat = os.path.join(OUTPUT_FOLDER, "PySAT", output_filename)
    t = run_pysat_solver(input_path, out_pysat)
    print(f"1. PySAT       : {t:.4f}s" if t is not None else "1. PySAT       : Error")

    # --- 2. AStar ---
    out_astar = os.path.join(OUTPUT_FOLDER, "AStar", output_filename)
    t = run_astar_solver(input_path, out_astar)
    print(f"2. AStar       : {t:.4f}s" if t is not None else "2. AStar       : Error/None")

    # --- 3. Brute Force ---
    out_bf = os.path.join(OUTPUT_FOLDER, "BruteForce", output_filename)
    t = run_bruteforce_solver(input_path, out_bf)
    print(f"3. BruteForce  : {t:.4f}s" if t is not None else "3. BruteForce  : Skipped/Error")

    # --- 4. Backtracking ---
    out_bt = os.path.join(OUTPUT_FOLDER, "Backtracking", output_filename)
    t = run_backtracking_solver(input_path, out_bt)
    print(f"4. Backtracking: {t:.4f}s" if t is not None else "4. Backtracking: Error")
    print("-" * 46)

def main():
    # 1. Tạo cấu trúc thư mục Output
    methods = ["PySAT", "AStar", "BruteForce", "Backtracking"]
    for method in methods:
        os.makedirs(os.path.join(OUTPUT_FOLDER, method), exist_ok=True)

    if not os.path.exists(INPUT_FOLDER):
        print(f"Lỗi: Không tìm thấy thư mục '{INPUT_FOLDER}'")
        return

    # 2. Xử lý theo cấu hình
    if TARGET_FILE:
        print(f"-> Đang chạy chế độ SINGLE FILE: {TARGET_FILE}\n")
        process_file(TARGET_FILE)
    else:
        # Lấy danh sách file và sắp xếp
        input_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")])
        print(f"-> Đang chạy chế độ BATCH (Tìm thấy {len(input_files)} file)\n")
        
        for filename in input_files:
            process_file(filename)

    print("\nHoàn tất! Kiểm tra thư mục 'output'.")

if __name__ == "__main__":
    main()