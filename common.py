"""
File chứa Grid, IO read/write
"""

import os
from typing import List, Any, Optional

### Đại diện cho lưới bài toán Hashi (Dùng chung)
class Grid:
    def __init__(self, width: int, height: int, data: List[List[Any]]):
        self.width = width
        self.height = height
        self.data = data # Ma trận 2D

### Đọc file input trả về đối tượng Grid
def read_grid_from_file(file_path: str) -> Optional[Grid]:
    try:
        matrix = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    matrix.append([int(x) for x in line.replace(',', ' ').split()])
        if not matrix: return None
        return Grid(len(matrix[0]), len(matrix), matrix)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

### Ghi kết quả ra file
def write_solution_to_file(data, filepath):
    try:
        with open(filepath, "w") as f:
            if data:
                for row in data:
                    f.write(str(row).replace("'", '"') + "\n")
            else:
                f.write("NO SOLUTION FOUND / UNSAT\n")
    except Exception as e:
        print(f"Lỗi khi ghi file {filepath}: {e}")

### Chuyển danh sách cạnh thành Grid hiển thị
def format_solution_output(grid: Grid, solution_edges: List[dict]):
    rows = grid.height
    cols = grid.width
    out_grid = [["0" for _ in range(cols)] for _ in range(rows)]
    
    # 1. Điền đảo
    for r in range(rows):
        for c in range(cols):
            if grid.data[r][c] > 0:
                out_grid[r][c] = str(grid.data[r][c])
        
    # 2. Điền cầu
    for edge in solution_edges:
        bridges = edge['bridges']
        c1, r1 = edge['u']
        c2, r2 = edge['v']
        
        symbol = ""
        is_hor = (r1 == r2)
        
        if is_hor:
            symbol = "-" if bridges == 1 else "="
            c_start, c_end = min(c1, c2), max(c1, c2)
            for c in range(c_start + 1, c_end):
                out_grid[r1][c] = symbol
        else:
            symbol = "|" if bridges == 1 else "$"
            r_start, r_end = min(r1, r2), max(r1, r2)
            for r in range(r_start + 1, r_end):
                out_grid[r][c1] = symbol

    return out_grid