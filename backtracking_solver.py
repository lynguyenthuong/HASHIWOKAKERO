"""
File chứa BacktrackingSolver & Wrapper
"""

import time
from graph_board import HashiBoard
from common import read_grid_from_file, write_solution_to_file

class BacktrackingSolver:
    def __init__(self, board):
        self.board = board
        self.solution = []
        # Pre-processing: Tạo map từ đảo sang các cạnh nối nó
        self.island_edges_map = {island: [] for island in self.board.islands}
        for idx, edge in enumerate(self.board.bridges):
            self.island_edges_map[edge['u']].append(idx)
            self.island_edges_map[edge['v']].append(idx)

    def solve(self):
        self.board.reset_state()
        self.solution = [0] * len(self.board.bridges)
        # Pre-processing: Tính tiềm năng tối đa ban đầu cho mỗi đảo
        self.island_max_potential = {}
        for island, edge_indices in self.island_edges_map.items():
            self.island_max_potential[island] = len(edge_indices) * 2
            
        if self._backtrack_recursive(0):
            return self.solution
        return None

    def _backtrack_recursive(self, edge_idx):
        # BASE CASE: Đã duyệt hết các cạnh
        if edge_idx == len(self.board.bridges):
            for island in self.board.islands:
                if island.current_bridges != island.number:
                    return False
            return self.board.check_connectivity(self.solution)

        # Lấy thông tin cạnh hiện tại
        edge = self.board.bridges[edge_idx]
        u, v = edge['u'], edge['v']

        # CẬP NHẬT TRẠNG THÁI TIỀM NĂNG TRƯỚC KHI THỬ CÁC GIÁ TRỊ
        self.island_max_potential[u] -= 2
        self.island_max_potential[v] -= 2

        # Xác định khoảng giá trị có thể thử (Heuristic cũ)
        rem_u = u.number - u.current_bridges
        rem_v = v.number - v.current_bridges
        max_possible = min(2, rem_u, rem_v)


        # (Greedy heuristic: try 2, then 1, then 0)
        try_values = range(max_possible, -1, -1) # Thử 2, 1, 0

        for val in try_values:
            # Pruning: Check Crossing
            is_conflict = False
            if val > 0:
                for conflict_idx in edge['conflicts']:
                    if conflict_idx < edge_idx:
                        if self.solution[conflict_idx] > 0:
                            is_conflict = True
                            break
            if is_conflict: continue
            
            # Kiểm tra U:
            if (u.current_bridges + val + self.island_max_potential[u]) < u.number:
                continue # Impossible -> Prune
            
            # Kiểm tra V:
            if (v.current_bridges + val + self.island_max_potential[v]) < v.number:
                continue # Impossible -> Prune

            # Action
            self.solution[edge_idx] = val
            u.current_bridges += val
            v.current_bridges += val
            
            # Recurse
            if self._backtrack_recursive(edge_idx + 1):
                return True
            
            # Backtrack
            u.current_bridges -= val
            v.current_bridges -= val
            self.solution[edge_idx] = 0

        # CẬP NHẬT TRẠNG THÁI TIỀM NĂNG KHI BACKTRACK
        self.island_max_potential[u] += 2
        self.island_max_potential[v] += 2
            
        return False

def run_backtracking_solver(input_path, output_path):
    try:
        grid = read_grid_from_file(input_path)
        if not grid: return None
        board = HashiBoard(grid)
        solver = BacktrackingSolver(board)
        
        start = time.time()
        res = solver.solve()
        dur = time.time() - start
        
        if res:
            out_grid = board.export_solution_to_grid(res)
            write_solution_to_file(out_grid, output_path)
        else:
            write_solution_to_file(None, output_path)
        return dur
    except Exception as e: print(e); return None