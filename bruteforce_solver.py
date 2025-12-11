"""
File chứa BruteForceSolver & Wrapper
"""

import itertools
import time
from graph_board import HashiBoard
from common import read_grid_from_file, write_solution_to_file

class BruteForceSolver:
    def __init__(self, board): 
        self.board = board

    # Hàm kiểm tra toàn diện cho một cấu hình
    def _is_valid_full_check(self, bridge_values):
        # Reset & Update số cầu (cần reset lại từ đầu mỗi lần check)
        for island in self.board.islands:
            island.current_bridges = 0
            
        for i, val in enumerate(bridge_values):
            if val > 0:
                edge = self.board.bridges[i]
                # Check Crossing
                for conflict_idx in edge['conflicts']:
                    if bridge_values[conflict_idx] > 0:
                        return False
                
                edge['u'].current_bridges += val
                edge['v'].current_bridges += val

        # Check Capacity
        for island in self.board.islands:
            if island.current_bridges != island.number:
                return False

        # Check Connectivity
        return self.board.check_connectivity(bridge_values)
    
    def solve(self):
        self.board.reset_state()
        num_edges = len(self.board.bridges)
        possible_values = [0, 1, 2]
        
        # Sinh tổ hợp
        for configuration in itertools.product(possible_values, repeat=num_edges):
            if self._is_valid_full_check(configuration):
                return list(configuration)
        return None

def run_bruteforce_solver(input_path, output_path):
    try:
        grid = read_grid_from_file(input_path)
        if not grid: return None
        board = HashiBoard(grid)
        if len(board.bridges) > 18: 
            with open(output_path, "w") as f: f.write("SKIPPED (>18 edges)")
            return None
        
        solver = BruteForceSolver(board)
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