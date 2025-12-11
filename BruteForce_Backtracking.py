import itertools
import time
import os

### CLASS DATA & LOGIC CƠ BẢN (BOARD)
# Class chịu trách nhiệm lưu trữ thông tin về đảo
class Island:
    def __init__(self, r, c, number):
        self.r = r
        self.c = c 
        self.number = number 
        self.current_bridges = 0  

    def __repr__(self):
        return f"({self.r},{self.c})[{self.number}]"

# Class chính của bảng Hashiwokakero chịu trách nhiệm: đọc input (grid), tìm cạnh tiềm năng (Pre-processing), cung cấp hàm kiểm tra hợp lệ chung (như Connectivity)
class HashiBoard:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.islands = []
        self.bridges = []  
        
        self._parse_input()
        self._find_potential_bridges()

    def _parse_input(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] > 0:
                    self.islands.append(Island(r, c, self.grid[r][c]))

    def _find_potential_bridges(self):
        potential_edges = []
        
        def get_neighbor(island, dr, dc):
            r, c = island.r + dr, island.c + dc
            while 0 <= r < self.rows and 0 <= c < self.cols:
                if self.grid[r][c] > 0:
                    for neighbor in self.islands:
                        if neighbor.r == r and neighbor.c == c:
                            return neighbor
                r += dr
                c += dc
            return None

        for island in self.islands:
            # Hướng Đông
            neighbor_right = get_neighbor(island, 0, 1)
            if neighbor_right:
                potential_edges.append((island, neighbor_right, 'H'))
            # Hướng Nam
            neighbor_down = get_neighbor(island, 1, 0)
            if neighbor_down:
                potential_edges.append((island, neighbor_down, 'V'))

        self.bridges = []
        for i, edge in enumerate(potential_edges):
            conflicts = []
            u1, v1, type1 = edge
            for j, other_edge in enumerate(potential_edges):
                if i == j: continue
                u2, v2, type2 = other_edge
                
                # Check crossing
                if type1 == 'H' and type2 == 'V':
                    if (u1.c < u2.c < v1.c) and (u2.r < u1.r < v2.r):
                        conflicts.append(j)
                elif type1 == 'V' and type2 == 'H':
                    if (u2.c < u1.c < v2.c) and (u1.r < u2.r < v1.r):
                        conflicts.append(j)
            
            self.bridges.append({
                'u': u1, 'v': v1, 'type': type1, 
                'conflicts': conflicts, 
                'val': 0
            })
    # Hàm reset trạng thái đếm cầu của các đảo về 0
    def reset_state(self):
        for island in self.islands:
            island.current_bridges = 0
        for bridge in self.bridges:
            bridge['val'] = 0
    # Hàm kiểm tra tính liên thông của đồ thị dựa trên danh sách giá trị cầu (dùng BFS)
    def check_connectivity(self, bridge_values):
        # Xây dựng đồ thị kề
        adj = {island: [] for island in self.islands}
        for i, val in enumerate(bridge_values):
            if val > 0:
                u = self.bridges[i]['u']
                v = self.bridges[i]['v']
                adj[u].append(v)
                adj[v].append(u)
        
        if not self.islands: return True
        
        start_node = self.islands[0]
        queue = [start_node]
        visited = {start_node}
        count = 0
        while queue:
            curr = queue.pop(0)
            count += 1
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return count == len(self.islands)
    # Hàm xuất lời giải ra dạng ma trận hiển thị
    def export_solution_to_grid(self, bridge_values):
        # Tạo grid cơ sở từ dữ liệu gốc
        output_grid = []
        for r in range(self.rows):
            row_strs = []
            for c in range(self.cols):
                val = self.grid[r][c]
                row_strs.append(str(val))
            output_grid.append(row_strs)
        # Vẽ cầu lên grid
        for i, val in enumerate(bridge_values):
            if val > 0:
                edge = self.bridges[i]
                u, v = edge['u'], edge['v']
                bridge_type = edge['type']
                symbol = ""
                if bridge_type == 'H':
                    symbol = "-" if val == 1 else "="
                else: 
                    symbol = "|" if val == 1 else "$" 
                if bridge_type == 'H':
                    c_start = min(u.c, v.c) + 1
                    c_end = max(u.c, v.c)
                    for c in range(c_start, c_end):
                        output_grid[u.r][c] = symbol
                else:
                    r_start = min(u.r, v.r) + 1
                    r_end = max(u.r, v.r)
                    for r in range(r_start, r_end):
                        output_grid[r][u.c] = symbol
                        
        return output_grid

### CLASS BRUTE FORCE SOLVER
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
    
## CLASS BACKTRACKING SOLVER
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
    
### CÁC HÀM HỖ TRỢ ĐỌC/GHI FILE
# Hàm đọc file txt và trả về ma trận 2D
def read_input_file(filename):
    grid = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue  
                parts = line.replace(',', ' ').split()
                row = [int(x) for x in parts]
                grid.append(row)
        return grid
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{filename}'")
        return None
    except ValueError:
        print("Lỗi: File chứa ký tự không phải số.")
        return None

### CÁC HÀM WRAPPER ĐỂ GỌI TỪ BÊN NGOÀI (Cho main.py)
def run_backtracking_solver(input_path, output_path):
    try:
        grid_data = read_input_file(input_path)
        if not grid_data: return None
        
        board = HashiBoard(grid_data)
        solver = BacktrackingSolver(board)
        
        start = time.time()
        solution = solver.solve()
        end = time.time()
        duration = end - start
        
        # Ghi kết quả
        with open(output_path, "w") as f:
            if solution:
                visual_grid = board.export_solution_to_grid(solution)
                for row in visual_grid:
                    f.write(str(row).replace("'", '"') + "\n")
            else:
                f.write("NO SOLUTION FOUND")
                
        return duration
    except Exception as e:
        print(f"Lỗi khi chạy {input_path}: {e}")
        return None

def run_bruteforce_solver(input_path, output_path):
    try:
        grid_data = read_input_file(input_path)
        if not grid_data: return None
        
        board = HashiBoard(grid_data)
        
        # Giới hạn số cạnh <= 18 để tránh treo máy
        if len(board.bridges) > 18: 
            with open(output_path, "w") as f:
                f.write(f"SKIPPED (Too many edges: {len(board.bridges)} > 18)")
            return None

        solver = BruteForceSolver(board)
        
        start = time.time()
        solution = solver.solve()
        end = time.time()
        duration = end - start
        
        with open(output_path, "w") as f:
            if solution:
                visual_grid = board.export_solution_to_grid(solution)
                for row in visual_grid:
                    f.write(str(row).replace("'", '"') + "\n")
            else:
                f.write("NO SOLUTION FOUND")
        
        return duration
    except Exception as e:
        print(f"Lỗi khi chạy {input_path}: {e}")
        return None