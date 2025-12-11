import itertools
import time
import os

### CLASS DATA & LOGIC CƠ BẢN (BOARD)
class Island:
    """
    Class này chịu trách nhiệm lưu trữ thông tin về đảo
    """
    def __init__(self, r, c, number):
        self.r = r # Chỉ số dòng của đảo
        self.c = c # Chỉ số cột của đảo
        self.number = number # Số cầu cần nối đến đảo này
        self.current_bridges = 0  # Biến trạng thái, dùng đếm số cầu hiện tại

    def __repr__(self):
        return f"({self.r},{self.c})[{self.number}]"
    
class HashiBoard:
    """
    Class này chịu trách nhiệm:
    1. Đọc input (grid)
    2. Tìm cạnh tiềm năng (Pre-processing)
    3. Cung cấp các hàm kiểm tra hợp lệ chung (như Connectivity)
    """
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.islands = []
        self.bridges = []  # List các dict: {'u', 'v', 'type', 'conflicts', 'val'}
        
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

    def reset_state(self):
        """Reset trạng thái đếm cầu của các đảo về 0"""
        for island in self.islands:
            island.current_bridges = 0
        for bridge in self.bridges:
            bridge['val'] = 0

    def check_connectivity(self, bridge_values):
        """
        Kiểm tra tính liên thông dựa trên danh sách giá trị cầu.
        Dùng BFS.
        """
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
    
    def export_solution_to_grid(self, bridge_values):
        """
        Chuyển đổi danh sách giá trị cầu thành ma trận hiển thị (visual grid).
        Output mong muốn: Ma trận các chuỗi ký tự (List of lists of strings).
        Ký hiệu:
            - Ngang đơn: "-"
            - Ngang đôi: "="
            - Dọc đơn:   "|"
            - Dọc đôi:   "$"  (Theo hình ảnh mẫu)
            - Nước:      "0"  (Theo hình ảnh mẫu)
            - Đảo:       "Số" (Giữ nguyên số trên đảo)
        """
        # 1. Tạo grid cơ sở từ dữ liệu gốc
        output_grid = []
        for r in range(self.rows):
            row_strs = []
            for c in range(self.cols):
                val = self.grid[r][c]
                # Chuyển số thành chuỗi, giữ "0" cho ô nước như hình mẫu
                row_strs.append(str(val))
            output_grid.append(row_strs)

        # 2. Vẽ cầu lên grid
        for i, val in enumerate(bridge_values):
            if val > 0:
                edge = self.bridges[i]
                u, v = edge['u'], edge['v']
                bridge_type = edge['type']
                
                # Xác định ký tự cầu
                symbol = ""
                if bridge_type == 'H':
                    symbol = "-" if val == 1 else "="
                else: # Vertical (Dọc)
                    symbol = "|" if val == 1 else "$" # "$" là cầu dọc đôi

                # Điền ký tự vào các ô nằm GIỮA 2 đảo
                if bridge_type == 'H':
                    # Cùng hàng, duyệt cột
                    c_start = min(u.c, v.c) + 1
                    c_end = max(u.c, v.c)
                    for c in range(c_start, c_end):
                        output_grid[u.r][c] = symbol
                else:
                    # Cùng cột, duyệt hàng
                    r_start = min(u.r, v.r) + 1
                    r_end = max(u.r, v.r)
                    for r in range(r_start, r_end):
                        output_grid[r][u.c] = symbol
                        
        return output_grid

### CLASS BRUTE FORCE SOLVER
class BruteForceSolver:
    def __init__(self, board):
        self.board = board

    def _is_valid_full_check(self, bridge_values):
        """
        Kiểm tra toàn diện cho một cấu hình (dùng cho Brute Force)
        """
        # 1. Reset & Update số cầu
        # Lưu ý: Brute force cần reset lại từ đầu mỗi lần check
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

        # 2. Check Capacity
        for island in self.board.islands:
            if island.current_bridges != island.number:
                return False

        # 3. Check Connectivity
        return self.board.check_connectivity(bridge_values)

    def solve(self):
        self.board.reset_state() # Đảm bảo sạch sẽ trước khi chạy
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
        # Pre-processing:
        # 1. Tạo map ánh xạ: Đảo -> Danh sách các index của cạnh nối với nó
        # Để biết đảo này còn bao nhiêu cạnh chưa được gán giá trị
        self.island_edges_map = {island: [] for island in self.board.islands}
        for idx, edge in enumerate(self.board.bridges):
            self.island_edges_map[edge['u']].append(idx)
            self.island_edges_map[edge['v']].append(idx)

    def solve(self):
        self.board.reset_state()
        self.solution = [0] * len(self.board.bridges)
        
        # Tạo mảng theo dõi "tiềm năng còn lại" của từng đảo
        # Ban đầu, tiềm năng = số cạnh nối * 2 (vì mỗi cạnh tối đa 2 cầu)
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

        # --- QUAN TRỌNG: CẬP NHẬT TIỀM NĂNG (LOOK-AHEAD) ---
        # Vì ta đang xét cạnh này, nó không còn là "tiềm năng" nữa mà sẽ thành giá trị thật.
        # Ta tạm trừ đi 2 điểm tiềm năng của u và v (giả sử mất đi cơ hội max)
        # Sau đó khi gán val, ta sẽ cộng val vào current_bridges.
        self.island_max_potential[u] -= 2
        self.island_max_potential[v] -= 2

        # Xác định khoảng giá trị có thể thử (Heuristic cũ)
        rem_u = u.number - u.current_bridges
        rem_v = v.number - v.current_bridges
        max_possible = min(2, rem_u, rem_v)

        # Sắp xếp thứ tự thử: Ưu tiên giá trị lớn (2) trước để dễ thỏa mãn các đảo số lớn
        # (Greedy heuristic: try 2, then 1, then 0)
        # Hoặc giữ nguyên 0->2 nếu muốn tìm nghiệm đơn giản. 
        # Với map lớn, thử 2 -> 0 thường hội tụ nhanh hơn.
        try_values = range(max_possible, -1, -1) # Thử 2, 1, 0

        for val in try_values:
            # 1. Pruning: Check Crossing
            is_conflict = False
            if val > 0:
                for conflict_idx in edge['conflicts']:
                    if conflict_idx < edge_idx:
                        if self.solution[conflict_idx] > 0:
                            is_conflict = True
                            break
            if is_conflict: continue

            # 2. PRUNING MẠNH (LOOK-AHEAD) - QUAN TRỌNG NHẤT
            # Kiểm tra: Liệu u và v có THỂ đủ cầu trong tương lai không?
            # Công thức: (Đã có) + (Sắp gán) + (Tối đa có thể có từ các cạnh chưa duyệt) < (Yêu cầu)
            # Nếu nhỏ hơn -> Cắt tỉa ngay!
            
            # Kiểm tra U:
            if (u.current_bridges + val + self.island_max_potential[u]) < u.number:
                continue # Impossible -> Prune
            
            # Kiểm tra V:
            if (v.current_bridges + val + self.island_max_potential[v]) < v.number:
                continue # Impossible -> Prune

            # 3. Action
            self.solution[edge_idx] = val
            u.current_bridges += val
            v.current_bridges += val
            
            # 4. Recurse
            if self._backtrack_recursive(edge_idx + 1):
                return True
            
            # 5. Backtrack
            u.current_bridges -= val
            v.current_bridges -= val
            self.solution[edge_idx] = 0

        # --- TRẢ LẠI TRẠNG THÁI TIỀM NĂNG TRƯỚC KHI RỜI ĐI ---
        self.island_max_potential[u] += 2
        self.island_max_potential[v] += 2
            
        return False
    
# class BacktrackingSolver:
#     def __init__(self, board):
#         self.board = board
#         self.solution = []

#     def solve(self):
#         self.board.reset_state() # Reset trạng thái các đảo
#         self.solution = [0] * len(self.board.bridges)
        
#         if self._backtrack_recursive(0):
#             return self.solution
#         return None

#     def _backtrack_recursive(self, edge_idx):
#         # BASE CASE: Đã duyệt hết các cạnh
#         if edge_idx == len(self.board.bridges):
#             # Kiểm tra số lượng cầu ở các đảo lần cuối
#             for island in self.board.islands:
#                 if island.current_bridges != island.number:
#                     return False
#             # Kiểm tra liên thông (chỉ check 1 lần duy nhất ở lá cây)
#             return self.board.check_connectivity(self.solution)

#         # RECURSIVE STEP
#         edge = self.board.bridges[edge_idx]
#         u, v = edge['u'], edge['v']
        
#         # Heuristic: Chỉ thử các giá trị không làm tràn số cầu của đảo
#         remaining_u = u.number - u.current_bridges
#         remaining_v = v.number - v.current_bridges
#         max_possible = min(2, remaining_u, remaining_v)
        
#         # Loop thử giá trị 0 -> max_possible
#         for val in range(max_possible + 1): 
#             # 1. Pruning: Check Crossing
#             is_conflict = False
#             if val > 0:
#                 for conflict_idx in edge['conflicts']:
#                     # Chỉ kiểm tra xung đột với các cạnh ĐÃ gán (index nhỏ hơn)
#                     if conflict_idx < edge_idx: 
#                         if self.solution[conflict_idx] > 0:
#                             is_conflict = True
#                             break
            
#             if is_conflict:
#                 continue

#             # 2. Action
#             self.solution[edge_idx] = val
#             u.current_bridges += val
#             v.current_bridges += val
            
#             # 3. Recurse
#             # (Có thể thêm pruning ở đây: kiểm tra xem đảo u đã bão hòa chưa)
#             if self._backtrack_recursive(edge_idx + 1):
#                 return True
            
#             # 4. Backtrack
#             u.current_bridges -= val
#             v.current_bridges -= val
#             self.solution[edge_idx] = 0
            
#         return False

### CÁC HÀM HỖ TRỢ ĐỌC/GHI FILE
def read_input_file(filename):
    """
    Đọc file txt và trả về ma trận 2D (list of lists).
    Hỗ trợ định dạng dấu phẩy (1,0,2) hoặc khoảng trắng (1 0 2).
    """
    grid = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                # Bỏ khoảng trắng thừa ở đầu/cuối dòng
                line = line.strip()
                if not line: continue  # Bỏ qua dòng trống
                
                # Thay thế dấu phẩy bằng khoảng trắng để xử lý thống nhất
                # Sau đó split() sẽ tách dựa trên khoảng trắng
                parts = line.replace(',', ' ').split()
                
                # Chuyển đổi thành số nguyên
                row = [int(x) for x in parts]
                grid.append(row)
        return grid
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{filename}'")
        return None
    except ValueError:
        print("Lỗi: File chứa ký tự không phải số.")
        return None

# ==========================================
# CÁC HÀM WRAPPER ĐỂ GỌI TỪ BÊN NGOÀI (Cho main.py)
# ==========================================

def run_backtracking_solver(input_path, output_path):
    """
    Hàm wrapper để chạy Backtracking cho 1 file input và lưu vào output.
    Trả về: Thời gian chạy (float) hoặc None nếu lỗi.
    """
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
    """
    Hàm wrapper để chạy Brute-force cho 1 file input và lưu vào output.
    """
    try:
        grid_data = read_input_file(input_path)
        if not grid_data: return None
        
        board = HashiBoard(grid_data)
        
        # --- CẬP NHẬT GIỚI HẠN ---
        # 3^15 ~ 14 triệu (nhanh)
        # 3^20 ~ 3.4 tỷ (rất lâu - vài chục phút đến vài giờ)
        # 3^25 ~ 847 tỷ (coi như vô tận với Python)
        # Tăng lên 25 để nó cố chạy các map 9x9 nhỏ, nhưng vẫn chặn map 20x20 (200 cạnh) để tránh crash máy.
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

# ### MAIN FUNCTION
# if __name__ == "__main__":
#     input_filename = "input_03.txt" # Nhập input file ở đây
#     output_filename = "output_03.txt" # Nhập output file ở đây
    
#     # Đọc dữ liệu
#     print(f"Đang đọc dữ liệu từ file: {input_filename} ...")
#     grid = read_input_file(input_filename)
    
#     if grid:
#         # Khởi tạo Board
#         board = HashiBoard(grid)
        
#         # --- IN THÔNG TIN BẢN ĐỒ THEO YÊU CẦU ---
#         print(f"\n--- THÔNG TIN BẢN ĐỒ ---")
#         print(f"Kích thước: {len(grid)}x{len(grid[0])}")
#         print(f"Số lượng đảo: {len(board.islands)}")
#         print(f"Số lượng cạnh tiềm năng: {len(board.bridges)}")

#         # ---------------------------------------------------------
#         # THUẬT TOÁN 1: QUAY LUI (BACKTRACKING)
#         # ---------------------------------------------------------
#         print("\n" + "="*50)
#         print(" 1. THUẬT TOÁN QUAY LUI (BACKTRACKING)")
#         print("="*50)
        
#         bt_solver = BacktrackingSolver(board)
        
#         start = time.time()
#         bt_solution = bt_solver.solve()
#         end = time.time()
        
#         if bt_solution:
#             print(f"-> Trạng thái: TÌM THẤY LỜI GIẢI")
#             print(f"-> Cấu hình cạnh: {bt_solution}")
#         else:
#             print(f"-> Trạng thái: KHÔNG TÌM THẤY")
        
#         print(f"-> Thời gian chạy: {end - start:.6f} giây")

#         # ---------------------------------------------------------
#         # THUẬT TOÁN 2: VÉT CẠN (BRUTE FORCE)
#         # ---------------------------------------------------------
#         print("\n" + "="*50)
#         print(" 2. THUẬT TOÁN VÉT CẠN (BRUTE FORCE)")
#         print("="*50)
        
#         bf_solution = None
#         # Chỉ chạy nếu số cạnh < 15 để tránh treo máy (vì 3^15 ~ 14 triệu trường hợp)
#         if len(board.bridges) < 15:
#             bf_solver = BruteForceSolver(board)
            
#             start = time.time()
#             bf_solution = bf_solver.solve()
#             end = time.time()
            
#             if bf_solution:
#                 print(f"-> Trạng thái: TÌM THẤY LỜI GIẢI")
#             else:
#                 print(f"-> Trạng thái: KHÔNG TÌM THẤY")
            
#             print(f"-> Thời gian chạy: {end - start:.6f} giây")
#         else:
#             print(f"-> Bỏ qua Brute-force vì số cạnh quá lớn ({len(board.bridges)} cạnh).")
#             print("-> Không gian tìm kiếm quá lớn để chạy thử nghiệm.")

#         # ---------------------------------------------------------
#         # XUẤT KẾT QUẢ RA FILE VÀ TERMINAL
#         # ---------------------------------------------------------
#         # Ưu tiên lấy kết quả từ Backtracking (vì nó luôn chạy được)
#         final_solution = bt_solution if bt_solution else bf_solution
        
#         if final_solution:
#             # Chuyển đổi sang dạng lưới hiển thị (String Grid)
#             visual_grid = board.export_solution_to_grid(final_solution)
            
#             print(f"\n--- KẾT QUẢ CHI TIẾT (Lưu vào {output_filename}) ---")
#             try:
#                 with open(output_filename, "w") as f:
#                     for row in visual_grid:
#                         # Format list string: ["1", "0", "="] thay vì ['1', '0', '=']
#                         formatted_row = str(row).replace("'", '"')
                        
#                         # 1. In lên Terminal
#                         print(formatted_row)
                        
#                         # 2. Ghi vào file
#                         f.write(formatted_row + "\n")
#                 print(f"\n-> Đã ghi xong file '{output_filename}'.")
#             except Exception as e:
#                 print(f"Lỗi khi ghi file: {e}")
#         else:
#             print("\n-> Không có lời giải nào để ghi vào file.")
            
#     else:
#         print("Lỗi: Không đọc được dữ liệu đầu vào.")