import itertools
import heapq
import os
import time
import traceback
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
from pysat.formula import CNF
from pysat.solvers import Glucose3

### CẤU TRÚC DỮ LIỆU CƠ BẢN & INPUT (SHARED)
class Grid:
    def __init__(self, width: int, height: int, data: List[List[Any]]):
        self.width = width
        self.height = height
        self.data = data #

def read_grid_from_file(file_path: str) -> Optional[Grid]:
    try:
        matrix = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    # Hỗ trợ cả dấu phẩy và khoảng trắng
                    matrix.append([int(x) for x in line.replace(',', ' ').split()])
        if not matrix:
            return None
        return Grid(len(matrix[0]), len(matrix), matrix)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def format_solution_output(grid: Grid, solution_edges: List[dict]):
    rows = grid.height
    cols = grid.width
    out_grid = [["0" for _ in range(cols)] for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            if grid.data[r][c] > 0:
                out_grid[r][c] = str(grid.data[r][c])

    for edge in solution_edges:
        if 'info' in edge: 
            u_r, u_c = edge['info']['r'], edge['info']['c_min'] 
            pass

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

### GENERATOR & CNF ENCODING
# Class sinh biến và mệnh đề CNF (Dùng chung cho cả PySAT và A*)
class HashiCNFGenerator:
    def __init__(self, grid_data):
        self.grid = grid_data
        self.rows = len(grid_data)
        self.cols = len(grid_data[0])
        self.islands = self._get_islands()
        self.potential_edges = self._get_potential_edges()

        # Mapping biến: (u_index, v_index, k) -> int ID
        self.var_map = {}
        # Mapping ngược: ID -> thông tin cạnh
        self.reverse_map = {}

        self.id_counter = 1
        self._init_variables()
        self.clauses = [] 

    def _get_islands(self):
        islands = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] > 0:
                    islands.append({'r': r, 'c': c, 'val': self.grid[r][c], 'id': len(islands)})
        return islands

    def _get_potential_edges(self):
        edges = []
        for i in range(len(self.islands)):
            for j in range(i + 1, len(self.islands)):
                u = self.islands[i]
                v = self.islands[j]

                # Cùng hàng
                if u['r'] == v['r']:
                    c_min, c_max = min(u['c'], v['c']), max(u['c'], v['c'])
                    blocked = False
                    for c in range(c_min + 1, c_max):
                        if self.grid[u['r']][c] > 0:
                            blocked = True
                            break
                    if not blocked:
                        edges.append({'u': i, 'v': j, 'type': 'hor', 
                                      'r': u['r'], 'c_min': c_min, 'c_max': c_max})

                # Cùng cột
                elif u['c'] == v['c']:
                    r_min, r_max = min(u['r'], v['r']), max(u['r'], v['r'])
                    blocked = False
                    for r in range(r_min + 1, r_max):
                        if self.grid[r][u['c']] > 0:
                            blocked = True
                            break
                    if not blocked:
                        edges.append({'u': i, 'v': j, 'type': 'ver', 
                                      'c': u['c'], 'r_min': r_min, 'r_max': r_max})
        return edges

    def _init_variables(self):
        for edge in self.potential_edges:
            u, v = edge['u'], edge['v']
            # Biến x_{u,v,1}: ID lẻ
            id_1 = self.id_counter
            self.var_map[(u, v, 1)] = id_1
            self.reverse_map[id_1] = {'u': u, 'v': v, 'type': 1, 'edge_idx': self.potential_edges.index(edge)}
            self.id_counter += 1
            
            # Biến x_{u,v,2}: ID chẵn
            id_2 = self.id_counter
            self.var_map[(u, v, 2)] = id_2
            self.reverse_map[id_2] = {'u': u, 'v': v, 'type': 2, 'edge_idx': self.potential_edges.index(edge)}
            self.id_counter += 1

    def get_var(self, u_idx, v_idx, k):
        if u_idx > v_idx: u_idx, v_idx = v_idx, u_idx
        return self.var_map.get((u_idx, v_idx, k))

    # Hàm sinh tất cả các ràng buộc tĩnh (Consistency, Crossing, Capacity).
    def generate_initial_cnf(self):
        self.clauses = []
        self._gen_consistency()
        self._gen_no_crossing()
        self._gen_capacity()
        return self.clauses

    def _gen_consistency(self):
        for edge in self.potential_edges:
            u, v = edge['u'], edge['v']
            x1 = self.get_var(u, v, 1)
            x2 = self.get_var(u, v, 2)
            self.clauses.append([-x2, x1])

    def _gen_no_crossing(self):
        for i in range(len(self.potential_edges)):
            for j in range(i + 1, len(self.potential_edges)):
                e1 = self.potential_edges[i]
                e2 = self.potential_edges[j]
                is_crossing = False
                if e1['type'] == 'hor' and e2['type'] == 'ver':
                    if (e1['c_min'] < e2['c'] < e1['c_max']) and (e2['r_min'] < e1['r'] < e2['r_max']):
                        is_crossing = True
                elif e1['type'] == 'ver' and e2['type'] == 'hor':
                    if (e2['c_min'] < e1['c'] < e2['c_max']) and (e1['r_min'] < e2['r'] < e1['r_max']):
                        is_crossing = True
                if is_crossing:
                    var1 = self.get_var(e1['u'], e1['v'], 1)
                    var2 = self.get_var(e2['u'], e2['v'], 1)
                    self.clauses.append([-var1, -var2])

    def _gen_capacity(self):
        for island in self.islands:
            u_idx = island['id']
            capacity = island['val']
            neighbor_vars = []
            for edge in self.potential_edges:
                if edge['u'] == u_idx or edge['v'] == u_idx:
                    v_idx = edge['v'] if edge['u'] == u_idx else edge['u']
                    neighbor_vars.append(self.get_var(u_idx, v_idx, 1))
                    neighbor_vars.append(self.get_var(u_idx, v_idx, 2))
            
            size_S = len(neighbor_vars)
            # At-Least
            subset_size_at_least = size_S - capacity + 1
            if subset_size_at_least > 0:
                for combination in itertools.combinations(neighbor_vars, subset_size_at_least):
                    self.clauses.append(list(combination))
            # At-Most
            subset_size_at_most = capacity + 1
            if subset_size_at_most <= size_S:
                for combination in itertools.combinations(neighbor_vars, subset_size_at_most):
                    self.clauses.append([-x for x in combination])

    # Chuyển model PySAT thành danh sách cạnh chuẩn để vẽ
    def decode_solution_to_edges(self, model):
        active_edges = []
        if model is None: return []
        model_set = set(model)
        
        for edge in self.potential_edges:
            u_idx, v_idx = edge['u'], edge['v']
            var1 = self.get_var(u_idx, v_idx, 1)
            var2 = self.get_var(u_idx, v_idx, 2)
            
            bridges = 0
            if var2 in model_set: bridges = 2
            elif var1 in model_set: bridges = 1
            
            if bridges > 0:
                # Chuyển đổi sang tọa độ (col, row)
                u_coords = (self.islands[u_idx]['c'], self.islands[u_idx]['r'])
                v_coords = (self.islands[v_idx]['c'], self.islands[v_idx]['r'])
                active_edges.append({'u': u_coords, 'v': v_coords, 'bridges': bridges})
        return active_edges

### GIẢI BẰNG PYSAT 
def check_connectivity_bfs(islands, active_edges_indices):
    """BFS kiểm tra liên thông dựa trên index đảo"""
    if not islands: return True, []
    adj = defaultdict(list)
    for edge in active_edges_indices:
        adj[edge['u']].append(edge['v'])
        adj[edge['v']].append(edge['u'])
        
    start_node = 0
    visited = set([start_node])
    queue = [start_node]
    while queue:
        node = queue.pop(0)
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    if len(visited) == len(islands):
        return True, []
    else:
        return False, list(visited)

def run_pysat(grid: Grid):
    generator = HashiCNFGenerator(grid.data)
    clauses = generator.generate_initial_cnf()
    
    solver = Glucose3()
    for clause in clauses:
        solver.add_clause(clause)
        
    loop_count = 0
    while True:
        loop_count += 1
        is_sat = solver.solve()
        
        if not is_sat:
            return None # UNSAT
        
        model = solver.get_model()
        
        # Decode ra danh sách cạnh (dùng index để check liên thông)
        active_edges_indices = []
        model_set = set(model)
        for edge in generator.potential_edges:
            u, v = edge['u'], edge['v']
            if generator.get_var(u, v, 1) in model_set:
                active_edges_indices.append({'u': u, 'v': v})
        
        # Kiểm tra liên thông
        is_connected, visited_group = check_connectivity_bfs(generator.islands, active_edges_indices)
        
        if is_connected:
            # Tìm thấy nghiệm hoàn chỉnh
            solution_edges = generator.decode_solution_to_edges(model)
            out_grid = format_solution_output(grid, solution_edges)
            return out_grid
        
        # Thêm Lazy Constraint (Cut-set)
        visited_set = set(visited_group)
        cut_clause = []
        for edge in generator.potential_edges:
            u, v = edge['u'], edge['v']
            # Nếu cạnh nối 1 đỉnh thuộc nhóm visited và 1 đỉnh ko thuộc
            if (u in visited_set) != (v in visited_set):
                cut_clause.append(generator.get_var(u, v, 1))
        
        if not cut_clause:
            return None # Lỗi: Đồ thị rời rạc nhưng ko có cạnh nối tiềm năng
        
        solver.add_clause(cut_clause)

### GIẢI BẰNG A*
class BridgeVar:
    def __init__(self, id, u, v, idx, direction):
        self.id, self.u, self.v, self.idx, self.direction = id, u, v, idx, direction

# Class tiền xử lý cho A* để giảm không gian tìm kiếm
class HashiPreprocessor:
    def __init__(self, grid_data, islands):
        self.grid = grid_data
        self.rows = len(grid_data)
        self.cols = len(grid_data[0])
        self.island_map = {(isl['r'], isl['c']): isl for isl in islands}
        self.islands_list = islands
        self.rem_cap = {(isl['r'], isl['c']): isl['val'] for isl in islands}
        self.bridges = {}
        self.neighbors = self._find_all_neighbors()
        self.id_to_coords = {isl['id']: (isl['r'], isl['c']) for isl in islands}

    def _find_all_neighbors(self):
        neighbors = {}
        for isl in self.islands_list:
            r, c = isl['r'], isl['c']
            neighbors[isl['id']] = []
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                while 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.grid[nr][nc] > 0:
                        neighbor_isl = self.island_map[(nr, nc)]
                        neighbors[isl['id']].append(neighbor_isl['id'])
                        break
                    if self.grid[nr][nc] == -1: break 
                    nr += dr
                    nc += dc
        return neighbors

    def _get_current_bridge_count(self, u_id, v_id):
        u, v = sorted((u_id, v_id))
        return self.bridges.get((u, v), 0)

    def _add_bridge(self, u_id, v_id, count=1):
        u, v = sorted((u_id, v_id))
        current = self.bridges.get((u, v), 0)
        if current >= 2: return False

        u_isl = self.islands_list[u]
        v_isl = self.islands_list[v]

        rem_u = self.rem_cap[(u_isl['r'], u_isl['c'])]
        rem_v = self.rem_cap[(v_isl['r'], v_isl['c'])]

        actual_add = min(2 - current, rem_u, rem_v, count)

        if actual_add > 0:
            self.bridges[(u, v)] = current + actual_add
            self.rem_cap[(u_isl['r'], u_isl['c'])] -= actual_add
            self.rem_cap[(v_isl['r'], v_isl['c'])] -= actual_add
            return True
        return False

    def run(self):
        changed = True
        loops = 0
        while changed and loops < 100:
            changed = False
            loops += 1

            current_islands = list(self.islands_list)
            for isl in current_islands:
                uid = isl['id']
                r, c = isl['r'], isl['c']
                rem = self.rem_cap[(r, c)]
                if rem <= 0: continue

                my_neighbors = self.neighbors[uid]
                neighbor_stats = []
                valid_neighbors_ids = []

                for nid in my_neighbors:
                    n_isl = self.islands_list[nid]
                    n_rem = self.rem_cap[(n_isl['r'], n_isl['c'])]
                    curr_b = self._get_current_bridge_count(uid, nid)
                    
                    can_accept = (curr_b < 2) and (n_rem > 0)
                    if can_accept: valid_neighbors_ids.append(nid)
                    
                    space = 0
                    if can_accept: space = min(2 - curr_b, n_rem)
                    neighbor_stats.append({'nid': nid, 'space': space})

                # RULE 1: Neighborhood Capacity
                total_possible_space = sum(n['space'] for n in neighbor_stats)
                if rem == total_possible_space and rem > 0:
                    for item in neighbor_stats:
                        if item['space'] > 0:
                            if self._add_bridge(uid, item['nid'], item['space']):
                                changed = True

                # RULE 2: Must-have Bridges
                if len(valid_neighbors_ids) > 0:
                    for target_nid in valid_neighbors_ids:
                        max_others = 0
                        for item in neighbor_stats:
                            if item['nid'] != target_nid: max_others += item['space']
                        
                        must_have_total = rem - max_others
                        curr_b = self._get_current_bridge_count(uid, target_nid)
                        needed_new = must_have_total - curr_b
                        
                        if needed_new > 0:
                            if self._add_bridge(uid, target_nid, needed_new):
                                changed = True
                                
        return self.bridges

# Class giải A* sử dụng MCI Heuristic trên CNF
class AStarSolver:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.generator = None
        self.cnf = CNF()
        self.edge_vars = {}
        self.var_index_map = {}
        self.all_vars = []
        self.islands_map = {}
        self.island_edge_map = defaultdict(list)
        self.nodes_expanded = 0

    # Map variables và tạo cấu trúc dữ liệu cho MCI.
    def _setup_variables(self):
        self.island_edge_map = defaultdict(list) 

        for var_id, info in self.generator.reverse_map.items():
            u_idx, v_idx = info['u'], info['v']
            u_isl = self.generator.islands[u_idx]
            v_isl = self.generator.islands[v_idx]
            u_coords = (u_isl['c'], u_isl['r'])
            v_coords = (v_isl['c'], v_isl['r'])
            
            direction = 'h' if u_coords[1] == v_coords[1] else 'v'
            
            self.edge_vars[var_id] = BridgeVar(
                var_id, u_coords, v_coords, info['type'], direction
            )

        self.islands_map = {(isl['c'], isl['r']): isl['val'] for isl in self.generator.islands}
        # Lấy tất cả biến từ CNF
        self.all_vars = sorted(list(set(abs(l) for c in self.cnf.clauses for l in c)))
        self.var_index_map = {var: i for i, var in enumerate(self.all_vars)}

        for var_id, bvar in self.edge_vars.items():
            if var_id in self.var_index_map:
                idx = self.var_index_map[var_id]
                self.island_edge_map[bvar.u].append(idx)
                self.island_edge_map[bvar.v].append(idx)

    # Đếm số mệnh đề chưa thỏa mãn.
    def _compute_heuristic(self, assignment):
        h = 0
        for clause in self.cnf.clauses:
            clause_sat = False
            all_assigned = True
            for lit in clause:
                var = abs(lit)
                idx = self.var_index_map.get(var)
                if idx is None:
                    all_assigned = False
                    continue
                val = assignment[idx]
                if val is None:
                    all_assigned = False
                elif (lit > 0 and val) or (lit < 0 and not val):
                    clause_sat = True
                    break

            if all_assigned and not clause_sat:
                h += 100 # Phạt nặng nếu vi phạm mệnh đề
            elif not clause_sat:
                h += 1
        return h

    def _clause_status(self, clause, assignment):
        unassigned = []
        for lit in clause:
            var = abs(lit)
            if var not in self.var_index_map: continue
            idx = self.var_index_map[var]
            val = assignment[idx]

            if val is not None:
                if (lit > 0 and val) or (lit < 0 and not val):
                    return "satisfied", None
            else:
                unassigned.append((var, lit))

        if not unassigned: return "conflict", None
        if len(unassigned) == 1: return "unit", unassigned[0]
        return "open", None

    def _unit_propagate(self, assignment):
        changed = True
        while changed:
            changed = False
            for clause in self.cnf.clauses:
                status, unit_lit = self._clause_status(clause, assignment)
                if status == "conflict": return None
                if status == "unit":
                    var, lit = unit_lit
                    idx = self.var_index_map[var]
                    val = (lit > 0)
                    if assignment[idx] is None:
                        assignment[idx] = val
                        changed = True
        return assignment

    # Heuristic MCI: Chọn biến thuộc về đảo khó thỏa mãn nhất
    def _get_mci_variable(self, assignment):
        best_island_vars = None
        min_gap = float('inf')
        found_candidate = False

        for (c, r), target_val in self.islands_map.items():
            connected_indices = self.island_edge_map.get((c, r), [])
            temp_edge_status = defaultdict(int)
            unassigned_vars = []
            
            for idx in connected_indices:
                var_id = self.all_vars[idx]
                bvar = self.edge_vars[var_id]
                pair = tuple(sorted((bvar.u, bvar.v)))
                val = assignment[idx]
                
                if val is True:
                    temp_edge_status[pair] = max(temp_edge_status[pair], bvar.idx)
                elif val is None:
                    if temp_edge_status[pair] != -1:
                        unassigned_vars.append(idx)
            
            current_degree = sum(v for v in temp_edge_status.values() if v > 0)
            remaining = target_val - current_degree
            
            if remaining <= 0: continue
            if not unassigned_vars: continue
            
            gap = len(unassigned_vars) - remaining
            if gap < min_gap:
                min_gap = gap
                best_island_vars = unassigned_vars
                found_candidate = True
        
        if found_candidate and best_island_vars:
            return best_island_vars[0]
            
        # Fallback
        for idx, val in enumerate(assignment):
            if val is None: return idx
        return None

    def _check_full_assignment(self, assignment):
        for clause in self.cnf.clauses:
            sat = False
            for lit in clause:
                var = abs(lit)
                if var in self.var_index_map:
                    val = assignment[self.var_index_map[var]]
                    if (lit > 0 and val) or (lit < 0 and not val):
                        sat = True; break
            if not sat: return False
        return True

    def _extract_solution_edges(self, model):
        solution = []
        active = set(x for x in model if x > 0)
        final_edges = {}
        for vid, bvar in self.edge_vars.items():
            pair = tuple(sorted((bvar.u, bvar.v)))
            if pair not in final_edges: final_edges[pair] = 0
            if vid in active: final_edges[pair] = max(final_edges[pair], bvar.idx)
        
        for (u, v), count in final_edges.items():
            if count > 0:
                solution.append({'u': u, 'v': v, 'bridges': count})
        return solution
    
    def _validate_solution_degree(self, model):
        # Kiểm tra lại số bậc đảo để chắc chắn
        active = set(x for x in model if x > 0)
        degrees = defaultdict(int)
        edge_states = {}
        for vid, bvar in self.edge_vars.items():
            pair = tuple(sorted((bvar.u, bvar.v)))
            if pair not in edge_states: edge_states[pair] = 0
            if vid in active: edge_states[pair] = max(edge_states[pair], bvar.idx)
        for (u, v), count in edge_states.items():
            degrees[u] += count
            degrees[v] += count
        for pos, target in self.islands_map.items():
            if degrees[pos] != target: return False
        return True

    def _check_connectivity_astar(self, sol_edges):
        if not self.islands_map: return True
        adj = defaultdict(list)
        for e in sol_edges:
            adj[e['u']].append(e['v'])
            adj[e['v']].append(e['u'])
        start = next(iter(self.islands_map))
        visited = {start}
        queue = [start]
        while queue:
            curr = queue.pop(0)
            for n in adj[curr]:
                if n not in visited: visited.add(n); queue.append(n)
        return len(visited) == len(self.islands_map)

    def solve(self):
        try:
            # Initialize & Preprocess
            self.generator = HashiCNFGenerator(self.grid.data)
            preprocessor = HashiPreprocessor(self.grid.data, self.generator.islands)
            fixed_bridges = preprocessor.run()
            
            # Generate CNF
            self.cnf.extend(self.generator.generate_initial_cnf())
            for (u, v), count in fixed_bridges.items():
                var1 = self.generator.get_var(u, v, 1)
                if var1: self.cnf.append([var1])
                if count == 2:
                    var2 = self.generator.get_var(u, v, 2)
                    if var2: self.cnf.append([var2])

            # Setup Variables
            self._setup_variables()

            initial_assignment = [None] * len(self.all_vars)
            initial_assignment = self._unit_propagate(initial_assignment)
            
            if initial_assignment is None: return None

            # A* Search
            h_initial = self._compute_heuristic(initial_assignment)
            g_initial = 0
            f_initial = g_initial + h_initial
            
            counter = itertools.count()
            open_list = []
            heapq.heappush(open_list, (f_initial, g_initial, next(counter), initial_assignment))

            while open_list:
                f, g, _, assignment = heapq.heappop(open_list)
                self.nodes_expanded += 1

                var_idx = self._get_mci_variable(assignment)

                if var_idx is None:
                    if self._check_full_assignment(assignment):
                        model = []
                        for var, idx in self.var_index_map.items():
                            if assignment[idx] is True: model.append(var)
                            elif assignment[idx] is False: model.append(-var)
                        
                        if self._validate_solution_degree(model):
                            sol = self._extract_solution_edges(model)
                            if self._check_connectivity_astar(sol):
                                return format_solution_output(self.grid, sol)
                    continue

                new_g = g + 1
                
                # Branch False
                new_assign_false = assignment[:]
                new_assign_false[var_idx] = False
                prop_false = self._unit_propagate(new_assign_false)
                if prop_false is not None:
                    h_val = self._compute_heuristic(prop_false)
                    heapq.heappush(open_list, (new_g + h_val, new_g, next(counter), prop_false))

                # Branch True
                new_assign_true = assignment[:]
                new_assign_true[var_idx] = True
                prop_true = self._unit_propagate(new_assign_true)
                if prop_true is not None:
                    h_val = self._compute_heuristic(prop_true)
                    heapq.heappush(open_list, (new_g + h_val, new_g, next(counter), prop_true))

                # Timeout guard (ví dụ: > 20000 nodes thì dừng để demo)
                if self.nodes_expanded > 50000:
                    return None

            return None # No solution found

        except Exception as e:
            traceback.print_exc()
            return None

def run_astar(grid: Grid):
    solver = AStarSolver(grid)
    result = solver.solve()
    return result

### CÁC HÀM WRAPPER (IO & TIMING)
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

def run_pysat_solver(input_path, output_path):
    try:
        grid = read_grid_from_file(input_path)
        if not grid: return None
        
        start = time.time()
        result_grid = run_pysat(grid)
        duration = time.time() - start
        
        write_solution_to_file(result_grid, output_path)
        return duration
    except Exception as e:
        print(f"PySAT Error at {input_path}: {e}")
        return None

def run_astar_solver(input_path, output_path):
    try:
        grid = read_grid_from_file(input_path)
        if not grid: return None
        
        start = time.time()
        result_grid = run_astar(grid)
        duration = time.time() - start
        
        write_solution_to_file(result_grid, output_path)
        return duration
    except Exception as e:
        print(f"AStar Error at {input_path}: {e}")
        return None