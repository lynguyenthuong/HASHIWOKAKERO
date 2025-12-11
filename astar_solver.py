"""
File chứa AStarSolver, Preprocessor & Wrapper
"""

import time
import heapq
import itertools
import traceback
from collections import defaultdict
from pysat.formula import CNF
from cnf_generator import HashiCNFGenerator
from common import read_grid_from_file, format_solution_output, write_solution_to_file

class BridgeVar:
    def __init__(self, id, u, v, idx, direction):
        self.id, self.u, self.v, self.idx, self.direction = id, u, v, idx, direction

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

class AStarSolver:
    def __init__(self, grid):
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

def run_astar(grid):
    solver = AStarSolver(grid)
    return solver.solve()

def run_astar_solver(input_path, output_path):
    try:
        grid = read_grid_from_file(input_path)
        if not grid: return None
        start = time.time()
        result = run_astar(grid)
        duration = time.time() - start
        write_solution_to_file(result, output_path)
        return duration
    except Exception as e:
        print(f"AStar Error: {e}"); return None