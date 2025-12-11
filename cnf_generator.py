"""
File chứa HashiCNFGenerator
"""

import itertools

### Class sinh biến và mệnh đề CNF
class HashiCNFGenerator:
    def __init__(self, grid_data):
        self.grid = grid_data
        self.rows = len(grid_data)
        self.cols = len(grid_data[0])
        self.islands = self._get_islands()
        self.potential_edges = self._get_potential_edges()
        self.var_map = {}
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
                if u['r'] == v['r']: # Ngang
                    c_min, c_max = min(u['c'], v['c']), max(u['c'], v['c'])
                    blocked = False
                    for c in range(c_min + 1, c_max):
                        if self.grid[u['r']][c] > 0: blocked = True; break
                    if not blocked:
                        edges.append({'u': i, 'v': j, 'type': 'hor', 'r': u['r'], 'c_min': c_min, 'c_max': c_max})
                elif u['c'] == v['c']: # Dọc
                    r_min, r_max = min(u['r'], v['r']), max(u['r'], v['r'])
                    blocked = False
                    for r in range(r_min + 1, r_max):
                        if self.grid[r][u['c']] > 0: blocked = True; break
                    if not blocked:
                        edges.append({'u': i, 'v': j, 'type': 'ver', 'c': u['c'], 'r_min': r_min, 'r_max': r_max})
        return edges

    def _init_variables(self):
        for edge in self.potential_edges:
            u, v = edge['u'], edge['v']
            id_1 = self.id_counter; self.var_map[(u, v, 1)] = id_1
            self.reverse_map[id_1] = {'u': u, 'v': v, 'type': 1, 'edge_idx': self.potential_edges.index(edge)}
            self.id_counter += 1
            id_2 = self.id_counter; self.var_map[(u, v, 2)] = id_2
            self.reverse_map[id_2] = {'u': u, 'v': v, 'type': 2, 'edge_idx': self.potential_edges.index(edge)}
            self.id_counter += 1

    def get_var(self, u_idx, v_idx, k):
        if u_idx > v_idx: u_idx, v_idx = v_idx, u_idx
        return self.var_map.get((u_idx, v_idx, k))

    def generate_initial_cnf(self):
        self.clauses = []
        self._gen_consistency()
        self._gen_no_crossing()
        self._gen_capacity()
        return self.clauses

    def _gen_consistency(self):
        for edge in self.potential_edges:
            u, v = edge['u'], edge['v']
            self.clauses.append([-self.get_var(u, v, 2), self.get_var(u, v, 1)])

    def _gen_no_crossing(self):
        for i in range(len(self.potential_edges)):
            for j in range(i + 1, len(self.potential_edges)):
                e1 = self.potential_edges[i]; e2 = self.potential_edges[j]
                is_crossing = False
                if e1['type'] == 'hor' and e2['type'] == 'ver':
                    if (e1['c_min'] < e2['c'] < e1['c_max']) and (e2['r_min'] < e1['r'] < e2['r_max']): is_crossing = True
                elif e1['type'] == 'ver' and e2['type'] == 'hor':
                    if (e2['c_min'] < e1['c'] < e2['c_max']) and (e1['r_min'] < e2['r'] < e1['r_max']): is_crossing = True
                if is_crossing:
                    self.clauses.append([-self.get_var(e1['u'], e1['v'], 1), -self.get_var(e2['u'], e2['v'], 1)])

    def _gen_capacity(self):
        for island in self.islands:
            u_idx = island['id']; capacity = island['val']
            neighbor_vars = []
            for edge in self.potential_edges:
                if edge['u'] == u_idx or edge['v'] == u_idx:
                    v_idx = edge['v'] if edge['u'] == u_idx else edge['u']
                    neighbor_vars.append(self.get_var(u_idx, v_idx, 1))
                    neighbor_vars.append(self.get_var(u_idx, v_idx, 2))
            
            size_S = len(neighbor_vars)
            if size_S - capacity + 1 > 0:
                for comb in itertools.combinations(neighbor_vars, size_S - capacity + 1): self.clauses.append(list(comb))
            if capacity + 1 <= size_S:
                for comb in itertools.combinations(neighbor_vars, capacity + 1): self.clauses.append([-x for x in comb])

    def decode_solution_to_edges(self, model):
        active_edges = []
        if model is None: return []
        model_set = set(model)
        for edge in self.potential_edges:
            u_idx, v_idx = edge['u'], edge['v']
            var1 = self.get_var(u_idx, v_idx, 1)
            var2 = self.get_var(u_idx, v_idx, 2)
            bridges = 2 if var2 in model_set else (1 if var1 in model_set else 0)
            if bridges > 0:
                u_coords = (self.islands[u_idx]['c'], self.islands[u_idx]['r'])
                v_coords = (self.islands[v_idx]['c'], self.islands[v_idx]['r'])
                active_edges.append({'u': u_coords, 'v': v_coords, 'bridges': bridges})
        return active_edges