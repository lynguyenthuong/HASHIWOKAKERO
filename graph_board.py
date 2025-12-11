"""
File chứa Island, HashiBoard (Generator cho BruteForce/Backtracking)
"""

class Island:
    def __init__(self, r, c, number):
        self.r = r; self.c = c; self.number = number; self.current_bridges = 0
    def __repr__(self): return f"({self.r},{self.c})[{self.number}]"

class HashiBoard:
    def __init__(self, grid_data):
        self.grid = grid_data.data if hasattr(grid_data, 'data') else grid_data
        self.rows = len(self.grid); self.cols = len(self.grid[0])
        self.islands = []; self.bridges = []
        self._parse_input(); self._find_potential_bridges()

    def _parse_input(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] > 0: self.islands.append(Island(r, c, self.grid[r][c]))

    def _find_potential_bridges(self):
        potential_edges = []
        def get_neighbor(island, dr, dc):
            r, c = island.r + dr, island.c + dc
            while 0 <= r < self.rows and 0 <= c < self.cols:
                if self.grid[r][c] > 0:
                    for n in self.islands: 
                        if n.r == r and n.c == c: return n
                r += dr; c += dc
            return None

        for island in self.islands:
            nr = get_neighbor(island, 0, 1); nd = get_neighbor(island, 1, 0)
            if nr: potential_edges.append((island, nr, 'H'))
            if nd: potential_edges.append((island, nd, 'V'))

        for i, edge in enumerate(potential_edges):
            conflicts = []; u1, v1, t1 = edge
            for j, other in enumerate(potential_edges):
                if i == j: continue
                u2, v2, t2 = other
                if t1 == 'H' and t2 == 'V':
                    if (u1.c < u2.c < v1.c) and (u2.r < u1.r < v2.r): conflicts.append(j)
                elif t1 == 'V' and t2 == 'H':
                    if (u2.c < u1.c < v2.c) and (u1.r < u2.r < v1.r): conflicts.append(j)
            self.bridges.append({'u': u1, 'v': v1, 'type': t1, 'conflicts': conflicts, 'val': 0})

    def reset_state(self):
        for i in self.islands: i.current_bridges = 0
        for b in self.bridges: b['val'] = 0

    def check_connectivity(self, bridge_values):
        adj = {i: [] for i in self.islands}
        for i, val in enumerate(bridge_values):
            if val > 0:
                u = self.bridges[i]['u']; v = self.bridges[i]['v']
                adj[u].append(v); adj[v].append(u)
        if not self.islands: return True
        start = self.islands[0]; queue = [start]; visited = {start}
        while queue:
            curr = queue.pop(0)
            for n in adj[curr]:
                if n not in visited: visited.add(n); queue.append(n)
        return len(visited) == len(self.islands)
    
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