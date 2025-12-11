"""
File chứa logic PySAT & Wrapper
"""

import time
from collections import defaultdict
from pysat.solvers import Glucose3
from cnf_generator import HashiCNFGenerator
from common import read_grid_from_file, format_solution_output, write_solution_to_file

def check_connectivity_bfs(islands, active_edges_indices):
    if not islands: return True, []
    adj = defaultdict(list)
    for edge in active_edges_indices:
        adj[edge['u']].append(edge['v']); adj[edge['v']].append(edge['u'])
    start = 0; visited = {start}; queue = [start]
    while queue:
        node = queue.pop(0)
        for neighbor in adj[node]:
            if neighbor not in visited: visited.add(neighbor); queue.append(neighbor)
    return len(visited) == len(islands), list(visited)

def run_pysat(grid):
    generator = HashiCNFGenerator(grid.data)
    clauses = generator.generate_initial_cnf()
    solver = Glucose3()
    for clause in clauses: solver.add_clause(clause)
    
    while True:
        if not solver.solve(): return None
        model = solver.get_model()
        
        # Check connectivity
        active_edges_indices = []
        model_set = set(model)
        for edge in generator.potential_edges:
            if generator.get_var(edge['u'], edge['v'], 1) in model_set:
                active_edges_indices.append({'u': edge['u'], 'v': edge['v']})
        
        is_connected, visited_group = check_connectivity_bfs(generator.islands, active_edges_indices)
        if is_connected:
            solution_edges = generator.decode_solution_to_edges(model)
            return format_solution_output(grid, solution_edges)
        
        # Lazy constraint
        visited_set = set(visited_group)
        cut_clause = []
        for edge in generator.potential_edges:
            if (edge['u'] in visited_set) != (edge['v'] in visited_set):
                cut_clause.append(generator.get_var(edge['u'], edge['v'], 1))
        if not cut_clause: return None
        solver.add_clause(cut_clause)

def run_pysat_solver(input_path, output_path):
    try:
        grid = read_grid_from_file(input_path)
        if not grid: return None
        start = time.time()
        result = run_pysat(grid)
        duration = time.time() - start
        write_solution_to_file(result, output_path)
        return duration
    except Exception as e:
        print(f"PySAT Error: {e}"); return None