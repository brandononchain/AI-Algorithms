import numpy as np
import matplotlib.pyplot as plt
import heapq
from typing import List, Tuple, Set, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import random

class CellType(Enum):
    """Enumeration for different cell types in the grid."""
    EMPTY = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3
    PATH = 4
    EXPLORED = 5

@dataclass
class Node:
    """
    Node class for A* search.
    
    Attributes:
        position (Tuple[int, int]): (row, col) position in grid
        g_cost (float): Cost from start to this node
        h_cost (float): Heuristic cost from this node to goal
        parent (Optional[Node]): Parent node in the path
    """
    position: Tuple[int, int]
    g_cost: float = 0.0
    h_cost: float = 0.0
    parent: Optional['Node'] = None
    
    @property
    def f_cost(self) -> float:
        """Total cost (g + h)."""
        return self.g_cost + self.h_cost
    
    def __lt__(self, other: 'Node') -> bool:
        """For priority queue comparison."""
        return self.f_cost < other.f_cost
    
    def __eq__(self, other: 'Node') -> bool:
        """For node comparison."""
        return self.position == other.position
    
    def __hash__(self) -> int:
        """For using nodes in sets."""
        return hash(self.position)

class AStar:
    """
    A* pathfinding algorithm implementation.
    
    Attributes:
        grid (np.ndarray): 2D grid representing the environment
        start (Tuple[int, int]): Starting position
        goal (Tuple[int, int]): Goal position
        heuristic (Callable): Heuristic function
        allow_diagonal (bool): Whether diagonal movement is allowed
        explored_nodes (Set[Tuple[int, int]]): Set of explored positions
        path (List[Tuple[int, int]]): Final path from start to goal
    """
    
    def __init__(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 heuristic: str = 'manhattan', allow_diagonal: bool = False):
        self.grid = grid.copy()
        self.start = start
        self.goal = goal
        self.allow_diagonal = allow_diagonal
        self.explored_nodes = set()
        self.path = []
        
        # Set heuristic function
        heuristic_functions = {
            'manhattan': self._manhattan_distance,
            'euclidean': self._euclidean_distance,
            'chebyshev': self._chebyshev_distance,
            'octile': self._octile_distance
        }
        
        if heuristic not in heuristic_functions:
            raise ValueError(f"Heuristic must be one of {list(heuristic_functions.keys())}")
        
        self.heuristic = heuristic_functions[heuristic]
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _chebyshev_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Chebyshev distance between two positions."""
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))
    
    def _octile_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Octile distance between two positions."""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)
    
    def _get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get valid neighbors of a position with their movement costs.
        
        Args:
            position (Tuple[int, int]): Current position
            
        Returns:
            List[Tuple[Tuple[int, int], float]]: List of (neighbor_position, cost) tuples
        """
        row, col = position
        neighbors = []
        
        # 4-directional movement
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
        # Add diagonal directions if allowed
        if self.allow_diagonal:
            directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])  # diagonals
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds
            if (0 <= new_row < self.grid.shape[0] and 
                0 <= new_col < self.grid.shape[1]):
                
                # Check if cell is not an obstacle
                if self.grid[new_row, new_col] != CellType.OBSTACLE.value:
                    # Calculate movement cost
                    cost = np.sqrt(2) if abs(dr) + abs(dc) == 2 else 1.0
                    neighbors.append(((new_row, new_col), cost))
        
        return neighbors
    
    def _reconstruct_path(self, node: Node) -> List[Tuple[int, int]]:
        """
        Reconstruct path from goal to start.
        
        Args:
            node (Node): Goal node
            
        Returns:
            List[Tuple[int, int]]: Path from start to goal
        """
        path = []
        current = node
        
        while current is not None:
            path.append(current.position)
            current = current.parent
        
        return path[::-1]  # Reverse to get start to goal
    
    def search(self) -> Tuple[List[Tuple[int, int]], Dict[str, any]]:
        """
        Perform A* search to find optimal path.
        
        Returns:
            Tuple[List[Tuple[int, int]], Dict]: Path and search statistics
        """
        # Initialize
        open_set = []  # Priority queue
        closed_set = set()  # Explored nodes
        g_costs = {}  # Best known g_cost for each position
        
        # Create start node
        start_node = Node(self.start, 0, self.heuristic(self.start, self.goal))
        heapq.heappush(open_set, start_node)
        g_costs[self.start] = 0
        
        nodes_explored = 0
        max_open_set_size = 0
        
        while open_set:
            max_open_set_size = max(max_open_set_size, len(open_set))
            
            # Get node with lowest f_cost
            current = heapq.heappop(open_set)
            nodes_explored += 1
            
            # Check if we reached the goal
            if current.position == self.goal:
                self.path = self._reconstruct_path(current)
                
                # Mark explored nodes
                for pos in closed_set:
                    if pos not in [self.start, self.goal] and pos not in self.path:
                        self.explored_nodes.add(pos)
                
                stats = {
                    'nodes_explored': nodes_explored,
                    'path_length': len(self.path),
                    'path_cost': current.g_cost,
                    'max_open_set_size': max_open_set_size
                }
                
                return self.path, stats
            
            # Add to closed set
            closed_set.add(current.position)
            
            # Explore neighbors
            for neighbor_pos, move_cost in self._get_neighbors(current.position):
                if neighbor_pos in closed_set:
                    continue
                
                tentative_g_cost = current.g_cost + move_cost
                
                # If we found a better path to this neighbor
                if (neighbor_pos not in g_costs or 
                    tentative_g_cost < g_costs[neighbor_pos]):
                    
                    g_costs[neighbor_pos] = tentative_g_cost
                    h_cost = self.heuristic(neighbor_pos, self.goal)
                    
                    neighbor_node = Node(
                        neighbor_pos, 
                        tentative_g_cost, 
                        h_cost, 
                        current
                    )
                    
                    heapq.heappush(open_set, neighbor_node)
        
        # No path found
        stats = {
            'nodes_explored': nodes_explored,
            'path_length': 0,
            'path_cost': float('inf'),
            'max_open_set_size': max_open_set_size
        }
        
        return [], stats
    
    def visualize_search(self, title: str = "A* Search Result") -> None:
        """
        Visualize the search result.
        
        Args:
            title (str): Plot title
        """
        # Create visualization grid
        vis_grid = self.grid.copy().astype(float)
        
        # Mark explored nodes
        for pos in self.explored_nodes:
            vis_grid[pos] = CellType.EXPLORED.value
        
        # Mark path
        for pos in self.path:
            if pos not in [self.start, self.goal]:
                vis_grid[pos] = CellType.PATH.value
        
        # Mark start and goal
        vis_grid[self.start] = CellType.START.value
        vis_grid[self.goal] = CellType.GOAL.value
        
        # Create color map
        colors = ['white', 'black', 'green', 'red', 'blue', 'lightblue']
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(vis_grid, cmap=cmap, interpolation='nearest')
        
        # Add grid lines
        plt.xticks(range(vis_grid.shape[1]))
        plt.yticks(range(vis_grid.shape[0]))
        plt.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='white', edgecolor='black', label='Empty'),
            plt.Rectangle((0,0),1,1, facecolor='black', label='Obstacle'),
            plt.Rectangle((0,0),1,1, facecolor='green', label='Start'),
            plt.Rectangle((0,0),1,1, facecolor='red', label='Goal'),
            plt.Rectangle((0,0),1,1, facecolor='blue', label='Path'),
            plt.Rectangle((0,0),1,1, facecolor='lightblue', label='Explored')
        ]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.title(title)
        plt.tight_layout()
        plt.show()

def create_sample_grids() -> List[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], str]]:
    """Create sample grids for testing."""
    grids = []
    
    # Grid 1: Simple maze
    grid1 = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    grids.append((grid1, (0, 0), (9, 9), "Simple Maze"))
    
    # Grid 2: Random obstacles
    grid2 = np.zeros((15, 15))
    random.seed(42)
    for _ in range(40):
        row = random.randint(0, 14)
        col = random.randint(0, 14)
        if (row, col) not in [(0, 0), (14, 14)]:
            grid2[row, col] = 1
    grids.append((grid2, (0, 0), (14, 14), "Random Obstacles"))
    
    # Grid 3: Corridor maze
    grid3 = np.ones((20, 20))
    # Create corridors
    for i in range(1, 19, 2):
        grid3[i, :] = 0
    for j in range(1, 19, 2):
        grid3[:, j] = 0
    # Add some connections
    grid3[2, 2] = 0
    grid3[4, 6] = 0
    grid3[8, 10] = 0
    grid3[12, 14] = 0
    grids.append((grid3, (1, 1), (18, 18), "Corridor Maze"))
    
    return grids

def compare_heuristics(grid: np.ndarray, start: Tuple[int, int], 
                      goal: Tuple[int, int]) -> Dict[str, Dict]:
    """Compare different heuristic functions."""
    heuristics = ['manhattan', 'euclidean', 'chebyshev', 'octile']
    results = {}
    
    print("Comparing Heuristic Functions:")
    print("-" * 50)
    
    for heuristic in heuristics:
        astar = AStar(grid, start, goal, heuristic=heuristic, allow_diagonal=True)
        path, stats = astar.search()
        
        results[heuristic] = {
            'path': path,
            'stats': stats,
            'astar_object': astar
        }
        
        print(f"{heuristic.title():10} | "
              f"Path Length: {stats['path_length']:3} | "
              f"Path Cost: {stats['path_cost']:6.2f} | "
              f"Nodes Explored: {stats['nodes_explored']:4}")
    
    return results

def analyze_diagonal_movement(grid: np.ndarray, start: Tuple[int, int], 
                            goal: Tuple[int, int]) -> None:
    """Analyze the effect of allowing diagonal movement."""
    print("\nComparing Movement Types:")
    print("-" * 40)
    
    movement_types = [
        (False, "4-directional"),
        (True, "8-directional (with diagonals)")
    ]
    
    plt.figure(figsize=(15, 6))
    
    for i, (allow_diagonal, description) in enumerate(movement_types, 1):
        astar = AStar(grid, start, goal, heuristic='octile', 
                     allow_diagonal=allow_diagonal)
        path, stats = astar.search()
        
        print(f"{description:30} | "
              f"Path Length: {stats['path_length']:3} | "
              f"Path Cost: {stats['path_cost']:6.2f} | "
              f"Nodes Explored: {stats['nodes_explored']:4}")
        
        # Visualize
        plt.subplot(1, 2, i)
        
        vis_grid = grid.copy().astype(float)
        
        # Mark explored nodes
        for pos in astar.explored_nodes:
            vis_grid[pos] = 5
        
        # Mark path
        for pos in path:
            if pos not in [start, goal]:
                vis_grid[pos] = 4
        
        vis_grid[start] = 2
        vis_grid[goal] = 3
        
        colors = ['white', 'black', 'green', 'red', 'blue', 'lightblue']
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors)
        
        plt.imshow(vis_grid, cmap=cmap, interpolation='nearest')
        plt.title(f"{description}\nCost: {stats['path_cost']:.2f}, "
                 f"Explored: {stats['nodes_explored']}")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def performance_benchmark(grid_sizes: List[int], obstacle_density: float = 0.2) -> None:
    """Benchmark A* performance on different grid sizes."""
    print("\nPerformance Benchmark:")
    print("-" * 40)
    
    results = []
    
    for size in grid_sizes:
        # Create random grid
        grid = np.zeros((size, size))
        random.seed(42)
        
        # Add obstacles
        num_obstacles = int(size * size * obstacle_density)
        for _ in range(num_obstacles):
            row = random.randint(0, size - 1)
            col = random.randint(0, size - 1)
            if (row, col) not in [(0, 0), (size-1, size-1)]:
                grid[row, col] = 1
        
        # Run A*
        astar = AStar(grid, (0, 0), (size-1, size-1), heuristic='manhattan')
        
        import time
        start_time = time.time()
        path, stats = astar.search()
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        results.append({
            'size': size,
            'time': execution_time,
            'nodes_explored': stats['nodes_explored'],
            'path_found': len(path) > 0
        })
        
        print(f"Grid {size:2}x{size:2} | "
              f"Time: {execution_time:6.3f}s | "
              f"Explored: {stats['nodes_explored']:5} | "
              f"Path Found: {len(path) > 0}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    sizes = [r['size'] for r in results]
    times = [r['time'] for r in results]
    plt.plot(sizes, times, 'o-', linewidth=2)
    plt.xlabel('Grid Size')
    plt.ylabel('Execution Time (s)')
    plt.title('A* Performance vs Grid Size')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    nodes = [r['nodes_explored'] for r in results]
    plt.plot(sizes, nodes, 's-', linewidth=2, color='orange')
    plt.xlabel('Grid Size')
    plt.ylabel('Nodes Explored')
    plt.title('Search Space vs Grid Size')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Demonstrate A* search algorithm."""
    print("A* Search Algorithm Demonstration")
    print("=" * 50)
    
    # Create sample grids
    grids = create_sample_grids()
    
    for i, (grid, start, goal, name) in enumerate(grids, 1):
        print(f"\nGrid {i}: {name}")
        print(f"Size: {grid.shape}")
        print(f"Start: {start}, Goal: {goal}")
        
        # Run A* with default settings
        astar = AStar(grid, start, goal, heuristic='manhattan', allow_diagonal=False)
        path, stats = astar.search()
        
        if path:
            print(f"Path found! Length: {stats['path_length']}, Cost: {stats['path_cost']:.2f}")
            print(f"Nodes explored: {stats['nodes_explored']}")
        else:
            print("No path found!")
        
        # Visualize result
        astar.visualize_search(f"{name} - A* Search Result")
        
        # Compare heuristics on first grid
        if i == 1:
            heuristic_results = compare_heuristics(grid, start, goal)
            
            # Visualize best and worst heuristics
            best_heuristic = min(heuristic_results.keys(), 
                               key=lambda h: heuristic_results[h]['stats']['nodes_explored'])
            worst_heuristic = max(heuristic_results.keys(), 
                                key=lambda h: heuristic_results[h]['stats']['nodes_explored'])
            
            print(f"Best heuristic: {best_heuristic} "
                  f"(explored {heuristic_results[best_heuristic]['stats']['nodes_explored']} nodes)")
            print(f"Worst heuristic: {worst_heuristic} "
                  f"(explored {heuristic_results[worst_heuristic]['stats']['nodes_explored']} nodes)")
        
        # Analyze diagonal movement on second grid
        if i == 2:
            analyze_diagonal_movement(grid, start, goal)
    
    # Performance benchmark
    print("\nRunning performance benchmark...")
    grid_sizes = [10, 15, 20, 25, 30]
    performance_benchmark(grid_sizes, obstacle_density=0.2)
    
    print("\n" + "=" * 50)
    print("A* Search Algorithm Demo Complete!")

if __name__ == "__main__":
    main()