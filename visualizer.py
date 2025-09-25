import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import argparse

class Visualizer:
    """
    Handles the visualization of the agent's path on the grid using matplotlib.
    """

    def __init__(self, map_path, log_path):
        """
        Initializes the visualizer with map and log file paths.
        """
        self.map_path = map_path
        self.log_path = log_path
        self.grid = None
        self.start_pos = None
        self.goal_pos = None
        self.dynamic_obstacles = []
        self.path = []
        self._load_map_data()
        self._load_log_data()

    def _load_map_data(self):
        """
        Loads and processes the map file to extract grid, start, goal, and dynamic obstacles.
        """
        with open(self.map_path, 'r') as file:
            content = [line.strip() for line in file if line.strip()]
        
        static_grid = [line for line in content if not line.startswith('M(')]
        if not static_grid:
            raise ValueError("Map file is empty or contains no grid data.")
            
        expected_width = len(static_grid[0].split('-'))
        
        self.grid = []
        for row_idx, line in enumerate(static_grid):
            cells = line.split('-')
            if len(cells) != expected_width:
                raise ValueError("Map grid must be rectangular.")
            
            self.grid.append(cells)
            for col_idx, cell in enumerate(cells):
                if cell == 'S':
                    self.start_pos = (row_idx, col_idx)
                elif cell == 'G':
                    self.goal_pos = (row_idx, col_idx)

        moving_obstacles = [line for line in content if line.startswith('M(')]
        for obstacle_def in moving_obstacles:
            # M(y,x,t) -> extract y,x,t
            coords_str = obstacle_def[2:-1]
            y_val, x_val, t_val = map(int, coords_str.split(','))
            self.dynamic_obstacles.append({'y': y_val, 'x': x_val, 't': t_val})

    def _load_log_data(self):
        """
        Loads and processes the log file to extract the agent's movement path.
        """
        with open(self.log_path, 'r') as file:
            log_entries = file.readlines()
        
        movements = []
        for entry in log_entries:
            if "Agent moves to" in entry:
                # Format: [1] Agent moves to (0, 1)
                try:
                    position_str = entry.split('(')[1].split(')')[0]
                    y_coord, x_coord = map(int, position_str.split(','))
                    movements.append((y_coord, x_coord))
                except (IndexError, ValueError):
                    print(f"Warning: Could not parse path from log line: {entry.strip()}")
            elif "Obstacle detected" in entry:
                # Format: [6] Obstacle detected at planned position (2, 4). Current position: (1, 4).
                try:
                    position_str = entry.split('(')[1].split(')')[0]
                    y_coord, x_coord = map(int, position_str.split(','))
                    self.path.append({'type': 'obstacle', 'pos': (y_coord, x_coord)})
                    print(f"Info: Obstacle detected at {y_coord},{x_coord}. Path visualization will reflect this.")
                except (IndexError, ValueError):
                    print(f"Warning: Could not parse obstacle from log line: {entry.strip()}")
        
        if movements:
            # Include start position if not already first
            if self.start_pos and movements[0] != self.start_pos:
                self.path.append({'type': 'path', 'pos': self.start_pos})
            
            for move in movements:
                self.path.append({'type': 'path', 'pos': move})

    def visualize(self):
        """
        Generates and displays the visualization of the grid and the agent's path.
        """
        if not self.grid:
            print("Error: Grid not loaded.")
            return

        # Create numeric representation for visualization
        grid_values = np.zeros((len(self.grid), len(self.grid[0])), dtype=float)
        for row in range(len(self.grid)):
            for col in range(len(self.grid[0])):
                cell_content = self.grid[row][col]
                if cell_content.isdigit():
                    grid_values[row][col] = int(cell_content)
                elif cell_content == '#':
                    grid_values[row][col] = 0 # Obstacle
                elif cell_content in ['S', 'G']:
                    grid_values[row][col] = 1 # Terrain cost 1 for start/goal

        # Configure color scheme
        color_scheme = mcolors.ListedColormap(['#6e6e6e', '#d2e7d7', '#9bce9f', '#56b86e'])
        value_boundaries = [0, 1, 2, 3]
        normalizer = mcolors.BoundaryNorm(value_boundaries, color_scheme.N)

        figure, axis = plt.subplots(figsize=(10, 8))
        axis.imshow(grid_values, cmap=color_scheme, norm=normalizer, origin='upper')

        # Configure grid lines
        axis.set_xticks(np.arange(-.5, len(self.grid[0]), 1))
        axis.set_yticks(np.arange(-.5, len(self.grid), 1))
        axis.grid(color='black', linestyle='-', linewidth=1)
        axis.set_xticklabels([])
        axis.set_yticklabels([])

        # Add cell labels
        for row in range(len(self.grid)):
            for col in range(len(self.grid[0])):
                cell_text = self.grid[row][col]
                if cell_text == 'S':
                    axis.text(col, row, 'S', ha='center', va='center', color='red', fontsize=16, fontweight='bold')
                elif cell_text == 'G':
                    axis.text(col, row, 'G', ha='center', va='center', color='blue', fontsize=16, fontweight='bold')
                elif cell_text == '#':
                    axis.text(col, row, '#', ha='center', va='center', color='white', fontsize=16, fontweight='bold')
                elif cell_text.isdigit():
                    axis.text(col, row, cell_text, ha='center', va='center', color='black')

        # Plot agent's trajectory
        x_coords = []
        y_coords = []
        
        for movement in self.path:
            location = movement['pos']
            if movement['type'] == 'path':
                y_coords.append(location[0])
                x_coords.append(location[1])
            elif movement['type'] == 'obstacle':
                # Show where replanning was needed
                axis.plot(location[1], location[0], 'X', color='purple', markersize=15, label='Replanning Obstacle')
        
        axis.plot(x_coords, y_coords, linestyle='--', color='purple', marker='o', markersize=8, label='Agent Path')
        
        # Show dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            axis.plot(obstacle['x'], obstacle['y'], 'o', color='red', markersize=12, label='Dynamic Obstacle')
            
        axis.set_title("Agent Path Visualization", fontsize=16)
        axis.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Visualize agent path from a log file.")
    arg_parser.add_argument("--map_path", type=str, required=True, help="Path to the map file.")
    arg_parser.add_argument("--log_path", type=str, required=True, help="Path to the log file.")
    arguments = arg_parser.parse_args()

    try:
        viz = Visualizer(arguments.map_path, arguments.log_path)
        viz.visualize()
    except (IOError, ValueError) as error:
        print(f"Error: {error}")
