import re

class GridCity:
    """
    Models the 2D grid environment for the delivery agent.

    The environment handles:
    - Loading a map from a file.
    - Storing terrain costs.
    - Managing static and dynamic obstacles.
    - Providing methods to check for valid moves and calculate costs.
    """

    def __init__(self, map_path):
        """Initializes the grid and loads the map."""
        self.grid = []
        self.start = None
        self.goal = None
        self.height = 0
        self.width = 0
        self.dynamic_obstacles = []
        self._parse_map(map_path)

    def _parse_map(self, map_path):
        """
        Parses a map file and initializes the grid and obstacles.
        """
        try:
            with open(map_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise ValueError(f"Map file not found at: {map_path}")

        grid_lines = []
        dynamic_obstacle_pattern = re.compile(r'^M\((\d+),(\d+),(\d+)\)')
        for line in lines:
            line = line.strip()
            if line and not dynamic_obstacle_pattern.match(line):
                grid_lines.append(line)
            else:
                match = dynamic_obstacle_pattern.match(line)
                if match:
                    # Store as (y, x, time)
                    self.dynamic_obstacles.append(
                        (int(match.group(1)), int(match.group(2)), int(match.group(3)))
                    )
        
        # Now parse the grid lines
        if not grid_lines:
            raise ValueError("Map file must contain at least a grid.")

        # Find the first non-empty line to determine the width
        first_row_len = 0
        for line in grid_lines:
            stripped_line = line.strip()
            if stripped_line:
                first_row_len = len(stripped_line)
                break

        if first_row_len == 0:
             raise ValueError("Map file must contain a non-empty grid line.")

        for line in grid_lines:
            line = line.strip()
            if not line:
                continue

            if len(line) != first_row_len:
                raise ValueError("Map grid must be rectangular.")
            
            row = []
            for char in line:
                if char == 'S':
                    self.start = (len(self.grid), len(row))
                    row.append(1)
                elif char == 'G':
                    self.goal = (len(self.grid), len(row))
                    row.append(1)
                elif char == '#':
                    row.append(float('inf'))
                elif char.isdigit():
                    row.append(int(char))
            self.grid.append(row)

        if not self.grid:
            raise ValueError("Map file must contain at least a grid after parsing.")

        self.height = len(self.grid)
        self.width = len(self.grid[0])

    def get_cost(self, y, x):
        """
        Returns the movement cost for a given cell (y, x).
        Returns infinity for obstacles.
        """
        if 0 <= y < self.height and 0 <= x < self.width:
            return self.grid[y][x]
        return float('inf')

    def is_valid_position(self, y, x):
        """
        Checks if a position is within grid boundaries and not a static obstacle.
        """
        if not (0 <= y < self.height and 0 <= x < self.width):
            return False
        return self.grid[y][x] != float('inf')

    def is_occupied(self, y, x, current_time=0):
        """
        Checks if a cell (y, x) is occupied by a static or dynamic obstacle
        at a specific time step.
        """
        if self.get_cost(y, x) == float('inf'):
            return True  # Static obstacle

        # Check for dynamic obstacles
        for dy, dx, t in self.dynamic_obstacles:
            if dy == y and dx == x and t == current_time:
                return True
        return False
