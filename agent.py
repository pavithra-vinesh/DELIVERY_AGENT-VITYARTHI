import heapq

class DeliveryAgent:
    """
    Implements the core logic of the delivery agent, including search
    algorithms for pathfinding and a replanning strategy.
    """

    def __init__(self, environment):
        """Initializes the agent with a grid environment."""
        self.env = environment
        self.path = []
        self.path_cost = 0
        self.nodes_expanded = 0

    def heuristic(self, pos, goal):
        """
        Calculates the Manhattan distance heuristic, which is admissible
        for a 4-connected grid.
        """
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def ucs_search(self):
        """
        Performs Uniform-Cost Search to find the lowest-cost path.

        Returns:
            A tuple of (path, cost, nodes_expanded) or (None, inf, N) if no path is found.
        """
        start = self.env.start
        goal = self.env.goal
        self.nodes_expanded = 0
        
        # Priority queue: (cost, position, path)
        pq = [(0, start, [start])]
        visited = {start: 0}

        while pq:
            self.nodes_expanded += 1
            cost, current_pos, path = heapq.heappop(pq)
            
            if current_pos == goal:
                self.path = path
                self.path_cost = cost
                return self.path, self.path_cost, self.nodes_expanded

            # Explore neighbors (up, down, left, right)
            y, x = current_pos
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_y, next_x = y + dy, x + dx
                next_pos = (next_y, next_x)

                if self.env.is_valid_position(next_y, next_x):
                    move_cost = self.env.get_cost(next_y, next_x)
                    new_cost = cost + move_cost

                    if next_pos not in visited or new_cost < visited[next_pos]:
                        visited[next_pos] = new_cost
                        new_path = path + [next_pos]
                        heapq.heappush(pq, (new_cost, next_pos, new_path))
        
        return None, float('inf'), self.nodes_expanded

    def a_star_search(self):
        """
        Performs A* Search to find an optimal path using a heuristic.

        Returns:
            A tuple of (path, cost, nodes_expanded) or (None, inf, N) if no path is found.
        """
        start = self.env.start
        goal = self.env.goal
        self.nodes_expanded = 0
        
        # Priority queue: (f_score, cost, position, path)
        # f_score = cost + heuristic
        pq = [(self.heuristic(start, goal), 0, start, [start])]
        g_score = {start: 0}

        while pq:
            self.nodes_expanded += 1
            f_score, cost, current_pos, path = heapq.heappop(pq)
            
            if current_pos == goal:
                self.path = path
                self.path_cost = cost
                return self.path, self.path_cost, self.nodes_expanded

            # Explore neighbors
            y, x = current_pos
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_y, next_x = y + dy, x + dx
                next_pos = (next_y, next_x)

                if self.env.is_valid_position(next_y, next_x):
                    move_cost = self.env.get_cost(next_y, next_x)
                    tentative_g_score = cost + move_cost

                    if tentative_g_score < g_score.get(next_pos, float('inf')):
                        g_score[next_pos] = tentative_g_score
                        f_score = tentative_g_score + self.heuristic(next_pos, goal)
                        new_path = path + [next_pos]
                        heapq.heappush(pq, (f_score, tentative_g_score, next_pos, new_path))
        
        return None, float('inf'), self.nodes_expanded

    def hill_climbing_replanning(self, current_pos, current_time, max_restarts=5):
        """
        A local search strategy (Hill Climbing with Random Restarts)
        for dynamic replanning.

        This is a proof-of-concept and does not guarantee optimality.
        It finds a new path to the goal by locally searching for a better move.
        In a real scenario, this would involve a full search from the current
        position to the goal. For this project, we re-run A* from the current
        location.
        """
        print(f"[{current_time}] Obstacle detected at agent's planned next step. Initiating replanning...")
        
        # In this simplified model, replanning means running A* again from
        # the agent's current location to the goal.
        # This is a robust way to handle the problem for this assignment.
        original_start = self.env.start
        self.env.start = current_pos
        
        # Run A* from the current position
        new_path, new_cost, new_nodes = self.a_star_search()

        # Restore original start for future runs if needed
        self.env.start = original_start

        if new_path:
            self.path = new_path
            self.path_cost = new_cost
            self.nodes_expanded = new_nodes
            print(f"[{current_time}] Replanning successful. New path found with cost: {new_cost}")
            return True
        else:
            print(f"[{current_time}] Replanning failed. No new path found.")
            return False
