import argparse
import time
import sys

# Import the environment and agent classes
from environment import GridCity
from agent import DeliveryAgent

def run_simulation(map_path, algorithm, log_file=None):
    """
    Runs a single simulation of the agent with the specified algorithm.
    """
    print(f"--- Running {algorithm} on {map_path} ---")
    
    try:
        env = GridCity(map_path)
    except ValueError as e:
        print(f"Error: {e}")
        return

    agent = DeliveryAgent(env)
    
    start_time = time.time()
    
    if algorithm == 'ucs':
        path, cost, nodes = agent.ucs_search()
    elif algorithm == 'a_star':
        path, cost, nodes = agent.a_star_search()
    elif algorithm == 'dynamic':
        path, cost, nodes = agent.a_star_search()  # Start with an A* plan
        if not path:
            print("Initial plan failed. Exiting.")
            return
        
        print("Initial path planned. Simulating agent movement...")
        log = open(log_file, 'w') if log_file else sys.stdout
        
        current_time_step = 0
        current_pos = env.start
        log.write(f"[{current_time_step}] Agent starts at {current_pos}\n")
        
        remaining_path = path[1:]
        
        # Simulate step-by-step
        while current_pos != env.goal:
            current_time_step += 1
            if not remaining_path:
                print("Agent reached a dead end. No further path available.")
                break

            next_pos = remaining_path[0]
            
            # Check for dynamic obstacle at the next step
            if env.is_occupied(next_pos[0], next_pos[1], current_time_step):
                log.write(f"[{current_time_step}] Obstacle detected at planned position {next_pos}. Current position: {current_pos}.\n")
                
                # Use local search for replanning
                if agent.hill_climbing_replanning(current_pos, current_time_step):
                    remaining_path = agent.path[1:]
                    next_pos = remaining_path[0] if remaining_path else env.goal
                    log.write(f"[{current_time_step}] Replanning successful. New path found.\n")
                else:
                    log.write(f"[{current_time_step}] Replanning failed. Agent is stuck.\n")
                    break
            
            current_pos = next_pos
            log.write(f"[{current_time_step}] Agent moves to {current_pos}\n")
            
            if remaining_path:
                remaining_path.pop(0)

        # Final state for dynamic algorithm
        path, cost, nodes = agent.path, agent.path_cost, agent.nodes_expanded
        
        if log_file:
            log.close()
            print(f"Simulation log saved to {log_file}")
    else:
        print(f"Unknown algorithm: {algorithm}. Please choose 'ucs', 'a_star', or 'dynamic'.")
        return

    end_time = time.time()
    
    print("\n--- Results ---")
    if path:
        print(f"Path found: {' -> '.join(str(p) for p in path)}")
        print(f"Total Path Cost: {cost}")
    else:
        print("No path found.")
    print(f"Nodes Expanded: {nodes}")
    print(f"Execution Time: {end_time - start_time:.4f} seconds")
    print("-" * 20)

def main():
    parser = argparse.ArgumentParser(description="Autonomous Delivery Agent Simulator")
    parser.add_argument("--map_path", required=True, help="Path to the map file (e.g., maps/small_map.txt)")
    parser.add_argument("--algorithm", help="The algorithm to use: 'ucs', 'a_star', or 'dynamic'")
    parser.add_argument("--run_experiments", action="store_true", help="Run all static algorithms on the map and compare results")
    args = parser.parse_args()

    if args.run_experiments:
        run_simulation(args.map_path, 'ucs')
        run_simulation(args.map_path, 'a_star')
    elif args.algorithm:
        if args.algorithm == 'dynamic':
            run_simulation(args.map_path, 'dynamic', log_file='proof_of_concept.log')
        else:
            run_simulation(args.map_path, args.algorithm)
    else:
        parser.print_help()
        print("\nError: Please specify an algorithm or use --run_experiments.")

if __name__ == "__main__":
    main()
