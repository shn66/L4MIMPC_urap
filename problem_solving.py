import copy
import random
import pickle
import numpy as np
import motion_planning as mp
import training_models as tm
from datetime import datetime

MODEL = "REF_NEW_0_0_0=leaky_relu.pth"

lower_arr = [[ 0.5, 1.7, 2.7, 2.7, 3.8], # x coords
             [-0.3,-0.7,-1.3, 0.3,-0.5]] # y coords
    
size_arr  = [[0.7, 0.5, 0.5, 0.5, 0.7],  # width: x
             [1.0, 0.7, 1.0, 1.0, 1.0]]  # height:y

limit = [[0.0,-1.2,-0.7,-0.7], # lower[pos_x, pos_y,
         [5.0, 1.2, 0.7, 0.7]] # upper vel_x, vel_y]
goal  =  [5.0, 0.0, 0.0, 0.0]

world_obs = mp.ObsMap(lower_arr, size_arr)
world = mp.World(limit, goal, world_obs, TOL=0.2)

def problem_solving(num_iters, plot_sol):
    # Randomize start, get vars & params
    for iter in range(num_iters):

        start = world.random_state(iters=100, bound=0.9)
        robot = mp.Robot(start, world_obs, TIME=0.1, FOV=1.5)

        print(f"\nDEBUG: world.random_state() done: {[round(x, 2) for x in start]}")

        problem, vars, params = mp.motion_planning(world, robot, relaxed=False)

        state , input, bool_low , bool_upp  = vars
        state0, goal0, lower_obs, upper_obs = params

        dist = lambda x: np.linalg.norm(np.array(robot.state) - np.array(x))

        # Initialize all CP.parameter values
        while dist(goal) > world.TOL:
            print(f"DEBUG: abs(distance) to goal: {round(dist(goal), 2)}")

            state0.value = np.array(robot.state)
            goal0.value  = np.array(goal)
            robot.detect_obs()

            obs = robot.local_obs
            lower_cpy = copy.deepcopy(obs.lower_arr)
            size_cpy  = copy.deepcopy(obs.size_arr)

            while len(lower_cpy[0]) < world.MAX:    # Ensure arr len = MAX
                low = min(limit[0][0], limit[0][1]) # Get low within big-M

                
                for i in [0, 1]:             # Add fake obs to x(0) & y(1)
                    lower_cpy[i].append(low) # Fake obs have lower x,y val
                    size_cpy [i].append(0.0) # outside of world; size: 0.0

            lower_obs.value = np.array(lower_cpy)
            upper_obs.value = np.array(lower_cpy) + np.array(size_cpy)

            # Now collect optimized trajectories
            problem.solve(verbose=False)
            print(f"\nSolving iter = {iter}")
            print(f"Status = {problem.status}")
            print(f"Optimal cost = {round(problem.value, 2)}")
            print(f"Solve time = {round(problem.solver_stats.solve_time, 4)}s")

            state_sol, input_sol = state.value, input.value
            arnd = lambda x: np.around(x.value, 1).tolist() # Rounds values to X.0
            bl_sol, bu_sol = [], []
            
            for i in range(world.MAX):
                bl_sol.append(arnd(bool_low[i]))
                bu_sol.append(arnd(bool_upp[i]))

            world.solutions.append([robot.state, [lower_cpy, size_cpy], bl_sol, bu_sol])
            robot.update_state(input_sol[0][0], input_sol[1][0])

            if plot_sol:
                world.plot_problem(state_sol, start, goal)

        if plot_sol:
            world.plot_problem(np.array(robot.state_traj), start, goal)

        file = open(f"data/sol{iter}.pkl", "wb")
        pickle.dump(world.solutions, file)
        world.solutions = []


def relaxed_problem(dataset, use_model, do_dagger=False):

    # Randomize start, get vars & params
    if use_model:
        # start = world.random_state(iters=100, bound=0.1)
        start = [0.0,random.uniform(-0.6,0.6), 0.0, 0.0]
    else:
        i = random.randint(0, dataset.size - 1) # Random sol @ index i
        start, obs_arr, bl_sol, bu_sol = dataset.sols[i]

    robot = mp.Robot(start, world_obs, TIME=0.1, FOV=1.5)
    print(f"\nDEBUG: randomed start done: {[round(x, 2) for x in start]}")

    problem, vars, params = mp.motion_planning(world, robot, relaxed=True)
    state, input = vars
    bool_low, bool_upp, state0, goal0, lower_obs, upper_obs = params

    full_problem, full_vars, full_params = mp.motion_planning(world, robot, relaxed=False)
    f_state , f_input, f_bool_low , f_bool_upp  = full_vars
    f_state0, f_goal0, f_lower_obs, f_upper_obs = full_params

    B_problem, B_vars, B_params = mp.motion_planning(world, robot, relaxed=False, horizon=5)
    B_state , B_input, _, _ = B_vars
    B_state0, B_goal0, B_lower_obs, B_upper_obs = B_params

    dist = lambda x: np.linalg.norm(np.array(robot.state) - np.array(x))

    # Initialize all CP.parameter values
    while dist(goal) > world.TOL:
        print(f"DEBUG: abs(distance) to goal: {round(dist(goal), 2)}")
        
        state0.value = np.array(robot.state)
        goal0.value  = np.array(goal)

        if use_model:
            robot.detect_obs()
            obs = robot.local_obs

            lower_cpy = copy.deepcopy(obs.lower_arr)
            size_cpy  = copy.deepcopy(obs.size_arr)


            while len(lower_cpy[0]) < world.MAX:    # Ensure arr len = MAX
                low = min(limit[0][0], limit[0][1]) # Get low within big-M
                
                for i in [0, 1]:             # Add fake obs to x(0) & y(1)
                    lower_cpy[i].append(low) # Fake obs have lower x,y val
                    size_cpy [i].append(0.0) # outside of world; size: 0.0
        else:
            lower_cpy, size_cpy = obs_arr[0], obs_arr[1]

        lower_obs.value = np.array(lower_cpy)
        upper_obs.value = np.array(lower_cpy) + np.array(size_cpy)

        # Now collect optimized trajectories
        if use_model:
            bl_sol, bu_sol, _, _, _ = tm.get_model_outs(
                dataset, MODEL, robot.state, [lower_cpy, size_cpy])

        for i in range(world.MAX):
            bool_low[i].value = bl_sol[i]
            bool_upp[i].value = bu_sol[i]

        problem.solve(verbose=False)
        print(f"Status = {problem.status}")
        print(f"Optimal cost = {round(problem.value, 2)}")
        print(f"Solve time = {round(problem.solver_stats.solve_time, 4)}s")

        state_sol, input_sol = state.value, input.value

        prev_dist = dist(goal)

        if not isinstance(state_sol, np.ndarray):
            
            print("\nDEBUG: invalid solution. Solving backup problem")

            B_state0.value = np.array(robot.state)
            B_goal0.value  = np.array(goal)

            B_lower_obs.value = np.array(lower_cpy)
            B_upper_obs.value = np.array(lower_cpy) + np.array(size_cpy)

            B_problem.solve(verbose=False)
            state_sol, input_sol = B_state.value, B_input.value

            if do_dagger:
                print("\nDEBUG: Doing data aggregation")
                f_state0.value = np.array(robot.state)
                f_goal0.value  = np.array(goal)
                f_lower_obs.value = np.array(lower_cpy)
                f_upper_obs.value = np.array(lower_cpy) + np.array(size_cpy)

                full_problem.solve(verbose=False)
                print(f"Status = {full_problem.status}")
                print(f"Optimal cost = {round(full_problem.value, 2)}")
                print(f"Solve time = {round(full_problem.solver_stats.solve_time, 4)}s")
                
                arnd = lambda x: np.around(x.value, 1).tolist() # Rounds values to X.0
                bl_sol, bu_sol = [], []
                
                for i in range(world.MAX):
                    bl_sol.append(arnd(f_bool_low[i]))
                    bu_sol.append(arnd(f_bool_upp[i]))

                world.solutions.append([robot.state, [lower_cpy, size_cpy], bl_sol, bu_sol])

                if abs(dist(goal)-prev_dist)<1e-5:             # If the robot is stuck, use full solution
                    state_sol, input_sol = f_state.value, f_input.value

        robot.update_state(input_sol[0][0], input_sol[1][0])
        world.plot_problem(state_sol, start, goal)
        
    # if do_dagger:
    #     day_time=datetime.now().strftime("%m_%d-%I-%M-%S")
    #     file = open(f"data/sol_dagger_{day_time}.pkl", "wb")
    #     pickle.dump(world.solutions, file)
    #     world.solutions = []


if __name__ == "__main__":
    dataset = tm.Dataset()
    SOLVE = False
    
    if SOLVE:
        problem_solving(num_iters=0, plot_sol=False)
    else:
        relaxed_problem(dataset, use_model=True, do_dagger=False)