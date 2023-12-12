import os
import copy
import pickle
import numpy as np
import motion_planning as mp
import training_models as tm

TOL = 0.2
LIM = 0.01
ITERS = 100
TIME = 0.1
FOV = 1.5
D = 1e-5

MODEL = "basic_1_0_1_8_512_2048=leaky_relu.pth"

lower_arr = [[ 0.5, 1.7, 2.7, 2.7, 3.8], # x coords
             [-0.3,-0.7,-1.3, 0.3,-0.5]] # y coords
    
size_arr  = [[0.7, 0.5, 0.5, 0.5, 0.7],  # width: x
             [1.0, 0.7, 1.0, 1.0, 1.0]]  # height:y

limit = [[0.0,-1.2,-0.7,-0.7], # lower[pos_x, pos_y,
         [5.0, 1.2, 0.7, 0.7]] # upper vel_x, vel_y]
goal  =  [5.0, 0.0, 0.0, 0.0]

world_obs = mp.ObsMap(lower_arr, size_arr)
world = mp.World(limit, goal, world_obs, TOL)


def data_collection(num_iters, plot_sol):
    for iter in range(num_iters):

        start = world.random_state(ITERS, LIM)
        robot = mp.Robot(start, world_obs, TIME, FOV)

        print(f"\nDEBUG: world.random_state() done: {[round(x, 2) for x in start]}")
        problem, vars, params = mp.motion_planning(world, robot, relaxed=False)

        state , input, bool_low , bool_upp  = vars
        state0, goal0, lower_obs, upper_obs = params

        dist = lambda x: np.linalg.norm(np.array(robot.state) - np.array(x))

        while dist(goal) > world.TOL:
            print(f"DEBUG: abs(distance) to goal: {round(dist(goal), 2)}")

            state0.value = np.array(robot.state)
            goal0.value  = np.array(goal)
            robot.detect_obs()

            lower_cpy = copy.deepcopy(robot.local_obs.lower_arr)
            size_cpy  = copy.deepcopy(robot.local_obs.size_arr)

            while len(lower_cpy[0]) < world.MAX:    # Ensure arr len = MAX
                low = min(limit[0][0], limit[0][1]) # Get low within big-M

                for i in [0, 1]:             # Add fake obs to x(0) & y(1)
                    lower_cpy[i].append(low) # Fake obs have lower x,y val
                    size_cpy [i].append(0.0) # outside of world; size: 0.0

            lower_obs.value = np.array(lower_cpy)
            upper_obs.value = np.array(lower_cpy) + np.array(size_cpy)

            problem.solve(verbose=False)
            print(f"\nSolving iter = {iter}")
            print(f"Status = {problem.status}")
            print(f"Optimal cost = {round(problem.value, 2)}")
            print(f"Solve time = {round(problem.solver_stats.solve_time, 4)}s")

            state_sol, input_sol = state.value, input.value
            arnd = lambda x: np.around(x.value, 1).tolist() # Rounds values to X.0
            
            bl_sol, bu_sol = [], []
            obs = [lower_cpy, size_cpy]
            
            for i in range(world.MAX):
                bl_sol.append(arnd(bool_low[i]))
                bu_sol.append(arnd(bool_upp[i]))

            world.solutions.append([robot.state, obs, bl_sol, bu_sol])
            robot.update_state(input_sol[0][0], input_sol[1][0])

            if plot_sol:
                world.plot_problem(state_sol, start, goal)

        if plot_sol:
            world.plot_problem(np.array(robot.state_traj), start, goal)

        file = open(f"data/sol{iter}.pkl", "wb")
        pickle.dump(world.solutions, file)
        world.solutions = []


def dagger_problem(num_iters, plot_sol):
    start = world.random_state(ITERS, LIM)
    robot = mp.Robot(start, world_obs, TIME, FOV)
    
    print(f"\nDEBUG: randomed start done: {[round(x, 2) for x in start]}")

    problem, vars, params = mp.motion_planning(world, robot, relaxed=True)
    state, input = vars
    bool_low, bool_upp, state0, goal0, lower_obs, upper_obs = params

    B_problem, B_vars , B_params = mp.motion_planning(world, robot, relaxed=False, horizon=5)
    B_state  , B_input, _ , _    = B_vars
    B_state0 , B_goal0, B_lower_obs, B_upper_obs = B_params

    f_problem, f_vars , f_params   = mp.motion_planning(world, robot, relaxed=False)
    f_state  , f_input, f_bool_low , f_bool_upp = f_vars
    f_state0 , f_goal0, f_lower_obs, f_upper_obs = f_params

    dist = lambda x: np.linalg.norm(np.array(robot.state) - np.array(x))

    prev_dist = float("inf")
    save_data = False

    while dist(goal) > world.TOL:
        print(f"DEBUG: abs(distance) to goal: {round(dist(goal), 2)}")
        
        state0.value = np.array(robot.state)
        goal0.value  = np.array(goal)
        robot.detect_obs()

        lower_cpy = copy.deepcopy(robot.local_obs.lower_arr)
        size_cpy  = copy.deepcopy(robot.local_obs.size_arr)


        while len(lower_cpy[0]) < world.MAX:    # Ensure arr len = MAX
            low = min(limit[0][0], limit[0][1]) # Get low within big-M
            
            for i in [0, 1]:             # Add fake obs to x(0) & y(1)
                lower_cpy[i].append(low) # Fake obs have lower x,y val
                size_cpy [i].append(0.0) # outside of world; size: 0.0

        lower_obs.value = np.array(lower_cpy)
        upper_obs.value = np.array(lower_cpy) + np.array(size_cpy)

        bl_sol, bu_sol, _, _, _ = tm.get_model_outs(
            None, MODEL, robot.state, [lower_cpy, size_cpy])

        for i in range(world.MAX):
            bool_low[i].value = bl_sol[i]
            bool_upp[i].value = bu_sol[i]

        problem.solve(verbose=False)
        print(f"Status = {problem.status}")
        print(f"Optimal cost = {round(problem.value, 2)}")
        print(f"Solve time = {round(problem.solver_stats.solve_time, 4)}s")

        state_sol, input_sol = state.value, input.value
        stuck = abs(dist(goal) - prev_dist) < D

        if not isinstance(state_sol, np.ndarray) or stuck:
            save_data = True
            print("\nDEBUG: invalid solution. Solving the backup problem")

            B_state0.value    = np.array(robot.state)
            B_goal0.value     = np.array(goal)
            B_lower_obs.value = np.array(lower_cpy)
            B_upper_obs.value = np.array(lower_cpy) + np.array(size_cpy)


            B_problem.solve(verbose=False)
            state_sol, input_sol = B_state.value, B_input.value
        
            if stuck:
                print("\nDEBUG: invalid again. Doing full dagger problem")

                f_state0.value    = np.array(robot.state)
                f_goal0.value     = np.array(goal)
                f_lower_obs.value = np.array(lower_cpy)
                f_upper_obs.value = np.array(lower_cpy) + np.array(size_cpy)

                f_problem.solve(verbose=False)
                state_sol, input_sol = f_state.value, f_input.value

                arnd = lambda x: np.around(x.value, 1).tolist() # Rounds values to X.0
                bl_sol, bu_sol = [], []
                
                for i in range(world.MAX):
                    bl_sol.append(arnd(f_bool_low[i]))
                    bu_sol.append(arnd(f_bool_upp[i]))   
        
        obs = [lower_cpy, size_cpy]
        world.solutions.append([robot.state, obs, bl_sol, bu_sol])

        prev_dist = dist(goal)
        robot.update_state(input_sol[0][0], input_sol[1][0])

        if plot_sol:
            world.plot_problem(state_sol, start, goal)
        
    if save_data:
        iter = len(os.listdir("dagger"))
        if iter > num_iters:
            exit()
        else:
            file = open(f"dagger/sol_dagger{iter}.pkl", "wb")
            pickle.dump(world.solutions, file)
            world.solutions = []


class Problem:
    
    def __init__(self):
        self.start = world.random_state(ITERS, LIM)
        self.robot = mp.Robot(self.start, world_obs, TIME, FOV)

        print(f"\nDEBUG: randomed start done: {[round(x, 2) for x in self.start]}")

        self.problem, self.vars, self.params = mp.motion_planning(world, self.robot, relaxed=True)
        self.state, self.input = self.vars
        self.bool_low, self.bool_upp , self.state0, self.goal0, self.lower_obs, self.upper_obs = self.params

        self.B_problem, self.B_vars , self.B_params = mp.motion_planning(world, self.robot, relaxed=False, horizon=5)
        self.B_state  , self.B_input, _, _ = self.B_vars
        self.B_state0 , self.B_goal0, self.B_lower_obs, self.B_upper_obs = self.B_params

        self.f_problem, self.f_vars , self.f_params = mp.motion_planning(world, self.robot, relaxed=False)
        self.f_state  , self.f_input, self.f_bool_low, self.f_bool_upp   = self.f_vars
        self.f_state0 , self.f_goal0, self.f_lower_obs, self.f_upper_obs = self.f_params

        self.dist = lambda x: np.linalg.norm(np.array(self.robot.state) - np.array(x))
        self.prev_dist = float("inf")


    def relaxed_problem(self, robot_state):

        while self.dist(goal) > world.TOL:
            print(f"DEBUG: abs(distance) to goal: {round(self.dist(goal), 2)}")
            
            self.state0.value = np.array(robot_state)
            self.goal0.value  = np.array(goal)
            self.robot.detect_obs()

            lower_cpy = copy.deepcopy(self.robot.local_obs.lower_arr)
            size_cpy  = copy.deepcopy(self.robot.local_obs.size_arr)

            while len(lower_cpy[0]) < world.MAX:    # Ensure arr len = MAX
                low = min(limit[0][0], limit[0][1]) # Get low within big-M
                
                for i in [0, 1]:             # Add fake obs to x(0) & y(1)
                    lower_cpy[i].append(low) # Fake obs have lower x,y val
                    size_cpy [i].append(0.0) # outside of world; size: 0.0
        
            self.lower_obs.value = np.array(lower_cpy)
            self.upper_obs.value = np.array(lower_cpy) + np.array(size_cpy)
        
            bl_sol, bu_sol, _, _, _ = tm.get_model_outs(None, 
                MODEL, self.robot.state, [lower_cpy, size_cpy])

            for i in range(world.MAX):
                self.bool_low[i].value = bl_sol[i]
                self.bool_upp[i].value = bu_sol[i]

            self.problem.solve(verbose=False)
            print(f"Status = {self.problem.status}")
            print(f"Optimal cost = {round(self.problem.value, 2)}")
            print(f"Solve time = {round(self.problem.solver_stats.solve_time, 4)}s")


            state_sol, input_sol = self.state.value, self.input.value
            stuck = abs(self.dist(goal) - prev_dist) < D

            if not isinstance(state_sol, np.ndarray) or stuck:
                print("\nDEBUG: invalid solution. Solving backup problem")

                self.B_state0.value    = np.array(self.robot.state)
                self.B_goal0.value     = np.array(goal)
                self.B_lower_obs.value = np.array(lower_cpy)
                self.B_upper_obs.value = np.array(lower_cpy) + np.array(size_cpy)

                self.B_problem.solve(verbose=False)
                state_sol, input_sol = self.B_state.value, self.B_input.value

                if stuck:
                    print("\nDEBUG: invalid again. Doing dagger problem:")

                    self.f_state0.value    = np.array(self.robot.state)
                    self.f_goal0.value     = np.array(goal)
                    self.f_lower_obs.value = np.array(lower_cpy)
                    self.f_upper_obs.value = np.array(lower_cpy) + np.array(size_cpy)

                    self.f_problem.solve(verbose=False)
                    state_sol, input_sol = self.f_state.value, self.f_input.value
                    
                    arnd = lambda x: np.around(x.value, 1).tolist() # Rounds values to X.0
                    bl_sol, bu_sol = [], []
                    
                    for i in range(world.MAX):
                        bl_sol.append(arnd(self.f_bool_low[i]))
                        bu_sol.append(arnd(self.f_bool_upp[i]))
                
            prev_dist = self.dist(goal)
            # self.robot.update_state(input_sol[0][0], input_sol[1][0])
            return state_sol


if __name__ == "__main__":
    problem = Problem()
    SOLVE, i = False, 0
    
    if SOLVE:
        data_collection(num_iters=1600, plot_sol=False)
    else:
        while True:
            print(f"\nCYCLE = {i}")
            dagger_problem(num_iters=100, plot_sol=False)
            i += 1