import copy
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