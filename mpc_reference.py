import os
import copy
import pickle
import numpy as np
import motion_planning as mp
import training_models as tm
import torch as th
import torch.nn.functional as fn
import re
import os


TOL = 0.2
LIM = 0.01
ITERS = 100
TIME = 0.1
FOV = 1.5
D = 1e-5

dir =  os.path.dirname(__file__)
MODEL = "dagger_1_0_1=leaky_relu.pth"
model_path = os.path.join(dir, "models/"+MODEL)



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
    
    def __init__(self, state = [0.,0.,0.,0.], model_path = MODEL):
        self.start = state
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


        nums = [int(x) for x in re.findall(r'\d+', f"models/{model_path}")]
        norms, drops = bool(nums[0]), bool(nums[1])       # Term 1,2 (int->bool)

        funct = model_path.split("=")[-1][:-4]  # Final value (string)
        activ = eval(f"fn.{funct}")       # TODO: fix hardcoding

        self.model = tm.BinaryNN(norms, drops, activ)

        dir =  os.path.dirname(__file__)
        model_path = os.path.join(dir, "models/"+MODEL)
        load  = th.load(model_path)
        self.model.load_state_dict(load)
        self.model.eval()


        self.tens = lambda x: th.Tensor(x).view(-1)

        self.output_dim= 500

        self.prev_dist =10.0


    def eval_model(self, state, obs_arr):
        input = th.cat((self.tens(state), self.tens(obs_arr)))
        input = input.unsqueeze(0)

        with th.no_grad():
            output = th.sigmoid(self.model(input))
    
        output = (output.view(-1) >= 0.5).float()          # Remove batch dim, rounds to 0 or 1

        nums = lambda x: x.view(5, 2, 25).detach().numpy() # Shape lst to multi-dim -> np.array
        bl_out, bu_out = nums(output[:self.output_dim//2]), nums(output[self.output_dim//2:])

        return bl_out, bu_out
        


    def get_reference(self, robot_state=None):

        # while self.dist(goal) > world.TOL:
        print(f"DEBUG: abs(distance) to goal: {round(self.dist(goal), 2)}")
        
        self.state0.value = np.array(robot_state) if robot_state else np.array(self.robot.state)
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
    
        bl_sol, bu_sol =  self.eval_model(self.robot.state, [lower_cpy, size_cpy])

        for i in range(world.MAX):
            self.bool_low[i].value = bl_sol[i]
            self.bool_upp[i].value = bu_sol[i]

        self.problem.solve(verbose=False)
        print(f"Status = {self.problem.status}")
        print(f"Optimal cost = {round(self.problem.value, 2)}")
        print(f"Solve time = {round(self.problem.solver_stats.solve_time, 4)}s")


        state_sol, input_sol = self.state.value, self.input.value
        stuck = abs(self.dist(goal) - self.prev_dist) < D

        if not isinstance(state_sol, np.ndarray) or stuck:
            print("\nDEBUG: invalid solution. Solving backup problem")

            self.B_state0.value    = np.array(self.robot.state)
            self.B_goal0.value     = np.array(goal)
            self.B_lower_obs.value = np.array(lower_cpy)
            self.B_upper_obs.value = np.array(lower_cpy) + np.array(size_cpy)

            self.B_problem.solve(verbose=False)
            state_sol, input_sol = self.B_state.value, self.B_input.value

            if stuck:
                print("\nDEBUG: invalid again. Solving full problem:")

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
                
        self.prev_dist = self.dist(goal)
        if not robot_state:
            self.robot.update_state(input_sol[0][0], input_sol[1][0])
        return state_sol, self.prev_dist < world.TOL
    

if __name__ == "__main__":
    problem = Problem()


    reached = False

    while not reached:
        reference, reached = problem.get_reference()

        print(f"X: {reference[0]}")
        print(f"Y: {reference[1]}")
        print(f"V_X: {reference[2]}")
        print(f"V_Y: {reference[3]}")