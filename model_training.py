import ast
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as fn
import torchvision.transforms as hrt
import motion_planning as mp

class Dataset:
    """
    solutions = [[state0, final_state, bl_sol, bu_sol] ...]
    trajects = [[start, goal, state_traj, input_traj], ...]

    data_map = {key: iteration, value: [[soln], [soln]...]}
    -> lets us map a final (start & goal) to each array in solutions
    """
    def __init__(self):
        self.solutions = [] # Dynamic data structure setup. Do not mutate.
        self.trajects  = []
        self.data_map  = {}
        
        print("\nDEBUG: Dataset initialized.")
        key = 0
        sols = open("data/solutions.txt", "r")

        for line in sols.readlines()[1:]:     # skip first formatting line
            if len(line) <= 5:                # if we see iteration label:

                key = int(line.split(":")[0]) # label = "0:\n", need the 0
                self.data_map[key] = []       # create new array using key
            else:
                arr = []
                for x in line.split(";"):     # data arrays separated by ;
                    arr.append(ast.literal_eval(x.strip())) # evaluate str

                self.data_map[key].append(arr)# add soln_list to arr_@_key
                self.solutions.append(arr)    # add to solutions arr, etc.

        traj = open("data/trajects.txt", "r")
        for line in traj.readlines()[1:]:

            if len(line) > 5:
                arr = []
                for x in line.split(";"):
                    arr.append(ast.literal_eval(x.strip()))
                self.trajects.append(arr)

        print(f"DEBUG: Data imported. # of keys = {len(self.data_map.keys())}")


    def start_data(self, iter=None, soln_id=None):
        max_t = len(self.trajects) - 1        # avoids index out of bounds

        if iter == None:
            iter = random.randint(0, max_t)   # random index if None given
        traj = self.trajects[min(iter, max_t)]# select item in array, etc.

        soln_arr = self.data_map[min(iter, max_t)]
        max_s = len(soln_arr) - 1

        if soln_id == None:
            soln_id = random.randint(0, max_s)
        soln = soln_arr[min(soln_id, max_s)]

        return soln, traj[0], traj[1]         # [[soln]], [start], [goal].


def model_training():
    """ TODO:
    - Trying to predict a series of binary variables (1,0), so do we use classification or regression? 
        (How many classes we use is hard to determine, regression is faster but has its own catches)
    - Problem: how do we go from the inputs (state, control) to predicting the binary variables? 
        (Most likely use feed-forward neural net).
    - Task: Set up a rudimentary neural network. Make a boilerplate function that spits out a basic 
        correct-dimensions output. Do some research into learning rates / ML concepts.
    """


def relaxed_problem(dataset):
    # MODIFIED motion planning problem

    lower_arr = [[0.0, 2.0, 2.0, 5.0, 7.5, 10.0, 12.0, 12.0, 15.0, 17.5], # x coords
                 [1.0,-5.0, 3.0,-2.0,-5.0, 1.0, -5.0,  3.0, -2.0, -5.0]]  # y coords
    
    size_arr  = [[1.5, 2.5, 2.5, 2.0, 2.0, 1.5, 2.5, 2.5, 2.0, 2.0],      # width: x
                 [2.0, 7.0, 2.0, 6.5, 6.0, 2.0, 7.0, 2.0, 6.5, 6.0]]      # height:y
    
    limit = [[0.0, -4.9,-1.0,-1.0], # lower[pos_x, pos_y,
             [20.0, 4.9, 1.0, 1.0]] # upper vel_x, vel_y]
    
    soln, start, goal = dataset.start_data(soln_id = 0) # [[soln]], [start], [goal]
    start, final, bl_sol, bu_sol = soln      # override global start with local one
    
    global_obs = mp.Obstacle_Map(lower_arr, size_arr)
    world = mp.Environment(limit, goal, global_obs, TOL=0.1)

    robot = mp.Robot(start, global_obs, TIME=0.2, FOV=10.0)
    problem, vars, params = mp.motion_planning(world, robot, relaxed=True)

    state, input = vars
    bool_low, bool_upp, state0, goal0, obs_lower, obs_upper = params

    diff = lambda x: np.linalg.norm(np.array(robot.state) - np.array(x))

    # Initialize all CP parameter values
    while diff(goal) > world.TOL: # while not at goal

        print(f"DEBUG: abs(distance) to goal: {round(diff(goal), 2)}")
        

        # TODO: bool_low, bool_upp are parameters instead of variables
        # Pass them into relaxed problem & compare solns / solve times
        # FIXME: works using any iter and soln_id=0, eventually errors

        state0.value = np.array(start)
        goal0.value  = np.array(goal)
        
        for i in range(world.MAX):
            bool_low[i].value = np.array(bl_sol[i])
            bool_upp[i].value = np.array(bu_sol[i])

        robot.detect_obs()
        l = copy.deepcopy(robot.local_obs.lower_arr)
        s = copy.deepcopy(robot.local_obs.size_arr)

        while (len(l[0]) < world.MAX):
            for i in range(2):
                l[i].append(-2.0) # [len(L) to world.MAX] are fake obs
                s[i].append(0.0)  # fake obs have lower x,y: -2.0,-2.0

        obs_lower.value = np.array(l)
        obs_upper.value = np.array(l) + np.array(s)

        # Now collect optimized trajectories
        print(f"\nSolving problem...")
        problem.solve(verbose = False)

        print(f"Status = {problem.status}")
        print(f"Optimal cost = {int(problem.value)}")
        print(f"Solve time = {problem.solver_stats.solve_time} secs.")


        x_sol = state.value
        u_sol = input.value

        robot.update_state(u_sol[0][0], u_sol[1][0])
        # 1st value in arr(  x_accel  ,   y_accel  )
        world.plot_problem(x_sol, start, goal)

if __name__ == "__main__":
    dataset = Dataset()
    relaxed_problem(dataset)