import ast
import copy
import torch
import numpy as np
import torch.nn as nn
import torchvision as tv
import matplotlib.pyplot as plt
import torch.nn.functional as fn
import torchvision.transforms as hrt
import motion_planning as mp

class Dataset:
    def __init__(self):
        self.solutions = [] # [[state0, final_state, bl_sol, bu_sol], ...]
        self.trajects  = [] # [[start, goal, state_traj, input_traj], ...]
        self.import_files()

    def import_files(self):

        sols = open("data/solutions.txt", "r")
        for line in sols.readlines():
            if len(line > 40):
                arr = []

                for l in line.split(";"):
                    arr.append(ast.literal_eval(l.strip()))
                self.solutions.append(arr)

        traj = open("data/trajects.txt", "r")
        for line in traj.readlines():
            if len(line > 40):
                arr = []
                
                for l in line.split(";"):
                    arr.append(ast.literal_eval(l.strip()))
                self.trajects.append(arr)

    def model_training(self):
        ...
    

def relaxed_problem(dataset):
    # MODIFIED motion planning problem

    lower_arr = [[0.0, 2.0, 2.0, 5.0, 7.5, 10.0, 12.0, 12.0, 15.0, 17.5], # x coords
                 [1.0,-5.0, 3.0,-2.0,-5.0, 1.0, -5.0,  3.0, -2.0, -5.0]]  # y coords
    
    size_arr  = [[1.5, 2.5, 2.5, 2.0, 2.0, 1.5, 2.5, 2.5, 2.0, 2.0],      # width: x
                 [2.0, 7.0, 2.0, 6.5, 6.0, 2.0, 7.0, 2.0, 6.5, 6.0]]      # height:y
    
    goal =   [20.0, 0.0, 0.0, 0.0]
    limit = [[0.0, -4.9,-1.0,-1.0], # lower[pos_x, pos_y,
             [20.0, 4.9, 1.0, 1.0]] # upper vel_x, vel_y]
    
    global_obs = mp.Obstacle_Map(lower_arr, size_arr)
    world = mp.Environment(limit, goal, global_obs, MAX=len(global_obs), TOL=0.1)

    # Randomize start, get vars & params
    for _ in range(1):

        start = world.random_state(bound = 0.5)
        print(f"\nDEBUG: world.random_state() done: {[round(x, 2) for x in start]}")

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

            d_state0, d_state1, bl_sol, bu_sol = dataset.solutions[0]

            state0.value   = np.array(d_state0)
            goal0.value    = np.array(goal)
            bool_low.value = [np.array(l) for l in bl_sol]
            bool_upp.value = [np.array(u) for u in bu_sol]

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
            print(f"\nSolving iter = {iter}")
            problem.solve(verbose = False)

            print(f"Status = {problem.status}")
            print(f"Optimal cost = {int(problem.value)}")
            print(f"Solve time = {problem.solver_stats.solve_time} secs.")

            x_sol = state.value
            u_sol = input.value

            robot.update_state(u_sol[0][0], u_sol[1][0])
            # 1st value in arr(  x_accel  ,   y_accel  )

dataset = Dataset()
relaxed_problem(dataset)