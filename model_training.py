import ast
import copy
import torch
import random
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import motion_planning as mp
from torch.utils.data import DataLoader

class Dataset:
    """
    ALL DATATYPES ARE PYTHON LISTS
    FOR BELOW: [data] = [limit, goal, lower_arr, size_arr]

    solutions.pkl = [[data], [state, bl_sol, bu_sol], ...]
    trajects.pkl  = [[data], [state_traj, input_traj] ...]
    """
    def __init__(self):
        self.limit = []
        self.goal  = []
        self.lower_arr = []
        self.size_arr  = []
        self.solutions = []
        self.trajects  = []

        with open("data/solutions.pkl", "rb") as x:
            soln = pickle.load(x)
            self.solutions = soln[1:]
            self.limit, self.goal, self.lower_arr, self.size_arr = soln[0]

        with open("data/trajects.pkl", "rb") as x:
            self.trajects = pickle.load(x)[1:]

        print(f"DEBUG: Dataset created. {len(self.solutions)} solutions.")


    def select(self, index = None):               # -> [state, bl_sol, bu_sol]
        max_id = len(self.solutions) - 1          # avoids index out of bounds

        if index == None:
            index = random.randint(0, max_id)     # random index if None given
        return self.solutions[min(index, max_id)] # select item in array, etc.


def relaxed_problem(dataset):
    # A MODIFIED motion planning problem

    limit = dataset.limit
    goal  = dataset.goal
    lower_arr = dataset.lower_arr
    size_arr  = dataset.size_arr

    start, bl_sol, bu_sol = dataset.select(index = 0)
    global_obs = mp.Obstacle_Map(lower_arr, size_arr)

    world = mp.Environment(limit, goal, global_obs, TOL = 0.1)
    robot = mp.Robot(start, global_obs, TIME=0.2, FOV=10.0)

    # Create problem, get vars & params
    problem, vars, params = mp.motion_planning(world, robot, relaxed=True)

    state, input = vars
    bool_low, bool_upp, state0, goal0, obs_lower, obs_upper = params

    dist = lambda x: np.linalg.norm(np.array(robot.state) - np.array(x))

    # Initialize all CP.parameter values
    while dist(goal) > world.TOL:

        print(f"DEBUG: abs(distance) to goal: {round(dist(goal), 2)}")
        
        state0.value = np.array(robot.state)
        goal0.value  = np.array(goal)


        # TODO: implement bl_sol, bu_sol = Neural_Network(dataset)

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
        print(f"Optimal cost = {round(problem.value, 2)}")
        print(f"Solve time = {round(problem.solver_stats.solve_time, 2)} secs.")

        x_sol = state.value
        u_sol = input.value

        robot.update_state(u_sol[0][0], u_sol[1][0])
        # 1st value in arr(  x_accel  ,   y_accel  )
        world.plot_problem(x_sol, start, goal)

if __name__ == "__main__":
    dataset = Dataset()
    relaxed_problem(dataset)