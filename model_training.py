import os
import copy
import torch
import random
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import motion_planning as mp
from torch.utils.data import DataLoader, TensorDataset, random_split

class Dataset:
    """
    ALL VARIABLES ARE PYTHON LISTS
    FOR BELOW: [info] = [limit, goal, lower_arr, size_arr]
    solutions.pkl = [[info], [state, bl_sol, bu_sol], ...]
    """
    def __init__(self, max_id):
        self.max_id  = max_id
        self.datamap = {
        # EACH KEY CONTAINS THESE VALUES
        #   limit = shape (2, 4)
        #   goal  = shape (4,  )
        #   lower_arr = shape (2, 10)
        #   size_arr  = shape (2, 10)
        #   solutions = [state, bl_sol, bu_sol]
        }
        print("\nDEBUG: Dataset starting.")

        for i in range(max_id):
            with open(f"data/solutions{i}.pkl", "rb") as x:
                self.datamap[i] = pickle.load(x)

        print(f"DEBUG: Dataset imported. {len(self.datamap)} iterations.")

    def solution(self, index = None): # -> [[info], [state, bl_sol, bu_sol], ...]
        if index == None:
            index = random.randint(0, self.max_id)   # random index if None given
        return self.datamap[min(index, self.max_id)] # avoids index out of bounds

INPUT  = 44
HIDDEN = 128
OUTPUT = 2000

class BinaryNN(nn.Module):
    # Num of classes = 2 because binary.
    
    def __init__(self):
        super(BinaryNN, self).__init__()
        
        # Input size = 44:
            # lower_arr shape = (2, 10) = 20
            # size_arr  shape = (2, 10) = 20
            # state_arr shape = (4,)    = 4
        self.input   = nn.Linear(INPUT, HIDDEN)
        
        # Hidden size = x -> (2 * x) -> x
            # nn.Linear args go (input, output)
        self.hidden1 = nn.Linear(HIDDEN, 2 * HIDDEN)
        self.hidden2 = nn.Linear(2 * HIDDEN, HIDDEN)
        
        # Output size = 2000:
            # bl_sol shape = (10, 2, 50) = 1000
            # bu_sol shape = (10, 2, 50) = 1000
        self.output  = nn.Linear(HIDDEN, OUTPUT)

    def forward(self, x):
        # Input, hidden: relu activation
        x = torch.relu(self.input(x))
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        
        # Output with sigmoid activation
        return torch.sigmoid(self.output(x))


def model_training(dataset):

    soln = dataset.solution(0) # -> [[info], [state, bl_sol, bu_sol], ...]
    size = len(soln[1:])

    # Extract data & labels from dataset
    data   = torch.zeros((size, INPUT))  # input  = 44:  (state, obs_arrs)
    labels = torch.zeros((size, OUTPUT)) # output = 2000: (bl_sol, bu_sol)

    for i in range(size):
        info_arr = soln[0]     # -> [limit, goal, lower_arr, size_arr]

        # Extract lower_arr & size_arr            .view(-1) flattens array into 1D
        data[i, 0: 20] = torch.Tensor(info_arr[2]).view(-1) # lower_arr = 20 items
        data[i, 20:40] = torch.Tensor(info_arr[3]).view(-1) # size_arr  = 20 items
        
        soln_arr = soln[i + 1] # -> [state, bl_sol, bu_sol]

        # Extract state from solutions
        data[i, 40:]     = torch.Tensor(soln_arr[0]) # state = 4 items
        
        # Extract bl_sol & bu_sol also              .view(-1) flattens array to 1D:
        labels[i, :1000] = torch.Tensor(soln_arr[1]).view(-1) # bl_sol = 1000 items
        labels[i, 1000:] = torch.Tensor(soln_arr[2]).view(-1) # bu_sol = 1000 items

    # Split data -> training, validation
    RATIO  = 0.8  # 80 % train, 20 % valid

    train_size = int(RATIO * len(data))
    valid_size   = len(data) - train_size

    td = TensorDataset(data, labels)
    train_data, valid_data = random_split(td, [train_size, valid_size])


    BATCH_SIZE = 32
    LEARN_RATE = 0.001
    NUM_ITERS  = 100

    # Create data loaders
    train_load = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_load = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

    # Create NN model, loss function, optimizer
    model = BinaryNN()
    criterion = nn.BCELoss() # binary cross entropy loss func
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

    # Start training loop
    for i in range(NUM_ITERS):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_load:
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Start validation loop
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():

            for inputs, targets in valid_load:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()
        
        print(f"\nIter = {i + 1}/{NUM_ITERS}")
        print(f"Train Loss = {train_loss / len(train_load):.2f}")
        print(f"Valid Loss = {valid_loss / len(valid_load):.2f}")

    print("\nmodel_training() done :)")


def relaxed_problem(dataset):
    # A MODIFIED motion planning problem

    data = dataset.solution(index = 1) # -> [[info], [state, bl_sol, bu_sol] ...]
    limit, goal, lower_arr, size_arr = data[0] # info: [limit, goal, lower, size]

    index = random.randint(1, len(data) - 1)  # random solutions array (data[1:])
    start, bl_sol, bu_sol = data[index]       # which is: [state, bl_sol, bu_sol]

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
    max_id  = len(os.listdir("data")) - 1
    dataset = Dataset(max_id)

    model_training(dataset)
    # relaxed_problem(dataset)