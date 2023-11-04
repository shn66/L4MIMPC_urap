import os, copy
import torch
import random
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import motion_planning as mp
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLR
from torch.utils.data import DataLoader, TensorDataset, random_split

class Dataset:
    """
    ALL VARIABLES FROM PKL FILES ARE PYTHON LISTS
    info.pkl = [limit, goal, lower_arr, size_arr]
    solutions.pkl = [[state, bl_sol, bu_sol] ...]
    """
    def __init__(self):
        print("\nDEBUG: Dataset initializing.")

        file = open("data/info.pkl", "rb")
        self.info = pickle.load(file)

        data = [x for x in os.listdir("data") if x.startswith("sol")]
        self.sols = []

        for i in range(len(data)):
            file = open(f"data/solutions{i}.pkl", "rb")
            self.sols += (pickle.load(file))

        self.size = len(self.sols)
        print(f"DEBUG: Dataset initialized. {self.size} datapoints read.")

INPUT  = 44
HIDDEN = 128
OUTPUT = 2000

class BinaryNN(nn.Module):
    
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
    S = dataset.size

    # Extract data & labels in dataset
    data   = torch.zeros((S, INPUT))  # input  = 44  (state + obs_arrs)
    labels = torch.zeros((S, OUTPUT)) # output = 2000 (bl_sol + bu_sol)

    for i in range(S):
        # Extract lower_arr & size_arr                .view(-1) flattens array into 1D
        data[i, 0: 20] = torch.Tensor(dataset.info[2]).view(-1) # lower_arr = 20 items
        data[i, 20:40] = torch.Tensor(dataset.info[3]).view(-1) # size_arr  = 20 items

        sols = dataset.sols[i] # Extract state from solutions
        data[i, 40:44] = torch.Tensor(sols[0])  # state = 4 items
        
        # Extract bl_sol & bu_sol also          .view(-1) flattens array to 1D:
        labels[i, :1000] = torch.Tensor(sols[1]).view(-1) # bl_sol = 1000 items
        labels[i, 1000:] = torch.Tensor(sols[2]).view(-1) # bu_sol = 1000 items

    RATIO = 0.8 # Split data -> 80% train, 20% valid

    train_size = int(RATIO * len(data))
    valid_size = len(data) - train_size

    td = TensorDataset(data, labels)
    train_data, valid_data = random_split(td, [train_size, valid_size])

    BATCH_SIZE = 32
    LEARN_RATE = 0.01
    NUM_ITERS  = 100

    print(f"\nDEBUG: model_training() started.\nBATCH_SIZE = {BATCH_SIZE}, LEARN_RATE = {LEARN_RATE}")


    # Create data loaders
    train_load = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_load = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

    # Create model, loss function, optimizer
    model = BinaryNN()
    loss_func = nn.BCELoss() # binary cross entropy loss func
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

    scheduler = ReduceLR(optimizer, "min")# ReduceLROnPlateau
    best_loss = float('inf') # track the best validation loss

    # Start training loop
    for i in range(NUM_ITERS):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_load:
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Start validation loop
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():

            for inputs, targets in valid_load:
                outputs = model(inputs)
                loss = loss_func(outputs, targets)
                valid_loss += loss.item()
        
        scheduler.step(valid_loss) # update scheduler and LR
        
        if valid_loss < best_loss: # saves least lossy model
            best_loss = valid_loss
            torch.save(model.state_dict(), PATH)
        
        print(f"\niteration  = {i + 1}/{NUM_ITERS}")
        print(f"train_loss = {round(train_loss / len(train_load), 4)}")
        print(f"valid_loss = {round(valid_loss / len(valid_load), 4)}")

    print("\nmodel_training() finished.")


def load_neural_net(start, lower_arr, size_arr):
    BOUND = 0.5

    start_t = torch.Tensor(start).view(-1)
    lower_t = torch.Tensor(lower_arr).view(-1)
    size_t  = torch.Tensor(size_arr ).view(-1)

    input = torch.cat((start_t, lower_t, size_t))

    model = BinaryNN()
    model.load_state_dict(torch.load(PATH))
    model.eval()
    
    with torch.no_grad():
        output = model(input)

    output = (output > BOUND).float()

    bl_sol  = output[:1000].view(10, 2, 50).tolist()
    bu_sol  = output[1000:].view(10, 2, 50).tolist()

    print(f"\nDEBUG: bl_sol = {bl_sol}")
    print(f"\nDEBUG: bu_sol = {bu_sol}")

    return bl_sol, bu_sol


def relaxed_problem(dataset):
    # A MODIFIED motion planning problem

    limit, goal, lower_arr, size_arr = dataset.info

    index = random.randint(0, dataset.size - 1) # random solutions array
    start = dataset.sols[index][0]              # 0th index: start state

    global_obs = mp.ObsMap(lower_arr, size_arr)

    world = mp.World(limit, goal, global_obs, TOL=0.1)
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

        bl_sol, bu_sol = load_neural_net(start, lower_arr, size_arr)


        for i in range(world.MAX):
            bool_low[i].value = np.array(bl_sol[i])
            bool_upp[i].value = np.array(bu_sol[i])

        robot.detect_obs()
        lower = copy.deepcopy(robot.local_obs.lower_arr)
        size = copy.deepcopy(robot.local_obs.size_arr)

        while (len(lower[0]) < world.MAX):
            for i in range(2):
                lower[i].append(-2.0) # [len(L) to world.MAX] are fake obs
                size[i].append(0.0)   # fake obs have lower x,y: -2.0,-2.0

        obs_lower.value = np.array(lower)
        obs_upper.value = np.array(lower) + np.array(size)

        # Now collect optimized trajectories
        print(f"\nSolving problem...")
        problem.solve(verbose = False)


        print(f"Status = {problem.status}")
        print(f"Optimal cost = {round(problem.value, 2)}")
        print(f"Solve time = {round(problem.solver_stats.solve_time, 2)} secs.")

        state_sol = state.value
        input_sol = input.value
        bl_sol, bu_sol = [], []

        robot.update_state(input_sol[0][0], input_sol[1][0])
        # 1st value in arr(    x_accel    ,    y_accel     )
        world.plot_problem(state_sol, start, goal)

"""
Data from training:

BATCH_SIZE || 128  || 128  ||  64  ||  64  ||  32  ||  32
LEARN_RATE || .01  || .001 || .01  || .001 || .01  || .001
VALID_LOSS ||.1219 ||.1119 ||.1238 ||.1083 ||.1305 ||.1102

Looks like lower batch_size and learn_rate -> lower loss
"""
PATH = "data/model_64_0.001_0.1083.pth"

if __name__ == "__main__":
    dataset = Dataset()
    TRAIN = False

    if TRAIN:
        model_training(dataset)
    else:
        relaxed_problem(dataset)