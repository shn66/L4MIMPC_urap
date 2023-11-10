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

INPUTS = 44
HIDDEN = 1024
OUTPUT = 2000

BATCH_SIZE = 1024
LEARN_RATE = 0.001
NUM_ITERS  = 100

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


class BinaryNN(nn.Module):
    """
    Experiment with hidden layers of constant width/size, but deeper / more layers
        - Last resort: extract parts of output from hidden layers & merge together
    """
    def __init__(self, hidden):
        super(BinaryNN, self).__init__()
        
        # Input size = 44:
            # lower_arr shape = (2, 10) = 20
            # size_arr  shape = (2, 10) = 20
            # state_arr shape = (4,)    = 4
        self.input  = nn.Linear(INPUTS, hidden)
        
        self.hidden = nn.ModuleList() # FIXED: 10 layers
        for _ in range(10):     # args: (input,  output)
            self.hidden.append(nn.Linear(hidden, hidden))
        
        # Output size = 2000:
            # bl_sol shape = (10, 2, 50) = 1000
            # bu_sol shape = (10, 2, 50) = 1000
        self.output = nn.Linear(hidden, OUTPUT)


    def forward(self, x):
        # Input and hidden: ReLU func
        x = torch.relu(self.input(x))

        for layer in self.hidden:
            x = torch.relu(layer(x))
        
        # Output with sigmoid activation
        return torch.sigmoid(self.output(x))


def model_training(dataset, model=None, hidden=HIDDEN, batch=BATCH_SIZE):
    """
    Experiment with 1.initial learning rate and 2.different learning rate schedulers
        - Use tensorboard to check learning rate and valid loss plots for stagnation
        - Normalize input along x and y axes before training, batch size around 1024
    """
    PATH = f"models/XXX=hidden_XXX=batch_0.XXX=loss.pth"
    RATIO = 0.8      # Split data -> 80% train, 20% valid
    S = dataset.size
    
    # Extract data & labels in dataset
    data   = torch.zeros((S, INPUTS)) # inputs = 44  (state + obs_arrs)
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

    print(f"\nDEBUG: model_training() started.\nhidden = {hidden}, batch = {batch}")

    train_size = int(RATIO * len(data))
    valid_size = len(data) - train_size


    td = TensorDataset(data, labels)
    train_data, valid_data = random_split(td, [train_size, valid_size])

    # Create data loaders
    train_load = DataLoader(train_data, batch_size=batch, shuffle=True)
    valid_load = DataLoader(valid_data, batch_size=batch, shuffle=False)

    # Model, loss, optimizer, scheduler:
    if not model:
        model = BinaryNN(hidden)

    loss_func = nn.BCELoss() # binary cross entropy loss func
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

    scheduler = ReduceLR(optimizer, patience=5)
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
        
        print(f"\niter = {i + 1}/{NUM_ITERS}")
        print(f"LR   = {optimizer.param_groups[0]['lr']}")
        print(f"train_loss = {round(train_loss / len(train_load), 4)}")
        print(f"valid_loss = {round(valid_loss / len(valid_load), 4)}")

    print("\nmodel_training() finished.")


def load_neural_net(dataset, index):
    """
    Compare dataset's bl_sol and bu_sol with neural network's to see how off we are
        - Collect samples where we're not too off and fine-tune/keep training model
    """
    BOUND = 0.5
    sols  = dataset.sols[index] # sample solution of relaxed_problem
    _, _, lower_arr, size_arr = dataset.info

    start_t = torch.Tensor(sols[0]  ).view(-1) # idx 0 = start state
    lower_t = torch.Tensor(lower_arr).view(-1)
    size_t  = torch.Tensor(size_arr ).view(-1)

    input_t = torch.cat((start_t, lower_t, size_t))

    nn_mod, nn_pth = "", ""
    nn_hid, nn_bat = 0 , 0
    bl_sol, bu_sol = [], []
    diff_min = float("inf")

    # Compare all model outputs in order
    for path in sorted(os.listdir("models")):

        split = path.split("=") # split path using "="
        hidden= int(split[0])   # hidden layer (1st #)

        split = split[1].split("_") # extract 2nd term
        batch = int(split[1])   # batch size (2nd num)

        model = BinaryNN(hidden)
        load  = torch.load(f"models/{path}")
        model.load_state_dict(load)

        model.eval()
        with torch.no_grad():
            output = model(input_t)


        output = (output > BOUND).float() # round (<0.5) to 0.0, (>0.5) to 1.0

        bl_out = output[:1000].view(10, 2, 50).tolist() # reshape to multi-dim
        bu_out = output[1000:].view(10, 2, 50).tolist() # and converts to list

        diff_l = np.sum(np.array(bl_out) != np.array(sols[1])) # compares differences
        diff_u = np.sum(np.array(bu_out) != np.array(sols[2])) # in NN and data b_sol

        diff_avg = (diff_l + diff_u) / 2
        if diff_avg < diff_min:

            nn_mod, nn_pth, nn_hid, nn_bat, bl_sol, bu_sol, diff_min = (
            model , path  , hidden, batch , bl_out, bu_out, diff_avg)
            
        print(f"\nDEBUG: differences in '{path}':\nbl_sol = {diff_l}, bu_sol = {diff_u}, diff_avg = {diff_avg}")
    
    print(f"\nDEBUG: best model = '{nn_pth}'")
    exit()
    return nn_mod, nn_hid, nn_bat, bl_sol, bu_sol


def relaxed_problem(dataset, retrain=False):
    # A MODIFIED motion planning problem

    index = random.randint(0, dataset.size - 1) # random sample solution
    start = dataset.sols[index][0]              # 0th index: start state

    limit, goal, lower_arr, size_arr = dataset.info
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

        nn_mod, nn_hid, nn_bat, bl_sol, bu_sol = load_neural_net(dataset, index)


        if retrain:
            model_training(dataset, model=nn_mod, hidden=nn_hid, batch=nn_bat)
            return

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


if __name__ == "__main__":
    dataset = Dataset()
    TRAIN = False
    if TRAIN:
        model_training(dataset)
    else:
        relaxed_problem(dataset, retrain=True)