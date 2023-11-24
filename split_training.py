import os
import re
import copy
import torch
import random
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import motion_planning as mp
import torch.nn.functional as fn
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split

INPUTS = 44
NUM_ITERS  = 100
LEARN_RATE = 0.001

DIR = "focus_norm" # Works on anything except new_models
SPLIT  = DIR == "split_training"
OUTPUT = 1000 if SPLIT else 2000

class Dataset:
    """
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

    
    def normalize(self):
        exit(); print("\nDEBUG: Dataset normalizing.")

        lower_arr = self.info[2]
        size_arr  = self.info[3]

        norm = lambda x, min, max: 2*(x-min)/(max-min)-1 # Normalize between [-1, 1]

        lower_arr[0] = [norm(x,-2.0,17.5) for x in lower_arr[0]] # range[-2.0, 17.5]
        size_arr[0]  = [norm(x, 0.0, 2.5) for x in size_arr[0]]  # range[0.0, 2.5]

        lower_arr[1] = [norm(y,-5.0, 5.0) for y in lower_arr[1]] # range[-5.0, 5.0]
        size_arr[1]  = [norm(y, 0.0, 7.0) for y in size_arr[1]]  # range[0.0, 7.0]

        for i in range(self.size):
            state = self.sols[i][0]

            state[0] = norm(state[0], 0.0, 20.0) # range[0.0, 20.0]
            state[1] = norm(state[1], -5.0, 5.0) # range[-5.0, 5.0]


class BinaryNN(nn.Module):

    def __init__(self, layers, hidden):
        super(BinaryNN, self).__init__()
        
        # Input size = 44:
            # lower_arr shape = (2, 10) = 20
            # size_arr  shape = (2, 10) = 20
            # state_arr shape = (4,)    = 4
        self.input  = nn.Linear(INPUTS, hidden)

        self.normal = nn.BatchNorm1d(hidden)
        self.modlst = nn.ModuleList()

        for _ in range(layers): # args: (input,  output)
            self.modlst.append(nn.Linear(hidden, hidden))
        
        # Output size = 2000:
            # bl_sol shape = (10, 2, 50) = 1000
            # bu_sol shape = (10, 2, 50) = 1000
        self.output = nn.Linear(hidden, OUTPUT)


    def forward(self, x):
        x = self.normal(self.input(x))
        x = torch.relu(x)

        for layer in self.modlst:
            x = torch.relu(layer(x))
        return self.output(x)


def model_training(dataset, is_low, layers=10, hidden=100, batch=1024, model=None):
    SIZE  = dataset.size
    BOUND = 0.8          # Split data -> 80% train, 20% valid

    if not os.path.exists(f"models/{DIR}"):
        os.mkdir(f"models/{DIR}")

    PATH = f"models/{DIR}/{layers}=layers_{hidden}=hidden_{batch}=batch.pth"
    if SPLIT:
        PATH = f"models/{DIR}/{is_low}=is_low_{layers}=layers_{hidden}=hidden_{batch}=batch.pth"

    data   = torch.zeros((SIZE, INPUTS)) # Inputs = 44  (state, obs_arrs)
    labels = torch.zeros((SIZE, OUTPUT)) # Output = 1000 (bl_sol, bu_sol)

    for i in range(SIZE):
        sols = dataset.sols[i]           # Sample sol @ index i

        state, bl_sol, bu_sol = sols[0], sols[1], sols[2]
        _, _, lower_arr, size_arr = dataset.info

        tens = lambda x: torch.Tensor(x).view(-1)

        data[i, 0: 20] = tens(lower_arr) # 20 items
        data[i, 20:40] = tens(size_arr ) # 20 items
        data[i, 40:44] = tens(state)     # 4. items

        if SPLIT:                 # 1000 items each
            labels[i] = tens(bl_sol) if is_low else tens(bu_sol)
        else:
            labels[i, :1000] = tens(bl_sol)
            labels[i, 1000:] = tens(bu_sol)


    print(f"\nDEBUG: model_training() started. PATH =\n{PATH}")

    train_size = int(BOUND * len(data))
    valid_size = len(data) - train_size

    td = TensorDataset(data, labels)
    train_data, valid_data = random_split(td, [train_size, valid_size])

    train_load = DataLoader(train_data, batch_size=batch, shuffle=True)
    valid_load = DataLoader(valid_data, batch_size=batch, shuffle=False)

    if not model:
        model = BinaryNN(layers, hidden)

    pos = labels.sum(dim=0)  # Create positive weights for labels
    pos_weigh = (labels.size(0) - pos) / pos

    nnBCELoss = nn.BCELoss() # Binary cross entropy (and sigmoid)
    logitLoss = nn.BCEWithLogitsLoss(pos_weight=pos_weigh)

    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

    scheduler = ReduceLROnPlateau(optimizer) # Update LR
    best_loss = float("inf") # Keep best validation loss

    writer = SummaryWriter(f"runs/{DIR}")
    writer.add_text(DIR, f"{layers} = layers, {hidden} = hidden, {batch} = batch")

    for i in range(NUM_ITERS):
        model.train()
        train_loss = 0.0

        for inputs, target in train_load:
            optimizer.zero_grad()
            output = model(inputs)
            loss = logitLoss(output, target)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():

            for inputs, target in valid_load:
                output = torch.sigmoid(model(inputs))

                loss = nnBCELoss(output, target)
                valid_loss += loss.item()

        scheduler.step(valid_loss)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), PATH)

        lr = optimizer.param_groups[0]["lr"]
        tl = train_loss / len(train_load)
        vl = valid_loss / len(valid_load)

        writer.add_scalar("learn_rate", lr, i)
        writer.add_scalar("train_loss", tl, i)
        writer.add_scalar("valid_loss", vl, i)
        
        print(f"\niter = {i + 1}/{NUM_ITERS}")
        print(f"learn_rate = {round(lr, 4)}")
        print(f"train_loss = {round(tl, 4)}")
        print(f"valid_loss = {round(vl, 4)}")

    writer.close()
    print("\nmodel_training() finished.")


def test_neural_net(dataset, verbose, retrain):

    def get_model_info(path):
        is_low = False
        if SPLIT:
            is_low = eval(path.split("=")[0])    # Find is_low(1st num)

        digits = re.findall(r"(\d+)=", path)     # Find digit after "="
        layers, hidden, batch = map(int, digits)

        return is_low, layers, hidden, batch
    
    def get_model_outs(model):
        BOUND = 0.5          # Sample solution using random index
        index = random.randint(0, dataset.size - 1)

        sols  = dataset.sols[index]
        state, bl_sol, bu_sol = sols
        _, _, lower_arr, size_arr = dataset.info

        start = torch.Tensor(state).view(-1)
        lower = torch.Tensor(lower_arr).view(-1)
        size  = torch.Tensor(size_arr ).view(-1)

        input = torch.cat((start, lower, size))
        input = input.unsqueeze(0)         # Add batch dim->input
        
        with torch.no_grad():
            output = torch.sigmoid(model(input))
        
        output = output.view(-1)           # Remove the batch dim
        output = (output >= BOUND).float() # Rounds (< 0.5) to 0, (>= 0.5) to 1

        return output, np.array(bl_sol), np.array(bu_sol)

    # Reshape to multi-dim, convert to np.array
    nums = lambda x: x.view(10, 2, 50).detach().numpy()

    def view_model_diff(model):
        output, bl_sol, bu_sol = get_model_outs(model)

        bl_out = nums(output[:1000])
        bu_out = nums(output[1000:])

        diff_l = np.where(bl_sol != bl_out, "X", ".")
        diff_u = np.where(bu_sol != bu_out, "X", ".")

        _, _, lower_arr, size_arr = dataset.info
        print(f"\nDEBUG: differences in output:")

        for i in range(10):
            lx, ly = lower_arr[0][i], lower_arr[1][i]
            sx, sy =  size_arr[0][i],  size_arr[1][i]
            print(f"obs @({lx}, {ly}), size ({sx}, {sy}): \ndiff_l = \n{diff_l[i]} \ndiff_u = \n{diff_u[i]}")

    nn_model, nn_path = None, None
    diff_min = float("inf")

    folder = sorted(os.listdir(f"models/{DIR}")) # Same path/types only
    folder = [x for x in folder if len(x) > 10]  # Except for .DS_Store

    for path in folder:
        diff_l, diff_u = 0.0, 0.0

        is_low, layers, hidden, batch = get_model_info(path)

        model = BinaryNN(layers, hidden)
        load = torch.load(f"models/{DIR}/{path}")

        model.load_state_dict(load)
        model.eval()

        for _ in range(NUM_ITERS):
            output, bl_sol, bu_sol = get_model_outs(model)

            bl_out = nums(output) if SPLIT else nums(output[:1000])
            bu_out = nums(output) if SPLIT else nums(output[1000:])

            diff_l += np.sum(bl_out != bl_sol) # Compares differences
            diff_u += np.sum(bu_out != bu_sol) # btwn output and data

        if SPLIT:
            if is_low:
                print(f"\nDEBUG: differences in '{DIR}/{path}':\nbl_sol = {diff_l / NUM_ITERS}")
            else:
                print(f"\nDEBUG: differences in '{DIR}/{path}':\nbu_sol = {diff_u / NUM_ITERS}")
        else:
            diff_avg = (diff_l + diff_u) / (2 * NUM_ITERS)

            if diff_avg < diff_min:
                nn_model, nn_path, diff_min = (model, path, diff_avg)
            
            print(f"\nDEBUG: differences in {DIR}/{path}: \nbl_sol = {diff_l / NUM_ITERS}, bu_sol = {diff_u / NUM_ITERS}, diff_avg = {diff_avg}")
    
    if not SPLIT:
        print(f"\nDEBUG: best model = {nn_path}")
        if verbose:
            view_model_diff(nn_model)

        if retrain:
            is_low, layers, hidden, batch = get_model_info(nn_path)
            model_training(dataset, is_low, layers, hidden, batch)
        return nn_model


def relaxed_problem(dataset):
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

        model = test_neural_net(dataset, index) # TODO: use neural net

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
    TRAIN   = False

    if TRAIN:
        for is_low in [False, True]:
            model_training(dataset, is_low)
    else:
        test_neural_net(dataset, verbose=True, retrain=False)