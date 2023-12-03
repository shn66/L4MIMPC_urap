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

LAYERS = 10
HIDDEN = 128
INPUTS = 24
OUTPUT = 500

ITERS = 100
BATCH = 1024
LEARN = 0.001
MODEL = "norms=1_drops=0_weigh=1_activ=leaky_relu.pth"

class Dataset:
    # solutions.pkl = [[state, local_obs, bl_sol, bu_sol], ...]

    def __init__(self):
        set, self.sols = "data/set.pkl", []

        print("\nDEBUG: Dataset initializing.")

        if os.path.exists(set):
            file = open(set, "rb")
            self.sols = pickle.load(file)
        else:
            data = [x for x in os.listdir("data") if x.startswith("sol")]

            for i in range(len(data)):
                file = open(f"data/sol{i}.pkl", "rb")
                self.sols += pickle.load(file)

            file = open(set, "wb")
            pickle.dump(self.sols, file)
            
        self.size = len(self.sols)
        print(f"DEBUG: Dataset initialized. {self.size} datapoints read.")


class BinaryNN(nn.Module):

    def __init__(self, norms, drops, activ):
        super(BinaryNN, self).__init__()

        self.norms = norms # boolean
        self.drops = drops # boolean
        self.activ = activ # function
        
        # Input size = 24:
            # lower_arr shape = (2, 5) = 10
            # size_arr  shape = (2, 5) = 10
            # state_arr shape = 4
        self.input  = nn.Linear(INPUTS, HIDDEN)
        self.normal = nn.BatchNorm1d(HIDDEN)

        self.dropout= nn.Dropout(0.1)
        self.modlst = nn.ModuleList()

        for _ in range(LAYERS): # args: (input , output)
            self.modlst.append(nn.Linear(HIDDEN, HIDDEN))
        
        # Output size = 1000:
            # bl_sol shape = (5, 2, 25) = 250
            # bu_sol shape = (5, 2, 25) = 250
        self.output = nn.Linear(HIDDEN, OUTPUT)


    def forward(self, x):
        x = self.input(x)

        if self.norms:
            x = self.normal(x)
        x = self.activ(x)

        for layer in self.modlst:
            if self.drops:
                x = self.dropout(x)
            x = self.activ(layer(x))

        x = self.output(x)
        return self.activ(x)


nums = lambda x: x.view(5, 2, 25).detach().numpy()   # Shape lst to multi-dim -> np.array
tens = lambda x: torch.Tensor(x).view(-1)            # Shape lst to tensor -> flattens 1D
outs = lambda arr, i: (nums(arr[:i]), nums(arr[i:])) # Computes bl_sol, bu_sol from model

def model_training(dataset, norms, drops, weigh, activ, optiv, model=None):
    SIZE, LIM    = dataset.size, OUTPUT // 2

    if not os.path.exists("models"):
        os.mkdir("models")

    activ_str = str(activ).split(" ")[1]
    PATH = f"models/norms={int(norms)}_drops={int(drops)}_weigh={int(weigh)}_activ={activ_str}.pth"
    
    data   = torch.zeros((SIZE, INPUTS)) # Inputs = 24 (state, obs_arrs)
    labels = torch.zeros((SIZE, OUTPUT)) # Output = 500 (bl_sol, bu_sol)

    for i in range(SIZE):                # Sample sol @ index i
        state, obs_arr, bl_sol, bu_sol = dataset.sols[i]

        data[i, :4] = tens(state)        # 4 items
        data[i, 4:] = tens(obs_arr)      # 20 items
        labels[i, :LIM] = tens(bl_sol)   # 250 items
        labels[i, LIM:] = tens(bu_sol)   # 250 items

    print(f"\nDEBUG: model_training() started. PATH =\n{PATH}")

    train_size = int(0.8 * len(data))    # Split data -> 80% train, 20% valid
    valid_size = len(data) - train_size

    td = TensorDataset(data, labels)
    train_data, valid_data = random_split(td, [train_size, valid_size])

    train_load = DataLoader(train_data, batch_size=BATCH, shuffle=True)
    valid_load = DataLoader(valid_data, batch_size=BATCH, shuffle=False)


    if not model:
        model = BinaryNN(norms, drops, activ)

    optimizer = optiv(model.parameters(), lr=LEARN)
    pos_weigh = None

    if weigh:
        pos_weigh = torch.full((OUTPUT,), 10) # weigh 1 more heavily

    nnBCELoss = nn.BCELoss() # Binary cross entropy (and sigmoid)
    logitLoss = nn.BCEWithLogitsLoss(pos_weight=pos_weigh)

    scheduler = ReduceLROnPlateau(optimizer) # Update LR
    best_loss = float("inf") # Keep best validation loss

    writer = SummaryWriter("runs")
    writer.add_text("PATH", PATH)

    for i in range(ITERS):
        model.train()
        train_loss = 0.0
        valid_loss = 0.0

        for inputs, target in train_load:
            optimizer.zero_grad()
            output = model(inputs)
            loss = logitLoss(output, target)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        

        model.eval()
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
        
        print(f"\niter = {i + 1}/{ITERS}")
        print(f"learn_rate = {round(lr, 4)}")
        print(f"train_loss = {round(tl, 4)}")
        print(f"valid_loss = {round(vl, 4)}")

    writer.close()
    print("\nmodel_training() finished.")


def get_model_outs(dataset, path, state=[], obs_arr=[], model=None):

    bools = [bool(int(x)) for x in re.findall(r'\d+', path)]
    norms, drops = bools[0], bools[1] # Term 1,2 (int->bool)

    funct = path.split("=")[-1][:-4]  # Final value (string)
    activ = eval(f"fn.{funct}")       # TODO: fix hardcoding
    
    if not model:
        model = BinaryNN(norms, drops, activ)
        load  = torch.load(f"models/{path}")
        model.load_state_dict(load)
        model.eval()

    bl_sol, bu_sol = [], []
    if (not state) or (not obs_arr): # Random sol @ index i
        
        i = random.randint(0, dataset.size - 1)
        state, obs_arr, bl_sol, bu_sol = dataset.sols[i]

    input = torch.cat((tens(state), tens(obs_arr)))
    if norms:
        input = input.unsqueeze(0)   # Add batch dim->input
    
    with torch.no_grad():
        output = torch.sigmoid(model(input))
    
    output = (output.view(-1) >= 0.5).float() # Remove batch dim, rounds to 0 or 1

    return output, obs_arr, np.array(bl_sol), np.array(bu_sol)


def test_neural_net(dataset, verbose):
    path_min, diff_min = "", float("inf")

    folder = sorted(os.listdir("models"))       # Same path/types only
    folder = [x for x in folder if len(x) > 10] # Except for .DS_Store

    for path in folder:
        diff_l, diff_u = 0.0, 0.0

        for _ in range(ITERS):
            output, _, bl_sol, bu_sol = get_model_outs(dataset, path)
            bl_out, bu_out = outs(output, OUTPUT // 2)

            diff_l += np.sum(bl_sol != bl_out) # Compares differences
            diff_u += np.sum(bu_sol != bu_out) # btwn output and data

        diff_avg = (diff_l + diff_u) / (2 * ITERS)

        if diff_avg < diff_min:
            path_min, diff_min = path, diff_avg
        
        print(f"\nDEBUG: differences in {path}:\nbl_sol = {diff_l / ITERS}")
        print(f"bu_sol = {diff_u / ITERS}\ndiff_avg = {diff_avg}")

    if verbose:
        output, obs_arr, bl_sol, bu_sol = get_model_outs(dataset, path_min)
        bl_out, bu_out = outs(output, OUTPUT // 2)

        diff_l = np.where(bl_sol != bl_out, "X", ".") # Compares differences
        diff_u = np.where(bu_sol != bu_out, "X", ".") # X is wrong . is good

        lower_x, lower_y = obs_arr[0][0], obs_arr[0][1]
        size_x , size_y  = obs_arr[1][0], obs_arr[1][1]

        for i in range(5):
            print(f"\nobs at ({lower_x[i]}, {lower_y[i]}), size ({size_x[i]}, {size_y[i]}):")
            print(f"\ndiff_l =\n{diff_l[i]}\ndiff_u =\n{diff_u[i]}")
    
    print(f"\nDEBUG: best model = {path_min}")


def relaxed_problem(use_model):
    # A MODIFIED motion planning problem

    lower_arr = [[ 0.5, 1.7, 2.7, 2.7, 3.8], # x coords
                 [-0.3,-0.7,-1.3, 0.3,-0.5]] # y coords
    
    size_arr  = [[0.7, 0.5, 0.5, 0.5, 0.7],  # width: x
                 [1.0, 0.7, 1.0, 1.0, 1.0]]  # height:y
    
    limit = [[0.0,-1.2,-0.7,-0.7], # lower[pos_x, pos_y,
             [5.0, 1.2, 0.7, 0.7]] # upper vel_x, vel_y]
    goal  =  [5.0, 0.0, 0.0, 0.0]
    
    world_obs = mp.ObsMap(lower_arr, size_arr)
    world = mp.World(limit, goal, world_obs, TOL=0.2)

    # Randomize start, get vars & params
    if use_model:
        start = world.random_state(iters=100, bound=0.9)
    else:
        i = random.randint(0, dataset.size - 1) # Random sol @ index i
        start, obs_arr, bl_sol, bu_sol = dataset.sols[i]
    
    robot = mp.Robot(start, world_obs, TIME=0.1, FOV=1.5)
    print(f"\nDEBUG: randomize start done: {[round(x, 2) for x in start]}")

    problem, vars, params = mp.motion_planning(world, robot, relaxed=True)

    state, input = vars
    bool_low, bool_upp, state0, goal0, lower_obs, upper_obs = params


    dist = lambda x: np.linalg.norm(np.array(robot.state) - np.array(x))

    # Initialize all CP.parameter values
    while dist(goal) > world.TOL:

        print(f"DEBUG: abs(distance) to goal: {round(dist(goal), 2)}")
        
        state0.value = np.array(robot.state)
        goal0.value  = np.array(goal)

        if use_model:
            robot.detect_obs()

            lower_cpy = copy.deepcopy(robot.local_obs.lower_arr)
            size_cpy  = copy.deepcopy(robot.local_obs.size_arr)

            while len(lower_cpy[0]) < world.MAX:    # Ensure arr len = MAX
                low = min(limit[0][0], limit[0][1]) # Get low within big-M
                
                for i in [0, 1]:             # Add fake obs to x(0) & y(1)
                    lower_cpy[i].append(low) # Fake obs have lower x,y val
                    size_cpy [i].append(0.0) # outside of world; size: 0.0
        else:
            lower_cpy = obs_arr[0]
            size_cpy  = obs_arr[1]

        lower_obs.value = np.array(lower_cpy)
        upper_obs.value = np.array(lower_cpy) + np.array(size_cpy)


        if use_model:
            output, _, bl_sol, bu_sol = get_model_outs(
                dataset, MODEL, robot.state, [lower_cpy, size_cpy])
            
            bl_sol, bu_sol = outs(output, OUTPUT // 2)

        ## CP.PROBLEM STUFF ##

        for i in range(world.MAX):
            bool_low[i].value = np.array(bl_sol[i])
            bool_upp[i].value = np.array(bu_sol[i])

        problem.solve(verbose=False)
        print(f"Status = {problem.status}")

        print(f"Optimal cost = {round(problem.value, 2)}")
        print(f"Solve time = {round(problem.solver_stats.solve_time, 4)}s")

        state_sol = state.value
        input_sol = input.value
        
        robot.update_state(input_sol[0][0], input_sol[1][0])
        # 1st value in arr(    x_accel    ,    y_accel     )
        world.plot_problem(state_sol, start, goal)


if __name__ == "__main__":
    dataset = Dataset()
    TEST = False
    
    if TEST:
        test_neural_net(dataset, verbose=True)
    else:
        relaxed_problem(use_model=True)