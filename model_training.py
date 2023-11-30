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
OUTPUT = 1000

ROUND = 0.5
ITERS = 100
BATCH = 1024
LEARN = 0.001

FOLDR = "first"
MODEL = "False=drops_10.0=weigh_leaky=0.066.pth"

class Dataset:
    # solutions.pkl = [[state, local_obs, bl_sol, bu_sol], ...]

    def __init__(self):
        print("\nDEBUG: Dataset initializing.")

        self.sols = []
        data = [x for x in os.listdir("data") if x.startswith("sol")]

        for i in range(len(data)):
            file = open(f"data/solutions{i}.pkl", "rb")
            self.sols += pickle.load(file)

        self.size = len(self.sols)
        print(f"DEBUG: Dataset initialized. {self.size} datapoints read.")


class BinaryNN(nn.Module):

    def __init__(self, drops, activ):
        super(BinaryNN, self).__init__()

        self.dropout = nn.Dropout(0.1)
        self.drops   = drops
        self.activ   = activ
        
        # Input size = 24:
            # lower_arr shape = (2, 5) = 10
            # size_arr  shape = (2, 5) = 10
            # state_arr shape = 4
        self.input  = nn.Linear(INPUTS, HIDDEN)

        self.normal = nn.BatchNorm1d(HIDDEN)
        self.modlst = nn.ModuleList()

        for _ in range(LAYERS): # args: (input , output)
            self.modlst.append(nn.Linear(HIDDEN, HIDDEN))
        
        # Output size = 1000:
            # bl_sol shape = (5, 2, 50) = 500
            # bu_sol shape = (5, 2, 50) = 500
        self.output = nn.Linear(HIDDEN, OUTPUT)


    def forward(self, x):
        x = self.normal(self.input(x))
        x = self.activ(x)

        for layer in self.modlst:
            if self.drops:
                x = self.dropout(x)
            x = self.activ(layer(x))

        x = self.output(x)
        return self.activ(x)


def model_training(dataset, weigh, drops, activ, optiv, model=None):
    SIZE  = dataset.size       

    if not os.path.exists(f"models/{FOLDR}"):
        os.mkdir(f"models/{FOLDR}")

    PATH = f"models/{FOLDR}/{drops}=drops_{weigh}=weigh_{activ}.pth"
    
    data   = torch.zeros((SIZE, INPUTS)) # Inputs = 24  (state, obs_arrs)
    labels = torch.zeros((SIZE, OUTPUT)) # Output = 1000 (bl_sol, bu_sol)

    for i in range(SIZE):                # Sample sol @ index i
        state, obs_arr, bl_sol, bu_sol = dataset.sols[i]

        data[i, :4] = tens(state)        # 4 items
        data[i, 4:] = tens(obs_arr)      # 20 items
        labels[i, :500] = tens(bl_sol)   # 500 items
        labels[i, 500:] = tens(bu_sol)   # 500 items

    print(f"\nDEBUG: model_training() started. PATH =\n{PATH}")

    train_size = int(0.8 * len(data))    # Split data -> 80% train, 20% valid
    valid_size = len(data) - train_size

    td = TensorDataset(data, labels)
    train_data, valid_data = random_split(td, [train_size, valid_size])

    train_load = DataLoader(train_data, batch_size=BATCH, shuffle=True)
    valid_load = DataLoader(valid_data, batch_size=BATCH, shuffle=False)


    if not model:
        model = BinaryNN(drops, activ)

    pos_weigh = torch.full((OUTPUT,), weigh) # weight 1s more/less

    nnBCELoss = nn.BCELoss() # Binary cross entropy (and sigmoid)
    logitLoss = nn.BCEWithLogitsLoss(pos_weight=pos_weigh)

    optimizer = optiv(model.parameters(), lr=LEARN)

    scheduler = ReduceLROnPlateau(optimizer) # Update LR
    best_loss = float("inf") # Keep best validation loss

    writer = SummaryWriter(f"runs/{FOLDR}")
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

nums = lambda x: x.view(5, 2, 50).detach().numpy() # Shape lst to multi-dim -> np.array
tens = lambda x: torch.Tensor(x).view(-1)          # Shape lst to tensor -> flattens 1D

def test_neural_net(dataset, verbose):

    def get_model_outs(model):

        i = random.randint(0, dataset.size - 1) # Random sol at i
        state, obs_arr, bl_sol, bu_sol = dataset.sols[i]

        input = torch.cat((tens(state), tens(obs_arr)))
        input = input.unsqueeze(0)         # Add batch dim->input
        
        with torch.no_grad():
            output = torch.sigmoid(model(input))
        
        output = output.view(-1)           # Remove the batch dim
        output = (output >= ROUND).float() # Round ~0.5 to 0 or 1

        return output, obs_arr, np.array(bl_sol), np.array(bu_sol)

    nn_model, nn_path = None, None
    diff_min = float("inf")

    folder = sorted(os.listdir(f"models/{FOLDR}"))  # Same path/types only
    folder = [x for x in folder if len(x) > 10]     # Except for .DS_Store

    for path in folder:
        diff_l, diff_u = 0.0, 0.0

        drops = eval(path.split("=")[0])            # First term (boolean)
        activ = (path.split("_")[-1]).split("=")[0] # 2-last term (string)


        if activ == "relu": # TODO: fix hardcoding
            activ = fn.relu
        else:
            activ = fn.leaky_relu

        model = BinaryNN(drops, activ)
        load  = torch.load(f"models/{FOLDR}/{path}")

        model.load_state_dict(load)
        model.eval()

        for _ in range(ITERS):
            output, _, bl_sol, bu_sol = get_model_outs(model)

            bl_out = nums(output[:500])
            bu_out = nums(output[500:])

            diff_l += np.sum(bl_out != bl_sol) # Compares differences
            diff_u += np.sum(bu_out != bu_sol) # btwn output and data

        diff_avg = (diff_l + diff_u) / (2 * ITERS)

        if diff_avg < diff_min:
            nn_model, nn_path, diff_min = (model, path, diff_avg)
        
        print(f"\nDEBUG: differences in {FOLDR}/{path}:\nbl_sol = {diff_l / ITERS}")
        print(f"bu_sol = {diff_u / ITERS}, diff_avg = {diff_avg}")

    if verbose:
        output, obs_arr, bl_sol, bu_sol = get_model_outs(model)


        bl_out = nums(output[:500])
        bu_out = nums(output[500:])

        diff_l = np.where(bl_sol != bl_out, "X", ".")
        diff_u = np.where(bu_sol != bu_out, "X", ".")

        lower_x, lower_y = obs_arr[0][0], obs_arr[0][1]
        size_x , size_y  = obs_arr[1][0], obs_arr[1][1]

        for i in range(5):
            print(f"\nobs at ({lower_x[i]}, {lower_y[i]}), size ({size_x[i]}, {size_y[i]}):")
            print(f"\ndiff_l =\n{diff_l[i]}\ndiff_u =\n{diff_u[i]}")
    
    print(f"\nDEBUG: best model = {nn_path}")
    return nn_model


def relaxed_problem():
    # A MODIFIED motion planning problem

    lower_arr = [[ 0.5, 1.7, 2.7, 2.7, 3.8], # x coords
                 [-0.3,-0.7,-1.3, 0.3,-0.5]] # y coords
    
    size_arr  = [[0.7, 0.5, 0.5, 0.5, 0.7],  # width: x
                 [1.0, 0.7, 1.0, 1.0, 1.0]]  # height:y
    
    limit = [[0.0,-1.2,-1.0,-1.0], # lower[pos_x, pos_y,
             [5.0, 1.2, 1.0, 1.0]] # upper vel_x, vel_y]
    goal  =  [5.0, 0.0, 0.0, 0.0]
    
    world_obs = mp.ObsMap(lower_arr, size_arr)
    world = mp.World(limit, goal, world_obs, TOL=0.2)

    # Randomize start, get vars & params
    start = world.random_state(iters=100, bound=0.9)
    robot = mp.Robot(start, world_obs, TIME=0.1, FOV=1.2)

    print(f"\nDEBUG: world.random_state() done: {[round(x, 2) for x in start]}")

    problem, vars, params = mp.motion_planning(world, robot, relaxed=True)

    state, input = vars
    bool_low, bool_upp, state0, goal0, lower_obs, upper_obs = params

    dist = lambda x: np.linalg.norm(np.array(robot.state) - np.array(x))


    # Initialize all CP.parameter values
    while dist(goal) > world.TOL:

        print(f"DEBUG: abs(distance) to goal: {round(dist(goal), 2)}")
        
        state0.value = np.array(robot.state)
        goal0.value  = np.array(goal)

        ## OBSTACLE STUFF ##

        robot.detect_obs()
        lower_cpy = copy.deepcopy(robot.local_obs.lower_arr)
        size_cpy  = copy.deepcopy(robot.local_obs.size_arr)

        while (len(lower_cpy[0]) < world.MAX):
            for i in range(2):
                lower_cpy[i].append(-1.5) # [len(L) to world.MAX] are fake obs
                size_cpy[i].append(0.0)   # fake obs have lower x,y: -1.5,-1.5

        lower_obs.value = np.array(lower_cpy)
        upper_obs.value = np.array(lower_cpy) + np.array(size_cpy)

        ## NEURALNET STUFF ##

        model = BinaryNN(False, fn.leaky_relu)
        load  = torch.load(f"models/{FOLDR}/{MODEL}")

        model.load_state_dict(load)
        model.eval()

        obs = [lower_cpy, size_cpy]
        intens = torch.cat((tens(robot.state), tens(obs)))
        intens = intens.unsqueeze(0)     # Add batch dim->input
        
        with torch.no_grad():
            output = torch.sigmoid(model(intens))
        
        output = output.view(-1)           # Remove the batch dim
        output = (output >= ROUND).float() # Round ~0.5 to 0 or 1

        bl_sol = nums(output[:500])
        bu_sol = nums(output[500:])

        for i in range(world.MAX):
            bool_low[i].value = np.array(bl_sol[i])
            bool_upp[i].value = np.array(bu_sol[i])

        ## CP.PROBLEM STUFF ##
        problem.solve(verbose = False)

        print(f"Status = {problem.status}")
        print(f"Optimal cost = {round(problem.value, 2)}")
        print(f"Solve time = {round(problem.solver_stats.solve_time, 2)} sec.")

        state_sol = state.value
        input_sol = input.value
        
        world.plot_problem(state_sol, start, goal)
        robot.update_state(input_sol[0][0], input_sol[1][0])
        # 1st value in arr(    x_accel    ,    y_accel     )


if __name__ == "__main__":
    dataset = Dataset()
    TRAIN   = False

    if TRAIN:
        for weigh in [5.0, 10.0, 20.0]:
            drops = False
            activ = fn.leaky_relu
            optiv = optim.Adam
            model_training(dataset, weigh, drops, activ, optiv)
    else:
        relaxed_problem()
        exit()
        test_neural_net(dataset, verbose=True)