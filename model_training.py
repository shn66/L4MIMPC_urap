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

INPUTS = 24
OUTPUT = 1000
NUM_ITERS  = 100
LEARN_RATE = 0.001

LAYERS = 10
HIDDEN = 128
BATCH  = 1024
DIR = "final"

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


def model_training(dataset, drops, weigh, activ, optiv, model=None):
    SIZE  = dataset.size
    BOUND = 0.8          # Split data -> 80% train, 20% valid

    if not os.path.exists(f"models/{DIR}"):
        os.mkdir(f"models/{DIR}")

    PATH = f"models/{DIR}/{drops}=drops_{weigh}=weigh.pth"
    
    data   = torch.zeros((SIZE, INPUTS)) # Inputs = 24  (state, obs_arrs)
    labels = torch.zeros((SIZE, OUTPUT)) # Output = 1000 (bl_sol, bu_sol)

    for i in range(SIZE):                # Sample sol @ index i
        state, obs_arr, bl_sol, bu_sol = dataset.sols[i]

        tens = lambda x: torch.Tensor(x).view(-1) # flat tensor

        data[i, :4] = tens(state)        # 4 items
        data[i, 4:] = tens(obs_arr)      # 20 items
        labels[i, :500] = tens(bl_sol)   # 500 items
        labels[i, 500:] = tens(bu_sol)   # 500 items

    print(f"\nDEBUG: model_training() started. PATH =\n{PATH}")

    train_size = int(BOUND * len(data))
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

    optimizer = optiv(model.parameters(), lr=LEARN_RATE)

    scheduler = ReduceLROnPlateau(optimizer) # Update LR
    best_loss = float("inf") # Keep best validation loss

    writer = SummaryWriter(f"runs/{DIR}")
    writer.add_text(DIR, f"{drops}=drops_{weigh}=weigh")

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


def test_neural_net(dataset, verbose):

    def get_model_outs(model):
        BOUND = 0.5

        i = random.randint(0, dataset.size - 1)   # Random soln
        state, obs_arr, bl_sol, bu_sol = dataset.sols[i]

        tens = lambda x: torch.Tensor(x).view(-1) # Flat tensor

        input = torch.cat((tens(state), tens(obs_arr)))
        input = input.unsqueeze(0)         # Add batch dim->input
        
        with torch.no_grad():
            output = torch.sigmoid(model(input))
        
        output = output.view(-1)           # Remove the batch dim
        output = (output >= BOUND).float() # Round ~0.5 to 0 or 1

        return output, obs_arr, np.array(bl_sol), np.array(bu_sol)

    # Reshape to multi-dim, convert to np.array
    nums = lambda x: x.view(5, 2, 50).detach().numpy()

    def view_model_diff(model):
        output, obs_arr, bl_sol, bu_sol = get_model_outs(model)

        bl_out = nums(output[:500])
        bu_out = nums(output[500:])

        diff_l = np.where(bl_sol != bl_out, "X", ".")
        diff_u = np.where(bu_sol != bu_out, "X", ".")

        print(f"\nDEBUG: differences in output:")

        for i in range(5):
            lx, ly = obs_arr[0][0][i], obs_arr[0][1][i]
            sx, sy = obs_arr[1][0][i], obs_arr[1][1][i]

            print(f"obs @({lx}, {ly}), size ({sx}, {sy}): \ndiff_l = \n{diff_l[i]} \ndiff_u = \n{diff_u[i]}")

    nn_model, nn_path = None, None
    diff_min = float("inf")

    folder = sorted(os.listdir(f"models/{DIR}")) # Same path/types only
    folder = [x for x in folder if len(x) > 10]  # Except for .DS_Store


    for path in folder:
        diff_l, diff_u = 0.0, 0.0
        drops = eval(path.split("=")[0])

        model = BinaryNN(drops, fn.leaky_relu) # TODO: fix hardcoding
        load  = torch.load(f"models/{DIR}/{path}")

        model.load_state_dict(load)
        model.eval()

        for _ in range(NUM_ITERS):
            output, _, bl_sol, bu_sol = get_model_outs(model)

            bl_out = nums(output[:500])
            bu_out = nums(output[500:])

            diff_l += np.sum(bl_out != bl_sol) # Compares differences
            diff_u += np.sum(bu_out != bu_sol) # btwn output and data

        diff_avg = (diff_l + diff_u) / (2 * NUM_ITERS)

        if diff_avg < diff_min:
            nn_model, nn_path, diff_min = (model, path, diff_avg)
        
        print(f"\nDEBUG: differences in {DIR}/{path}: \nbl_sol = {diff_l / NUM_ITERS}, bu_sol = {diff_u / NUM_ITERS}, diff_avg = {diff_avg}")

    print(f"\nDEBUG: best model = {nn_path}")

    if verbose:
        view_model_diff(nn_model)
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
    TRAIN   = True

    if TRAIN:
        for drops in [True, False]:
            for weigh in [0.1, 10.0]:
                for activ in [fn.relu, fn.leaky_relu]:
                    model_training(dataset, drops, weigh, activ, optim.Adam)
    else:
        test_neural_net(dataset, verbose=True)