import os
import copy
import torch
import random
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import motion_planning as mp
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split

INPUTS = 44
LAYERS = 10
OUTPUT = 2000
NUM_ITERS  = 100
LEARN_RATE = 0.001

LOGITS = True  # LOGITS must be True
NORMAL = True  # if NORMAL is = True

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

    
    def normalize(self, index):
        lower_arr = copy.deepcopy(self.info[2])
        size_arr  = copy.deepcopy(dataset.info[3])
        state = copy.deepcopy(self.sols[index][0])

        lower_arr[0] = [(x-7.75) / 9.75 for x in lower_arr[0]]  # 20 = range[-2.0, 17.5]
        size_arr[0]  = [(x-1.25) / 2.5 for x in size_arr[0]]    # 2.5 = range[0.0, 2.5]

        # along y-axis
        lower_arr[1] = [y / 5 for y in lower_arr[1]]            # 5 = range[-5.0, 5.0]
        size_arr[1]  = [(y-3.5) / 3.5 for y in size_arr[1]]     # 7 = range[0.0, 7.0]

        state[0] = (state[0]-10)/ 10                            # 20 = range[0.0, 20.0]
        state[1] /= 5                                           # 5 = range[-5.0, 5.0]

        return state, lower_arr, size_arr


class BinaryNN(nn.Module):

    def __init__(self, hidden):
        super(BinaryNN, self).__init__()
        
        # Input size = 44:
            # lower_arr shape = (2, 10) = 20
            # size_arr  shape = (2, 10) = 20
            # state_arr shape = (4,)    = 4
        self.input  = nn.Linear(INPUTS, hidden)
        
        self.hidden = nn.ModuleList()
        for _ in range(LAYERS): # args: (input,  output)
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
        if LOGITS:
            return self.output(x)
        return torch.sigmoid(self.output(x))


def get_model_type():
    if not LOGITS and NORMAL:
        print("\nERROR: LOGITS must be True if NORMAL is = True"); exit()
    elif LOGITS and NORMAL:
        return "normal"
    elif LOGITS and not NORMAL:
        return "logits"
    else:
        return "first"


def model_training(dataset, hidden, batch, model=None):
    TYPE = get_model_type()
    SIZE = dataset.size

    PATH = f"models/{TYPE}/{hidden}=hidden_{batch}=batch_NEW=loss.pth"
    BOUND= 0.8 # Split data -> 80% train, 20% valid
    
    # Extract data & labels in dataset
    data   = torch.zeros((SIZE, INPUTS)) # inputs = 44  (state + obs_arrs)
    labels = torch.zeros((SIZE, OUTPUT)) # output = 2000 (bl_sol + bu_sol)

    for i in range(SIZE):
        lower_arr = dataset.info[2]
        size_arr  = dataset.info[3]

        sols  = dataset.sols[i]
        state, bl_sol, bu_sol = sols[0], sols[1], sols[2]

        if NORMAL:
            state, lower_arr, size_arr = dataset.normalize(i)

        # Extract lower_arr & size_arr          .view(-1): arr to 1D
        data[i, 0: 20] = torch.Tensor(lower_arr).view(-1) # 20 items
        data[i, 20:40] = torch.Tensor(size_arr ).view(-1) # 20 items

        # Extract state, bl and bu_sol
        data[i, 40:44]   = torch.Tensor(state ).view(-1) # 4 items
        
        # Extract bl_sol & bu_sol also         .view(-1): arr into 1D
        labels[i, :1000] = torch.Tensor(bl_sol).view(-1) # 1000 items
        labels[i, 1000:] = torch.Tensor(bu_sol).view(-1) # 1000 items


    print(f"\nDEBUG: model_training() started.\n{TYPE} = TYPE, {hidden} = hidden, {batch} = batch")

    train_size = int(BOUND * len(data))
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
    if LOGITS:
        loss_func = nn.BCEWithLogitsLoss() # BCE with sigmoid

    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
    scheduler = ReduceLROnPlateau(optimizer)
    best_loss = float('inf') # track the best validation loss


    writer = SummaryWriter(f"runs/{TYPE}")

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

            for inputs, target in valid_load:
                output = model(inputs)

                if LOGITS: # needs sigmoid fn also
                    output = torch.sigmoid(output)

                loss = loss_func(output, target)
                valid_loss += loss.item()


        scheduler.step(valid_loss) # update scheduler and LR
        
        if valid_loss < best_loss: # saves least lossy model
            best_loss = valid_loss
            torch.save(model.state_dict(), PATH)

        lr = optimizer.param_groups[0]['lr']
        tl = train_loss / len(train_load)
        vl = valid_loss / len(valid_load)

        writer.add_scalar('learn_rate', lr, i)
        writer.add_scalar('train_loss', tl, i)
        writer.add_scalar('valid_loss', vl, i)
        
        print(f"\niter = {i + 1}/{NUM_ITERS}")
        print(f"learn_rate = {round(lr, 4)}")
        print(f"train_loss = {round(tl, 4)}")
        print(f"valid_loss = {round(vl, 4)}")

    writer.close()
    print("\nmodel_training() finished.")


def load_neural_net(dataset, index):
    TYPE  = get_model_type()
    BOUND = 0.5

    sols  = dataset.sols[index] # sample solution of relaxed_problem
    #_, _, lower_arr, size_arr = dataset.info

    state, lower_arr, size_arr = dataset.normalize(index)

    start_t = torch.Tensor(state).view(-1) # idx 0 = start state
    lower_t = torch.Tensor(lower_arr).view(-1)
    size_t  = torch.Tensor(size_arr ).view(-1)

    input_t = torch.cat((start_t, lower_t, size_t))

    nn_mod, nn_pth = "", ""
    nn_hid, nn_bat = 0 , 0
    bl_sol, bu_sol = [], []
    diff_min = float("inf")

    # Compare all model outputs in order
    folder = sorted(os.listdir(f"models/{TYPE}"))
    folder = [x for x in folder if len(x) > 10] # avoids .DS_Store

    for path in folder:
        split = path.split("=") # split path using "="
        hidden= int(split[0])   # hidden layer (1st #)

        split = split[1].split("_") # extract 2nd term
        batch = int(split[1])   # batch size (2nd num)

        model = BinaryNN(hidden)
        load  = torch.load(f"models/{TYPE}/{path}")
        model.load_state_dict(load)


        model.eval()
        with torch.no_grad():
            output = model(input_t)

            if LOGITS: # needs sigmoid fn also
                output = torch.sigmoid(output)

        output = (output >= BOUND).float() # round (<0.5) to 0.0, (>=0.5) to 1.0

        bl_out = output[:1000].view(10, 2, 50).tolist() # reshape to multi-dim
        bu_out = output[1000:].view(10, 2, 50).tolist() # and converts to list

        diff_l = np.sum(np.array(bl_out) != np.array(sols[1])) # compares differences
        diff_u = np.sum(np.array(bu_out) != np.array(sols[2])) # in NN and data b_sol
        diff_avg = (diff_l + diff_u) / 2

        if diff_avg < diff_min:
            nn_mod, nn_pth, nn_hid, nn_bat, bl_sol, bu_sol, diff_min = (
            model , path  , hidden, batch , bl_out, bu_out, diff_avg)
            
        print(f"\nDEBUG: differences in '{TYPE}/{path}':\nbl_sol = {diff_l}, bu_sol = {diff_u}, diff_avg = {diff_avg}")
    
    print(f"\nDEBUG: best model = '{TYPE}/{nn_pth}'")

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
            model_training(dataset, nn_hid, nn_bat, model=nn_mod)
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
    TRAIN   = True
    RETRAIN = True

    if TRAIN:
        model_training(dataset, hidden=128, batch=128)
    else:
        relaxed_problem(dataset, RETRAIN)