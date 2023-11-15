import os, copy
import torch
import random
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import motion_planning as mp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split

INPUTS = 44
LAYERS = 10
OUTPUT = 2000
NUM_ITERS  = 100
LEARN_RATE = 0.001

LOGITS = True  # LOGITS must be True
Normal = False # if NORMAL is = True

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


    def get_type(self):
        if not LOGITS and Normal:
            print("\nERROR: LOGITS must be True if NORMAL is True"); exit()
        elif LOGITS and Normal:
            return "normal"
        elif LOGITS and not Normal:
            return "logits"
        else:
            return "first"



class BinaryNN(nn.Module):

    def __init__(self, hidden):
        super(BinaryNN, self).__init__()
        
        # Input size = 44:
            # lower_arr shape = (2, 10) = 20
            # size_arr  shape = (2, 10) = 20
            # state_arr shape = (4,)    = 4
        self.input  = nn.Linear(INPUTS, hidden)

        self.norm_1 = nn.BatchNorm1d(hidden)
        self.modlst = nn.ModuleList()

        for _ in range(LAYERS): # args: (input,  output)
            self.modlst.append(nn.Linear(hidden, hidden))
            self.modlst.append(nn.BatchNorm1d(hidden))
        
        # Output size = 2000:
            # bl_sol shape = (10, 2, 50) = 1000
            # bu_sol shape = (10, 2, 50) = 1000
        self.output = nn.Linear(hidden, OUTPUT)
        self.norm_2 = nn.BatchNorm1d(OUTPUT)


    def forward(self, x):
        # Input, hidden: ReLU activation
        x = self.input(x)
        if Normal:
            x = self.norm_1(x)
        x = torch.relu(x)

        for i in range(0, len(self.modlst), 2):
            x = self.modlst[i](x)         # Apply linear layers
            if Normal:
                x = self.modlst[i + 1](x) # Batch normalization
            x = torch.relu(x)
        
        # Output with sigmoid activation
        x = self.output(x)
        if Normal:
            x = self.norm_2(x)
        if LOGITS:
            return x
        return torch.sigmoid(x)
    

def model_training(dataset, hidden, batch, model=None):
    TYPE = dataset.get_type()
    SIZE = dataset.size

    PATH = f"models/{TYPE}/{hidden}=hidden_{batch}=batch_0.XXX=loss.pth"
    BOUND= 0.8 # Split data -> 80% train, 20% valid
    
    data   = torch.zeros((SIZE, INPUTS)) # Inputs = 44  (state + obs_arrs)
    labels = torch.zeros((SIZE, OUTPUT)) # Output = 2000 (bl_sol + bu_sol)

    for i in range(SIZE):
        # Sample solution at dataset index i

        sols = dataset.sols[i]
        state, bl_sol, bu_sol = sols[0], sols[1], sols[2]
        _, _, lower_arr, size_arr = dataset.info

        data[i, 0: 20] = torch.Tensor(lower_arr).view(-1) # 20 items
        data[i, 20:40] = torch.Tensor(size_arr ).view(-1) # 20 items

        data[i, 40:44]   = torch.Tensor(state ).view(-1)  # 4 items
        labels[i, :1000] = torch.Tensor(bl_sol).view(-1)  # 1000 items
        labels[i, 1000:] = torch.Tensor(bu_sol).view(-1)  # 1000 items

    print(f"\nDEBUG: model_training() started.\n{TYPE} = TYPE, {hidden} = hidden, {batch} = batch")

    train_size = int(BOUND * len(data))
    valid_size = len(data) - train_size

    td = TensorDataset(data, labels)
    train_data, valid_data = random_split(td, [train_size, valid_size])


    train_load = DataLoader(train_data, batch_size=batch, shuffle=True)
    valid_load = DataLoader(valid_data, batch_size=batch, shuffle=False)

    if not model:
        model = BinaryNN(hidden)

    nnBCELoss = nn.BCELoss() # Binary cross entropy loss
    logitLoss = nn.BCEWithLogitsLoss() # BCE and sigmoid

    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

    scheduler = ReduceLROnPlateau(optimizer) # update LR
    best_loss = float('inf') # Keep best validation loss

    writer = SummaryWriter(f"runs/{TYPE}")
    #writer.add_text(f"{hidden} = hidden, {batch} = batch")

    for i in range(NUM_ITERS):
        model.train()
        train_loss = 0.0

        for inputs, target in train_load:
            optimizer.zero_grad()
            output = model(inputs)

            lossFn = logitLoss if LOGITS else nnBCELoss
            loss = lossFn(output, target)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():

            for inputs, target in valid_load:
                output = model(inputs)

                if LOGITS: # needs sigmoid fn also
                    output = torch.sigmoid(output)

                loss = nnBCELoss(output, target)
                valid_loss += loss.item()

        scheduler.step(valid_loss) # update LR with valid_loss
        
        if valid_loss < best_loss: # Keep best validation loss
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
    TYPE  = dataset.get_type()
    BOUND = 0.5

    # Sample solution of relaxed_problem
    sols = dataset.sols[index]
    state, bl_sol, bu_sol = sols[0], sols[1], sols[2]
    _, _, lower_arr, size_arr = dataset.info

    start_t = torch.Tensor(state).view(-1)
    lower_t = torch.Tensor(lower_arr).view(-1)
    size_t  = torch.Tensor(size_arr ).view(-1)
    input_t = torch.cat((start_t, lower_t, size_t))

    nn_mod, nn_pth = "", ""
    nn_hid, nn_bat = 0 , 0
    bl_sol, bu_sol = [], []
    diff_min = float("inf")

    # Compare all model outputs in order
    folder = sorted(os.listdir(f"models/{TYPE}")) # same path/types only
    folder = [x for x in folder if len(x) > 10]   # except for .DS_Store

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

        output = (output >= BOUND).float() # rounds (< 0.5) to 0, (>= 0.5) to 1

        bl_out = output[:1000].view(10, 2, 50).tolist() # reshapes to multi-dim
        bu_out = output[1000:].view(10, 2, 50).tolist() # then converts to list

        diff_l = np.sum(np.array(bl_out) != np.array(sols[1])) # compares differences
        diff_u = np.sum(np.array(bu_out) != np.array(sols[2])) # in NN and data b_sol
        diff_avg = (diff_l + diff_u) / 2

        if diff_avg < diff_min:
            nn_mod, nn_pth, nn_hid, nn_bat, bl_sol, bu_sol, diff_min = (
            model , path  , hidden, batch , bl_out, bu_out, diff_avg)
            
        print(f"\nDEBUG: differences in '{TYPE}/{path}':\nbl_sol = {diff_l}, bu_sol = {diff_u}, diff_avg = {diff_avg}")
    
    print(f"\nDEBUG: best model = '{TYPE}/{nn_pth}'")

    return nn_mod, nn_hid, nn_bat, bl_sol, bu_sol


def relaxed_problem(dataset, retrain):
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
            exit()

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
        model_training(dataset, hidden=128, batch=1024)
        model_training(dataset, hidden=1024, batch=128)
        model_training(dataset, hidden=1024, batch=1024)

        dataset.normalize()
        Normal = True

        model_training(dataset, hidden=128, batch=128)
        model_training(dataset, hidden=128, batch=1024)
        model_training(dataset, hidden=1024, batch=128)
        model_training(dataset, hidden=1024, batch=1024)
    else:
        relaxed_problem(dataset, RETRAIN)