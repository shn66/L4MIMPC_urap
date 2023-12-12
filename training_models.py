import os, re
import torch
import random
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fn
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split

LAYERS = 8
INPUTS = 24
HIDDEN = 512
OUTPUT = 500

ITERS = 256
BATCH = 2048
LEARN = 0.001
L = OUTPUT // 2

class Dataset:
    # solutions.pkl = [[state, local_obs, bl_sol, bu_sol] ...]

    def __init__(self, dagger=False):
        print("\nDEBUG: Dataset initializing.")
        self.sols = []

        rule = "sol_dagger" if dagger else "sol"
        data = [x for x in os.listdir("data") if x.startswith(rule)]

        for path in data:
            file = open(f"data/{path}", "rb")
            self.sols += pickle.load(file)

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

tens = lambda x: torch.Tensor(x).view(-1)

def model_training(dataset, norms, drops, weigh, activ, optiv, model=None, in_str=""):
    SIZE = dataset.size

    if not os.path.exists("models"):
        os.mkdir("models")

    activ_str = str(activ).split(" ")[1]
    PATH = f"models/{int(norms)}_{int(drops)}_{int(weigh)}{in_str}={activ_str}.pth"
    
    data   = torch.zeros((SIZE, INPUTS)) # Inputs = 24 (state, obs_arrs)
    labels = torch.zeros((SIZE, OUTPUT)) # Output = 500 (bl_sol, bu_sol)

    for i in range(SIZE):                # Sample sol @ index i
        state, obs_arr, bl_sol, bu_sol = dataset.sols[i]

        data  [i, :4] = tens(state)      # 4 items
        data  [i, 4:] = tens(obs_arr)    # 20 items
        labels[i, :L] = tens(bl_sol)     # 250 items
        labels[i, L:] = tens(bu_sol)     # 250 items

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

    nums = [int(x) for x in re.findall(r'\d+', f"models/{path}")]
    norms, drops = bool(nums[0]), bool(nums[1])       # Term 1,2 (int->bool)

    funct = path.split("=")[-1][:-4]  # Final value (string)
    activ = eval(f"fn.{funct}")       # TODO: fix hardcoding

    if len(nums) > 3:
       global LAYERS, HIDDEN, BATCH
       LAYERS, HIDDEN, BATCH = nums[3], nums[4], nums[5]
    
    if not model:
        model = BinaryNN(norms, drops, activ)
        load  = torch.load(f"models/{path}")
        model.load_state_dict(load)
        model.eval()

    bl_sol, bu_sol = [], []
    if not state or not obs_arr: # Random sol @ index i
        
        i = random.randint(0, dataset.size - 1)
        state, obs_arr, bl_sol, bu_sol = dataset.sols[i]

    input = torch.cat((tens(state), tens(obs_arr)))
    
    if norms:
        input = input.unsqueeze(0)   # Add batch dim->input
    
    with torch.no_grad():
        output = torch.sigmoid(model(input))
    
    output = (output.view(-1) >= 0.5).float()          # Remove batch dim, rounds to 0 or 1

    nums = lambda x: x.view(5, 2, 25).detach().numpy() # Shape lst to multi-dim -> np.array
    bl_out, bu_out = nums(output[:L]), nums(output[L:])
    
    return bl_out, bu_out, obs_arr, np.array(bl_sol), np.array(bu_sol)


def test_model_diff(dataset, verbose):
    path_min, diff_min = "", float("inf")

    folder = sorted(os.listdir("models"))       # Same path/types only
    folder = [x for x in folder if len(x) > 10] # Except for .DS_Store

    for path in folder:
        diff_l, diff_u = 0.0, 0.0

        for _ in range(ITERS):
            bl_out, bu_out, _, bl_sol, bu_sol = get_model_outs(dataset, path)

            diff_l += np.sum(bl_sol != bl_out) # Compares differences
            diff_u += np.sum(bu_sol != bu_out) # btwn output and data

        diff_avg = (diff_l + diff_u) / (2 * ITERS)

        if diff_avg < diff_min:
            path_min, diff_min = path, diff_avg
        
        print(f"\nDEBUG: differences in {path}:\nbl_sol = {diff_l / ITERS}")
        print(f"bu_sol = {diff_u / ITERS}\ndiff_avg = {diff_avg}")

    if verbose:
        bl_out, bu_out, obs_arr, bl_sol, bu_sol = get_model_outs(dataset, path_min)

        diff_l = np.where(bl_sol != bl_out, "X", ".") # Compares differences
        diff_u = np.where(bu_sol != bu_out, "X", ".") # X is wrong . is good

        lower_x, lower_y = obs_arr[0][0], obs_arr[0][1]
        size_x , size_y  = obs_arr[1][0], obs_arr[1][1]

        for i in range(len(obs_arr[0])):
            print(f"\nobs at ({lower_x[i]}, {lower_y[i]}), size ({size_x[i]}, {size_y[i]}):")
            print(f"\ndiff_l =\n{diff_l[i]}\ndiff_u =\n{diff_u[i]}")
    
    print(f"\nDEBUG: best model = {path_min}")


if __name__ == "__main__":
    dataset = Dataset()
    TRAIN = False

    if TRAIN:
        model_training(dataset, True, False, True, fn.leaky_relu, optim.Adam)
    else:
        test_model_diff(dataset, verbose=True)