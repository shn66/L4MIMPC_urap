import torch
import numpy as np
import torch.nn as nn
import torchvision as tv
import matplotlib.pyplot as plt
import torch.nn.functional as fn
import torchvision.transforms as hrt
from torchvision import datasets, models
from torch.optim import lr_scheduler as lr
from torch.utils.tensorboard import SummaryWriter as sw

PATH = "/path/on/local/device"

def softmax(vals):
    return np.exp(vals) / np.sum(np.exp(vals), axis = 0)

def cross_en(real, pred):
   loss = -np.sum(real * np.log(pred))
   return loss # / float(pred.shape[0])


def softmax_cross_en():

    x = np.array([2.0, 1.0, 0.1])
    print("softmax numpy: ", softmax(x))

    x = torch.tensor([2.0, 1.0, 0.1])
    ans = torch.softmax(x, dim = 0)
    print("softmax pytorch: ", ans)

    y = np.array([1, 0, 0]) # one-hot encoded
    y_pred = np.array([0.7, 0.2, 0.1])# probs

    loss_np = cross_en(y, y_pred)
    print(f"loss numpy: {loss_np:.3f}")

    loss_py = nn.CrossEntropyLoss()
    y = torch.tensor([0]) # NOT 1HE, use[0]

    # size = samples * classes = 1x3
    y_pred = torch.tensor([[2.0, 1.0, 0.1]])

    ans = loss_py(y_pred, y) # no softmax here
    print(f"loss pytorch: {ans.item():.3f}")

    _, pred = torch.max(y_pred, 1) # return[0]
    print("prediction pytorch: ", pred)

    binary_net = BinaryNet(input=28*28, hidden=5)
    binary_loss = nn.BCELoss()

    multi_net = MultiNet(input=28*28, hidden=5, classes=3)
    multi_loss = nn.CrossEntropyLoss() # applies softmax


class BinaryNet(nn.Module):

    def __init__(self, input, hidden):
        super(BinaryNet, self).__init__()
        self.lin1 = nn.Linear(input, hidden) 
        self.relu = nn.ReLU() # activation func
        self.lin2 = nn.Linear(hidden, 1)  
    
    def forward(self, x):
        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)
        return torch.sigmoid(out)

class MultiNet(nn.Module):

    def __init__(self, input, hidden, classes):
        super(MultiNet, self).__init__()
        self.lin1 = nn.Linear(input, hidden) 
        self.relu = nn.ReLU() # activation func
        self.lin2 = nn.Linear(hidden, classes)
    
    def forward(self, x):
        out = self.lin1(x)
        out = self.relu(out)
        return self.lin2(out)

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.full1 = nn.Linear(16 * 5 * 5, 120)
        self.full2 = nn.Linear(120, 84)
        self.full3 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(fn.relu(self.conv1(x))) # -> n, 6, 14, 14
        x = self.pool(fn.relu(self.conv2(x))) # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5) # -> n, 400
        x = fn.relu(self.full1(x)) # -> n, 120
        x = fn.relu(self.full2(x)) # -> n, 84
        return self.full3(x)       # -> n, 10


def feed_forward_nn(BOARD = False):
    """
    1. MNIST Dataset
    2. DataLoader, Transforms
    3. MultiNet, Active Funcs
    4. Loss and Optimizer
    5. Training Loop (Batch)
    6. Model Evaluation
    """
    writer = sw(PATH)

    # Hyper Variables
    INPUT = 784 # 28 x 28 picture
    HIDDEN = 100
    CLASSES = 10
    ITER = 2
    BATCH = 100
    RATE = 0.01

    # 1. MNIST Dataset
    train_data = tv.datasets.MNIST(root=PATH, train=True, transform=hrt.ToTensor(), download=True)
    test_data = tv.datasets.MNIST(root=PATH, train=False, transform=hrt.ToTensor())

    # 2. DataLoader, Transforms
    train_load = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH, shuffle=True)
    test_load = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH, shuffle=False)

    # 2.1 Prints Data
    samples, labels = next(iter(train_load))
    print(f"samples: {samples.shape}, labels: {labels.shape}\n")

    for i in range(6): # TENSORBRD IMAGE
        plt.subplot(2, 3, i + 1)
        plt.imshow(samples[i][0], cmap = "gray")

    if BOARD:
        grid = tv.utils.make_grid(samples)
        writer.add_image("mnist images", grid)
        writer.close()
    # plt.show()
    
    model = MultiNet(INPUT, HIDDEN, CLASSES) # 3. MultiNet, Active Funcs
    loss_fn = nn.CrossEntropyLoss()          # 4. Loss and Optimizer
    optim = torch.optim.Adam(model.parameters(), lr = RATE)

    if BOARD: # TENSORBRD GRAPH
        writer.add_graph(model, samples.reshape(-1, 28 * 28))
        writer.close()
    run_loss = 0.0
    run_right = 0.0

    # 5. Training Loop (Batch)
    steps = len(train_load)
    for i in range(ITER):
        for n, (images, labels) in enumerate(train_load):

            images = images.reshape(-1, 28 * 28) # 28 x 28 picture
            output = model(images) # forward pass
            loss = loss_fn(output, labels)

            optim.zero_grad() # backward pass
            loss.backward()
            optim.step()

            run_loss += loss.item()
            _, pred = torch.max(output.data, 1) # -> value, index
            run_right += (pred == labels).sum().item()

            if (n + 1) % 100 == 0:
                print(f"iter {i+1} / {ITER}, step {n+1} / {steps}, loss = {loss.item():.3f}")

                if BOARD: # TENSORBRD GRAPH
                    writer.add_scalar("training loss", run_loss / 100, i * steps + n)
                    writer.add_scalar("accuracy", run_right / 100, i * steps + n)
                    run_loss = 0.0
                    run_right = 0.0
    
    with torch.no_grad():
        x_axis = []
        y_axis = []
        right = 0
        total = 0

        for images, labels in test_load:
            images = images.reshape(-1, 28 * 28) # 28 x 28 picture
            output = model(images)

            _, pred = torch.max(output, 1) # -> value, index
            total += labels.shape[0]
            right += (pred == labels).sum().item()

            x_axis.append(pred)
            y_axis.append([fn.softmax(i,dim=0) for i in output])

        print(f"accuracy = {100.0 * right / total}")

        if BOARD: # TBOARD_PR_CURVE
            x_axis = torch.cat(x_axis)
            y_axis = torch.cat([torch.stack(i) for i in y_axis])

            for i in range(10):
                x = (x_axis == i)
                y = y_axis[:, i]
                writer.add_pr_curve(str(i), x, y, global_step=0)
                writer.close()
    return model


def convolution_nn():
    ITER = 4
    BATCH = 4
    RATE = 0.01

    # dataset has PILImage images of range [0, 1]
    # transform to tensors of norm. range [-1, 1]
    trans = hrt.Compose([hrt.ToTensor(), hrt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = tv.datasets.CIFAR10(root=PATH, train=True, download=True, transform=trans)
    test_data = tv.datasets.CIFAR10(root=PATH, train=False, download=True, transform=trans)

    train_load = torch.utils.data.DataLoader(train_data, batch_size=BATCH, shuffle=True)
    test_load = torch.utils.data.DataLoader(train_data, batch_size=BATCH, shuffle=False)

    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    def show_img(image):
        npimg = (image / 2 + 0.5).numpy() # unnormalize
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        # plt.show()
    
    images, labels = next(iter(train_load)) # show images
    show_img(tv.utils.make_grid(images))

    model = ConvNet()
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr = RATE)

    for i in range(ITER):
        for n, (images, labels) in enumerate(train_load):
            # origin shape: [4, 3, 32, 32] = 4, 3, 1024
            # input layer: 3 input, 6 output, 5 kernel

            output = model(images)
            loss = loss_fn(output, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if (n + 1) % 2000 == 0:
                print(f"iter {i+1} / {ITER}, step {n+1} / {len(train_load)}, loss = {loss.item():.3f}")

    with torch.no_grad():
        right = 0
        total = 0
        c_right = [0 for i in range(10)]
        c_total = [0 for i in range(10)]

        for images, labels in test_load:
            output = model(images)
            _, pred = torch.max(output, 1) # -> value, index

            total += labels.size(0)
            right += (pred == labels).sum().item()
            
            for i in range(BATCH):
                label = labels[i]
                guess = pred[i]

                if (label == guess):
                    c_right[label] += 1
                c_total[label] += 1

        print(f"Accuracy = {100.0 * right / total}")

        for i in range(10):
            ans = 100.0 * c_right[i] / c_total[i]
            print(f"Accuracy of {classes[i]} = {ans}")


def transferlearn(train_model):

    # https://github.com/patrickloeber/pytorchTutorial/blob/master/15_transfer_learning.py
    FREEZE = True # freeze all layers, only retrain last

    model = models.resnet18(pretrained = True)
    if FREEZE:
        for param in model.parameters():
            param.requires_grad = False

    n_feat = model.fc.in_features
    model.fc = nn.Linear(n_feat, 2)
    loss_fn = nn.CrossEntropyLoss()
    optim = optim.SGD(model.parameters(), lr = 0.01)

    # every STEP iters, lr *= GAMMA
    step_lr = lr.StepLR(optim, step_size=7, gamma=0.1)
    model = train_model(model, loss_fn, optim, step_lr, num_iter = 20)

def tensorboard_n():
    # cd Desktop/python/data
    # tensorboard --logdir=runs
    feed_forward_nn(BOARD = True)


def save_n_load(SAVING = False):
    model = MultiNet(784, 100, 10)
    if 0:
        # Lazy way: full model
        torch.save(model, PATH)
        model = torch.load(PATH)
        model.eval()

        # Good way: state dict
        torch.save(model.state_dict(), PATH)
        model.load_state_dict(torch.load(PATH))
        model.eval()

    if SAVING:
        model = feed_forward_nn()
        torch.save(model.state_dict(), PATH)
    else:
        model.load_state_dict(torch.load(PATH))
        model.eval()
    for param in model.parameters():
        print(param)
 

def main():
    funcs = [
        "softmax_cross_en()",
        "feed_forward_nn()",
        "convolution_nn()",
        "tensorboard_n()",
        "save_n_load()"
        ]
    print("\nChoose a function.\n")

    # print list, get input, run func
    for i in range(len(funcs)):
        print(f"{i + 1}: {funcs[i]}")
    eval(funcs[int(input()) - 1])
main()