## Task (from last week):
We have a dataset that contains 4 items per row: `[initial_state, final_state, binary_lower, binary_upper] = solutions`.

`initial_state` and `final_state` are 4-item arrays that contain: `[x_position, y_position, x_velocity, y_velocity]`.

`binary_lower` and `binary_upper` are formatted like: [[[0,...], [0,...]], [[0,...], [0,...]],... up to 10 pairs]. The length of the top level is 10. Both items in each pair contain 50 items each (the pair has shape (2, 50)).

Our function takes an `initial_state` and a set of obstacles, and computes the boolean variables to return solution / control inputs.

We want the neural network to predict the boolean variables (`binary_lower` and `binary_upper`) given an `initial_state`.

**Question 1:** Do we use classification (and how many classes?) or regression?

**Question 2:** What type of neural network do we use to predict boolean variables?

## Solution (this week):

**Answer 1:**  We have a multi-label binary classification problem. Each boolean variable is a separate binary classification task.

We have 2 arrays of size 10, so we have 10 tasks for `binary_lower` and another 10 for `binary_upper`, giving us a total of 20 tasks.

#### TODO:
total output size = vectorised (binary_lower+binary_upper) = (N_obs_max*2*N + N_obs_max*2*N) = (10*2*50 + 10*2*50) = 2000

**Answer 2:** We should use a feed-forward neural network (FF-NN).

#### TODO:
**Input Layer:** Size will be size of `initial_state` array (+ obstacles too), which is 4.

**Hidden Layers:** We can start with 2. (# of neurons can be: input size < # < output size).

#### TODO:
**Output Layer:** Size will be total # of boolean variables, which is (update correctly to 2000).
#### TODO: (This should be a one-hot encoding for the multi-label binary classes)

**Activation Functions:** For the hidden layers, we can use `ReLU`. For the output layer, we can use `Sigmoid`. Each neuron outputs a value from 0 to 1, so we can threshold at 0.5 to determine the class (0 or 1).

## Training (and misc.):

#### TODO:
**Loss Function:** We should use the binary cross-entropy loss function. (Check if output needs to be in one-hot encoding)

**Optimizer:** We can start with the Adam optimizer and see if it works.

**Metrics:** We can start by considering accuracy, then precision or recall. We can try F1-score if a class (0 or 1) is more frequent than the other.

