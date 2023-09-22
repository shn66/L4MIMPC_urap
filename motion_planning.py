import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
"""
Task:
- Assume the rightmost bound is x = 80, and the robot can only see x = 10 units ahead of itself.
- Given all the obstacle positions and the current state x_t, solve the motion planning problem.

Hint:
- Obstacle and robot positions are parameters. Depending on state, set params, solve from there.
- Input: current state / position; Change: Find what obstacles are in view with limited horizon.
"""
class Obstacle_Map:
    """
    lower_arr = [[-1, 2, 2,  5,  7],    # lower x coord
                 [1, -5, 3, -2, -5]])   # lower y coord
    size_arr =  [[2.5, 2.5, 2.5, 1.7, 2],   # width (x)
                 [2,   7,   2,   6.5, 8]])  # height(y)
    """
    def __init__(self, lower_arr, size_arr):
        self.lower_arr = lower_arr # <- self explanatory
        self.size_arr = size_arr   # <- self explanatory

    def unwrap(self, coord): 
        i = 0 if coord == "x" else 1 #(x, y) = arr[0, 1]
        return self.lower_arr[i], self.size_arr[i]
        # return tuple (lower, size)
        
    def insert(self, coord, items):
        i = 0 if coord == "x" else 1 #(x, y) = arr[0, 1]
        
        self.lower_arr[i].append(items[0])
        self.size_arr[i].append(items[1])
        # insert tuple (lower, size)

    def __str__(self):
        return f"lower_arr = {self.lower_arr}\nsize_arr = {self.size_arr}"
    
    def __len__(self):
        return len(self.size_arr[0])


class Robot:
    """
    Assume FOV is rectangle that spans x = 10 units ahead of robot, and spans all y coordinates.
    Prototype solution. Modify for circular FOV, x and y dependency, and line of sight blocking.
    """
    def __init__(self, state, global_obs, FOV):
        self.x = state[0]   # modified during iterations
        self.FOV = FOV      # field of view (range in x)

        # TODO: maintain state and input trajectories

        self.local_obs = Obstacle_Map([[], []], [[], []])
        self.global_obs = global_obs # also Obstacle_Map

        print(f"\nDEBUG: Robot.init() done.")
        print(f"Robot.FOV: x = {self.FOV} ahead.")
        print(f"\nglobal_obs passed in:\n{self.global_obs}")
    

    #TODO: add a function that takes  control as the argument, and updates the robot's state

    def detect_obs(self):
        lower_arr, size_arr = self.global_obs.unwrap("x") # global x arrs
        lower_y, size_y = self.global_obs.unwrap("y")     # global y arrs

        for i in range(len(self.global_obs)):# loop through each obstacle
            obs_lower = lower_arr[i]         # obs's lower corner x value
            obs_size = size_arr[i]           # obs's width in x direction

            obs_upper = obs_lower + obs_size                        # obs's upper corner x value
            in_FOV = lambda x: (x >= self.x and x <= self.FOV)      # is obs_corner in robot.FOV

            if (in_FOV(obs_lower) and in_FOV(obs_upper)):           # FOV fully capture obstacle
                self.local_obs.insert("x", (obs_lower, obs_size))   # add unchanged x: local map
                self.local_obs.insert("y", (lower_y[i], size_y[i])) # add unchanged y: local map
                continue                     # skip to next obs in global

            if (in_FOV(obs_lower) or in_FOV(obs_upper)): # FOV partially captures obs

                new_lower = min(obs_lower, self.FOV)   # new lower and upper coords
                new_upper = min(obs_upper, self.FOV)   # min(obs_corner, FOVs edge)

                new_size = new_upper - new_lower                    # new size_x = upper - lower
                self.local_obs.insert("x", (new_lower, new_size))   # add MODIFIED x-> local map
                self.local_obs.insert("y", (lower_y[i], size_y[i])) # add unchanged y: local map
        
        print(f"\nDEBUG: Detect_obs() done. Robot.local_obs:\n{self.local_obs}")


def motion_planner(robot):
    '''
    Inputs:
    obs_size:   2 x num_obs dimensional array, describing the width and height of obstacles. num_obs = # of obstacles
    obs_lower:  2 x num_obs dimensional array, describing the lower (south-western) corner of the obstacles

    Outputs:
    problem:    motion planning problem that can take starting position and goal position as parameters
    vars:       variables for the motion planning problem = [state trajectory, input trajectory, binary variables for obstacle avoidance]
    params:     parameters for the motion planning problem = [initial state, goal state]
    '''
# Obstacle "i" occupies the 2-dimensional interval: 
    # [obs_lower[:, i], obs_lower[:, i] + obs_size[:, i]]

    obs_lower = np.array(robot.local_obs.lower_arr)
    obs_size  = np.array(robot.local_obs.size_arr)

    obs_upper = obs_lower + obs_size
    num_obs   = len(robot.local_obs)

    ROBOT_MAX = robot.x + robot.FOV - 1 # Robot's max range of motion (x). Set upper bound (x) to this.

    print(f"\nDEBUG: motion_planner() started.")
    print(f"obs_lower = {obs_lower}")
    print(f"obs_size = {obs_size}")
    print(f"num_obs = {num_obs}")


#### Dynamics model data ####
    ## SEE SCREENSHOT 1 ##
    dt = 0.2
    A = np.matrix(
        [[1, 0, dt, 0],
         [0, 1, 0, dt],
         [0, 0, 1,  0],
         [0, 0, 0,  1]])
    
    B = dt * np.matrix(
        [[0, 0],
         [0, 0],
         [1, 0],
         [0, 1]])
    num_states = A.shape[1]
    num_inputs = B.shape[1]
    
#### Robot constraints ####
    ## SEE SCREENSHOT 2 ##
    Q = 1000 * np.identity(num_states)
    R = 50   * np.identity(num_inputs)
    N = 100

## State constraints
    # The robot state is subject to lower_x <= x_t <= upper_x
    lower_x = np.array([-1, -5, -1, -1])
    upper_x = np.array([ROBOT_MAX, 5, 1, 1])
        # robot must lie inside the rectangle with diagonally opposite points [-1, -5] and [10, 5]
        # robot's speed must be within -1 to 1 m/s in both X and Y directions
   
## Control constraints
    # The robot's control (accleration) is subject to lower_u <= u_t <= upper_u
    lower_u = np.array([-2, -2])
    upper_u = -lower_u

    # Declaring variables for state and input trajectories
    state = cp.Variable((num_states, N + 1)) # STATE IS X
    input = cp.Variable((num_inputs, N))     # INPUT IS U

    # Declare parameters
    state0 = cp.Parameter(num_states)
    goal   = cp.Parameter(num_states) 

#### Obstacle avoidance ####

    # Declaring binary variables for obstacle avoidance formulation
    boxes_low = [cp.Variable((2, N), boolean=True) for _ in range(num_obs)] # BOXES_LOW IS B_L
    boxes_upp = [cp.Variable((2, N), boolean=True) for _ in range(num_obs)] # BOXES_UPP IS B_U

    # big-M parameter
    M = np.diag([2 * upper_x[0], 2 * upper_x[1]])
    
    # Motion planning problem setup
    constraints = [state[:, 0] == state0]   # initial condition constraint
    objective = 0
    
    for k in range(N):
        ## SEE SCREENSHOT 1 ##
        # @ is matrix (dot) multiplication
        constraints += [state[:, k + 1] == A @ state[:, k] + B @ input[:, k]] # adding dynamics constraints
          
        constraints += [lower_x <= state[:, k + 1], upper_x >= state[:, k + 1]] # adding state constraints
    
        constraints += [lower_u <= input[:, k], upper_u >= input[:, k]] # adding control constraints

        # big-M formulation of obstacle avoidance constraints
        for i in range(num_obs):
            constraints += [state[0:2, k + 1] <= obs_lower[:, i] + M @ boxes_low[i][:, k],
                            state[0:2, k + 1] >= obs_upper[:, i] - M @ boxes_upp[i][:, k],

                            # IF YOU SATISFY ALL 4 OF A BOX'S CONSTRAINTS, YOURE IN THE BOX.
                            boxes_low[i][0, k] + boxes_low[i][1, k] + boxes_upp[i][0, k] + boxes_upp[i][1, k] <= 3]

        ## SEE SCREENSHOT 2 ##
        # calculating cumulative cost
        objective += cp.norm(Q @ (state[:, k] - goal), 'inf') + cp.norm(R @ input[:, k], 'inf') 
    
    # adding extreme penalty on terminal state to encourage getting close to the goal
    objective += 100 * cp.norm(Q @ (state[:, -1] - goal), 'inf')

    # Now, we define the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)

    print(f"\nDEBUG: motion_planner() returned.\nstate.value = {state.value}\n")
    return problem, (state, input, boxes_low, boxes_upp), (state0, goal)

## Construct the motion planning problem

obs_lower = [[-1.0, 2.0, 2.0, 5.0, 7.0],    # lower x coord
             [1.0, -5.0, 3.0,-2.0,-5.0]]    # lower y coord
obs_size  = [[2.5, 2.5, 2.5, 1.7, 2.0],     # width (x)
             [2.0, 7.0, 2.0, 6.5, 8.0]]     # height(y)

global_obs = Obstacle_Map(obs_lower, obs_size)
robot_state = [0.0, 0.0, 0.0, 0.0]

robot = Robot(robot_state, global_obs, FOV = 3.0) # Robot's max range of view (x). We only go this far.
robot.detect_obs()

num_obs = len(robot.local_obs)
problem, vars, params = motion_planner(robot)

# X    # U    # B_L     # B_U
state, input, boxes_low, boxes_upp = vars
state0, goal = params

## Instantiate with an initial and goal condition
state_list= [i+ np.array([0.1 0.1, 0. 0.]) for i in range(10)]

while not reached_goal:
    state0.value=robot.traj[-1]
    gaol.value= np.array([0.0,  0.0, 0.0 ,0.0])
    robot.detect_obs()
# state0.value = np.array([10.0, 0.0, 0.0, 0.0])
# goal.value   = np.array([0.0,  0.0, 0.0 ,0.0])

    problem.solve(verbose=False)

    print("Status: ", problem.status)
    print("Optimal cost: ", problem.value)
    print("Solve time (seconds): ", problem.solver_stats.solve_time)

    

    # Finally, collect the optimized trajectory
    x_sol = state.value
    u_sol = input.value
    bl_sol = [boxes_low[i].value for i in range(num_obs)]
    bu_sol = [boxes_upp[i].value for i in range(num_obs)]

    control=u_sol[0]

    robot.step(control)


## Plot motion planning problem with matplotlib

figure = plt.figure()
plt.gca().add_patch(Rectangle((-1, -5), 11, 10, linewidth=5.0, ec='g', fc='w', alpha=0.2, label="boundary"))

plt.plot(x_sol[0, :], x_sol[1, :], 'o', label="trajectory")
plt.plot(0.0, 0.0, '*', linewidth=10, label="goal")

for i in range(num_obs):
    if i == 0:
        plt.gca().add_patch(Rectangle((obs_lower[0][i], obs_lower[1][i]), obs_size[0][i], obs_size[1][i], ec='r', fc='r', label="obstacle"))
    else:
        plt.gca().add_patch(Rectangle((obs_lower[0][i], obs_lower[1][i]), obs_size[0][i], obs_size[1][i], ec='r', fc='r'))
plt.legend(loc = 3)
plt.show()