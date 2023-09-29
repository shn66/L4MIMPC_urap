import random
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

"""
For simulation (next week):
- Define step function (input: control object) -> (return: new state / trajectory)
  - Adds every calculated state to array of states in robot object

- Define environment class with global obstacles, walls of room
  - Make a memory array containing solutions to 100+ simulations
    - Fixed obstacles and goal position, but randomized starting states

- Implement changes to motion_planning from sid's file
  - new state, (unchanging) goal, and newly detected obstacles are CVXPY parameters
  - these parameter.value get changed every timestep before problem.solve() is called

>>> Also, in Robot.detect_obs() see if after multiple timesteps / detection cycles
  - Is there a risk of obstacles being added to the list of local_obs multiple times?
  - What about slices / parts of obstacles? Do we need to check for overlaps?
"""

class Obstacle_Map:
    """
    lower_arr = [[-1.0, 2.0, 2.0, 5.0, 7.0],   # x coords
                 [1.0, -5.0, 3.0,-2.0,-5.0]]   # y coords

    size_arr  = [[2.5, 2.5, 2.5, 1.7, 2.0],    # width: x
                 [2.0, 7.0, 2.0, 6.5, 8.0]]    # height:y
    """
    def __init__(self, lower_arr, size_arr):
        self.lower_arr = lower_arr
        self.size_arr = size_arr

    def unwrap(self, coord): # return tuple (lower, size)
        i = 0 if coord == "x" else 1 # (x, y) = arr[0, 1]
        return self.lower_arr[i], self.size_arr[i]
    

    def insert(self, coord, items):  # insert tuple (lower, size)
        i = 0 if coord == "x" else 1 # (x, y) = arr[0, 1]
        lower, size = items
        found = False

        for j in range(len(self)): # loop through lower, size arr
            if self.lower_arr[i][j] == lower and self.size_arr[i][j] == size:
                found = True # if lower, size pair already exists
        if not found:
            self.lower_arr[i].append(lower) # if pair not in arr:
            self.size_arr[i].append(size)   # append(lower, size)
        else:
            print("\nDEBUG: insert() failed, obs found in array")
        
    def __str__(self):
        return f"lower_arr = {self.lower_arr}\nsize_arr = {self.size_arr}"
    
    def __len__(self):
        return len(self.size_arr[0])


class Environment:
    """
    limit = [[-1.0,-5.0,-1.0,-1.0], # lower[pos_x, pos_y,
             [10.0, 5.0, 1.0, 1.0]] # upper vel_x, vel_y]

    goal = [x_pos, y_pos, x_vel, y_vel]
    global_obs = Obstacle_Map()
    """
    def __init__(self, limit, goal, global_obs):
        self.limit = limit
        self.goal = goal
        self.global_obs = global_obs
        self.solutions = [] # [soln=[state0, state_traj, input_traj], ...]


    def random_state(self):
        lower, upper = self.limit[0], self.limit[1] # 0 = lower, 1 = upper

        for _ in range(100):                       # 0 = x vals, 1 = y val
            x = random.uniform(lower[0], upper[0]) # random float #: lower
            y = random.uniform(lower[1], upper[1]) # to upper xy inclusive

            l_obs_x, s_obs_x = self.global_obs.unwrap("x") # global x arrs
            l_obs_y, s_obs_y = self.global_obs.unwrap("y") # global y arrs

            u_obs_x = l_obs_x + s_obs_x # upper_obs_x_y =
            u_obs_y = l_obs_y + s_obs_y # lower + size_xy
            
            for i in range(len(self.global_obs)): # loop through every obs
                if (x >= l_obs_x[i] and x <= u_obs_x[i] and
                    y >= l_obs_y[i] and y <= u_obs_y[i]):
                    break                         # if in obs: get new x,y
                else: return [round(x, 2), round(y, 2), 0.0, 0.0]
                                                  # else: return this x, y
        print("\nERROR: random_state() couldn't find valid state"); exit()


    def plot_problem(self, x_sol):
        # Graph the motion planning problem
        # %matplotlib inline

        figure = plt.figure()
        plt.gca().add_patch(Rectangle((-1, -5), 11, 10, linewidth=5.0, ec='g', fc='w', alpha=0.2, label="boundary"))

        plt.plot(x_sol[0, :], x_sol[1, :], 'o', label="trajectory")
        plt.plot(1.0, 0.0, '*', linewidth=10, label="goal")

        obs_lower = self.global_obs.lower_arr
        obs_size = self.global_obs.size_arr

        for i in range(len(self.global_obs)):
            if i == 0:
                plt.gca().add_patch(Rectangle((obs_lower[0][i], obs_lower[1][i]), obs_size[0][i], obs_size[1][i], ec='r', fc='r', label="obstacle"))
            else:
                plt.gca().add_patch(Rectangle((obs_lower[0][i], obs_lower[1][i]), obs_size[0][i], obs_size[1][i], ec='r', fc='r'))
        plt.legend(loc = 3)
        plt.show()


class Robot:
    """
    state = [pos_x, pos_y, vel_x, vel_y]
    time = seconds between state updates
    FOV = Field Of View: range in x dir.

    global/local_obs = Obstacle_Map()
    state/input_traj = [[x1], [x2], ...]
    """
    def __init__(self, state, global_obs, time, FOV):
        self.state = state
        self.time = time
        self.FOV = FOV

        self.local_obs = Obstacle_Map([[], []], [[], []])
        self.global_obs = global_obs # Obstacle_Map again

        self.state_traj = [self.state] # track all states
        self.input_traj = [] # & inputs by updating array


    def update_state(self, input):
        t = self.time
        acc_x, acc_y = tuple(input)  # unwrap list->tuple
        pos_x, pos_y, vel_x, vel_y = tuple(self.state)

        pos_x += vel_x * t + (0.5 * acc_x * t ** 2)
        pos_y += vel_y * t + (0.5 * acc_y * t ** 2)
        vel_x += acc_x * t # x = x0 + v * t + 0.5(a* t^2)
        vel_y += acc_y * t # v = v0 + a * t
        
        self.state = [pos_x, pos_y, vel_x, vel_y] # create new state list
        self.state_traj.append(list(self.state))  # append a copy of list
        self.input_traj.append(list(input))       # to avoid mutation bug

        print(f"\nDEBUG: update_state() done. Robot.state:\n{self.state}")


    def detect_obs(self):
        lower_arr, size_arr = self.global_obs.unwrap("x") # global x arrs
        lower_y, size_y = self.global_obs.unwrap("y")     # global y arrs

        for i in range(len(self.global_obs)):# loop through each obstacle
            obs_lower = lower_arr[i]         # obs's lower corner x value
            obs_size = size_arr[i]           # obs's width in x direction

            obs_upper = obs_lower + obs_size                        # obs's upper corner x value
            in_FOV = lambda x: x >= self.state[0] and x <= self.FOV # is obs_corner in robot.FOV

            if (in_FOV(obs_lower) and in_FOV(obs_upper)):           # FOV fully capture obstacle
                self.local_obs.insert("x", (obs_lower, obs_size))   # add unchanged x: local map
                self.local_obs.insert("y", (lower_y[i], size_y[i])) # add unchanged y: local map
                continue                               # skip to next obs in global

            if (in_FOV(obs_lower) or in_FOV(obs_upper)): # FOV captures partial obs

                new_lower = min(obs_lower, self.FOV)   # new lower and upper coords
                new_upper = min(obs_upper, self.FOV)   # min(obs_corner, FOVs edge)

                new_size = new_upper - new_lower                    # new size_x = upper - lower
                self.local_obs.insert("x", (new_lower, new_size))   # add MODIFIED x-> local map
                self.local_obs.insert("y", (lower_y[i], size_y[i])) # add unchanged y: local map
        
        print(f"\nDEBUG: detect_obs() done. Robot.local_obs:\n{self.local_obs}")


def motion_planning(world):
    """
    Inputs:
    obs_size:   2 x num_obs array, describing width and height of the obstacles, num_obs = # of obstacles
    obs_lower:  2 x num_obs array, describing lower (south-western) corners of obstacles

    Outputs:
    problem:    motion planning problem that can take a starting position and goal position as parameters
    vars:       variables = [state trajectory, input trajectory, binary variables for obstacle avoidance]
    params:     parameters for the motion planning problem = [initial state, goal state]
    """
#### Dynamics model data ####
    ## SEE SCREENSHOT 1 ##
    dt = 0.2
    A = np.matrix(
        [[1, 0, dt,0],
         [0, 1, 0,dt],
         [0, 0, 1, 0],
         [0, 0, 0, 1]])
    
    B = dt * np.matrix(
        [[0, 0],
         [0, 0],
         [1, 0],
         [0, 1]])
    dim_state = A.shape[1]
    dim_input = B.shape[1]
    

#### Robot constraints ####
    ## SEE SCREENSHOT 2 ##
    Q = 1000 * np.identity(dim_state)
    R = 50   * np.identity(dim_input)
    N = 100

## State constraints
    lower_x = np.array(world.limit[0]) # [[-1.0,-5.0,-1.0,-1.0], [pos_x, pos_y,
    upper_x = np.array(world.limit[1]) #  [10.0, 5.0, 1.0, 1.0]]  vel_x, vel_y]
   
## Input constraints
    lower_u = np.array([-2, -2]) # input / acceleration u_t must be
    upper_u = -1 * lower_u       # within lower_u <= u_t <= upper_u

    # Declare variables for state and input trajectories
    state = cp.Variable((dim_state, N + 1)) # STATE IS X
    input = cp.Variable((dim_input, N))     # INPUT IS U

    # Declare parameters for state0, goal, and obstacles
    state0 = cp.Parameter(dim_state) # state0, goal need
    goal   = cp.Parameter(dim_state) # of dimensions = 4
    
    """
    TODO: Fix 2 issues with cp.Parameter().

        cp.Parameter(n) must be initialized with a size/dimensions n,
        but we don't know # of columns/obs in robot_obs ahead of time

        We can't use robot_obs.value because cp.Param values are null
        when initialized and when we reach line 254 - 274 of function

        TODO: Try initializing obs_lower,obs_size = cp.Parameter(MAX)
        where MAX >> # of obs. Then ignore values after last obstacle
    """
    robot_obs = cp.Parameter()  # Obstacle_Map() dim = ?
    robot_FOV = cp.Parameter(2) # [low_x, upp_x] dim = 2

    obs_lower = np.array(robot_obs.value.lower_arr)
    obs_size = np.array(robot_obs.value.size_arr)

    obs_upper = obs_lower + obs_size # All = 2D np.array
    num_obs   = obs_size.shape[1] # of columns = num_obs


#### Obstacle avoidance ####

    # Declaring binary variables for obstacle avoidance formulation
    boxes_low = [cp.Variable((2, N), boolean=True) for _ in range(num_obs)] # BOXES_LOW IS B_L
    boxes_upp = [cp.Variable((2, N), boolean=True) for _ in range(num_obs)] # BOXES_UPP IS B_U

    M = np.diag([2 * upper_x[0], 2 * upper_x[1]]) # big-M parameter
    
    # Setup the motion planning problem
    """
    TODO: See if robot_FOV has cp.Parameter() issues.
    """
    lower_x[0] = max(lower_x[0], robot_FOV[0]) # robot_FOV = [lower_x, upper_x],
    upper_x[0] = min(upper_x[0], robot_FOV[1]) # max/min of itself and robot_FOV

    constraints = [state[:,0] == state0] # initial state constraint
    objective = 0
    
    for k in range(N):
        ## SEE SCREENSHOT 1 ##
        # @ is matrix (dot) multiplication
        constraints += [state[:, k + 1] == A @ state[:, k] + B @ input[:, k]]   # add dynamics constraints
          
        constraints += [lower_x <= state[:, k + 1], upper_x >= state[:, k + 1]] # adding state constraints
    
        constraints += [lower_u <= input[:, k], upper_u >= input[:, k]]         # adding input constraints


        # big-M formulation of obstacle avoidance constraints
        for i in range(num_obs):
            constraints += [
                state[0:2, k + 1] <= obs_lower[:, i] + M @ boxes_low[i][:, k],
                state[0:2, k + 1] >= obs_upper[:, i] - M @ boxes_upp[i][:, k],

                # IF YOU SATISFY ALL 4 OF BOX'S CONSTRAINTS, YOURE IN THE BOX.
                boxes_low[i][0, k] + boxes_low[i][1, k] + boxes_upp[i][0, k] + boxes_upp[i][1, k] <= 3]

        ## SEE SCREENSHOT 2 ##
        # calculating cumulative cost
        objective += cp.norm(Q @ (state[:, k] - goal), 'inf') + cp.norm(R @ input[:, k], 'inf') 
    
    # adding extreme penalty on terminal state to encourage getting close to the goal
    objective += 100 * cp.norm(Q @ (state[:, -1] - goal), 'inf')

    # Define the motion planning problem
    problem = cp.Problem(cp.Minimize(objective), constraints)

    print(f"\nDEBUG: motion_planning() done.\nstate.value = {state.value}")
    return problem, (state, input, boxes_low, boxes_upp), (state0, goal, robot_obs, robot_FOV)


def run_simulations(num_iters):
    # Create the motion planning problem

    lower_arr = [[-1.0, 2.0, 2.0, 5.0, 7.0],   # x coords
                 [1.0, -5.0, 3.0,-2.0,-5.0]]   # y coords
    
    size_arr  = [[2.5, 2.5, 2.5, 1.7, 2.0],    # width: x
                 [2.0, 7.0, 2.0, 6.5, 8.0]]    # height:y
    
    limit = [[-1.0,-5.0,-1.0,-1.0], # lower[pos_x, pos_y,
             [10.0, 5.0, 1.0, 1.0]] # upper vel_x, vel_y]
    
    goal0 = [10.0, 0.0, 0.0, 0.0]
    global_obs = Obstacle_Map(lower_arr, size_arr)
    world = Environment(limit, goal0, global_obs)

    # Randomize start, get vars & params
    for _ in range(num_iters):

        start = world.random_state()
        robot = Robot(start, global_obs, time=1.0, FOV=5.0)

        problem, vars, params = motion_planning(world)

        state, input, boxes_low, boxes_upp = vars
        state0, goal, robot_obs, robot_FOV = params

        # Initialize all CP parameter values
        while (robot.state != goal):

            state0.value = np.array(robot.state)
            goal.value = np.array(world.goal)

            robot.detect_obs()
            num_obs = len(robot.local_obs)

            robot_obs.value = np.array(robot.local_obs)
            robot_FOV.value = np.array([robot.state, robot.state + robot.FOV])

            # Done: collect optimized trajectory
            problem.solve(verbose=False)

            print("Status: ", problem.status)
            print("Optimal cost: ", problem.value)
            print("Solve time (seconds): ", problem.solver_stats.solve_time)

            x_sol = state.value
            u_sol = input.value
            bl_sol = [boxes_low[i].value for i in range(num_obs.value)]
            bu_sol = [boxes_upp[i].value for i in range(num_obs.value)]

            # Collect solutions in robot & world
            robot.update_state(u_sol)

            # robot.state = x_sol
            # robot.state_traj.append(list(x_sol))
            # robot.input_traj.append(list(u_sol))
            
            world.plot_problem(x_sol)

        world.solutions.append([start, goal0, robot.state_traj, robot.input_traj])

run_simulations(num_iters = 1)