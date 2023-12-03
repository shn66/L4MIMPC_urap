import os
import copy
import random
import pickle
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class ObsMap:
    """
    lower_arr = [[ 0.5, 1.7, 2.7, 2.7, 3.8], # x coords
                 [-0.3,-0.7,-1.3, 0.3,-0.5]] # y coords
    
    size_arr  = [[0.7, 0.5, 0.5, 0.5, 0.7],  # width: x
                 [1.0, 0.7, 1.0, 1.0, 1.0]]  # height:y
    """
    def __init__(self, lower_arr, size_arr):
        self.lower_arr = lower_arr
        self.size_arr  = size_arr

        case = 0 # checks both arrays have x, y
        if not (len(lower_arr) == len(size_arr) == 2):
            case = 1
        else:    # check all len(arr) are equal
            for arr in lower_arr + size_arr:
                if len(arr) != len(lower_arr[0]):
                    case = 2
                    break       
                if not all(isinstance(i, float) for i in arr):
                    case = 3    # checks all values are floats
                    break 
        if case != 0:
            print(f"\nERROR: Obs_Map.init() failed @ case {case}"); exit()


    def insert(self, items_x, items_y):
        lower_x, size_x = items_x # items_x = (lower_x, size_x)
        lower_y, size_y = items_y

        found = False # check all 4 vals at index i
        for i in range(len(self.lower_arr[0])):

            if (lower_x == self.lower_arr[0][i] and
                lower_y == self.lower_arr[1][i] and
                size_x  == self.size_arr [0][i] and
                size_y  == self.size_arr [1][i]):
                found = True

        if not found: # append all 4 vals to arrays
            self.lower_arr[0].append(lower_x)
            self.lower_arr[1].append(lower_y)
            self.size_arr [0].append(size_x)
            self.size_arr [1].append(size_y)

    
    def unwrap(self): # return (x,y)->(lower, size)
        lower = self.lower_arr
        size  = self.size_arr
        return lower[0], size[0], lower[1], size[1]
    
    def __str__(self):
        return f"lower_arr = {self.lower_arr}\nsize_arr = {self.size_arr}"
    
    def __len__(self):
        return len(self.size_arr[0])


class World:
    """
    limit = [[0.0,-1.2,-1.0,-1.0], # lower[pos_x, pos_y,
             [5.0, 1.2, 1.0, 1.0]] # upper vel_x, vel_y]

    goal = [x_pos, y_pos, x_vel, y_vel]
    solutions = [[state, bl_sol, bu_sol], ...]

    world_obs = Obstacle_Map()
    MAX = integer max # of obs
    """
    def __init__(self, limit, goal, world_obs, TOL):
        self.limit = limit
        self.goal  = goal
        self.TOL   = TOL

        self.solutions = []
        self.world_obs = world_obs
        self.MAX = len(world_obs)


    def random_state(self, iters, bound):
        lower, upper = self.limit # unpack array -> 2 vars

        for _ in range(iters):                             # random starting state:
            x = random.uniform(lower[0], upper[0] * bound) # within x_width * bound
            y = random.uniform(lower[1] + self.TOL, upper[1] - self.TOL)

            lower_x, size_x, lower_y, size_y = self.world_obs.unwrap()

            upper_x = np.array(lower_x) + np.array(size_x) # upper_x, upper_y =
            upper_y = np.array(lower_y) + np.array(size_y) # lower_xy + size_xy
            
            in_obs = False
            for i in range(len(self.world_obs)):  # loop through every obs
                if (x >= lower_x[i] and x <= upper_x[i] and
                    y >= lower_y[i] and y <= upper_y[i]):
                    
                    in_obs = True                 # if inside walls of obs
                    break                         # break to get new state
            if not in_obs:                        # else return this state
                return [x, y, 0.0, 0.0]

        print("\nERROR: random_state() couldn't find valid state"); exit()


    def plot_problem(self, state_sol, start, goal):
        lower_arr = self.world_obs.lower_arr
        size_arr  = self.world_obs.size_arr

        plt.gca().add_patch(Rectangle((0, -1.25), 5, 2.5, linewidth=1.0, 
            ec="g", fc="w", alpha=0.2, label="boundary"))
    
        plt.plot(state_sol[0, :], state_sol[1, :], "o", label="path")
        plt.plot(start[0], start[1], "*", linewidth=10, label="start")
        plt.plot(goal[0],  goal[1],  "*", linewidth=10, label="goal")

        for i in range(len(self.world_obs)):
            label = "obstacle" if i == 0 else ""

            plt.gca().add_patch(Rectangle((lower_arr[0][i], lower_arr[1][i]),
                size_arr[0][i], size_arr[1][i], ec="r", fc="r", label=label))
        plt.legend(loc = 4)
        plt.show()


class Robot:
    """
    state = [pos_x, pos_y, vel_x, vel_y]
    global/local_obs = Obstacle_Map()

    TIME = seconds between state updates
    FOV = Field Of View: range in x dir.
    """
    def __init__(self, state, world_obs, TIME, FOV):
        self.state = state
        self.TIME  = TIME
        self.FOV   = FOV

        self.world_obs = world_obs
        self.local_obs  = ObsMap([[], []], [[], []])

        self.state_traj = [[], [], [], []]    # track vars by updating arr
        self.input_traj = [[], []]


    def detect_obs(self):
        x_pos = self.state[0]                 # state = [x_pos, y_pos,...]
        lower_x, size_x, lower_y, size_y = self.world_obs.unwrap()

        for i in range(len(self.world_obs)):  # loop through each obstacle
            lower = lower_x[i]                # obs's lower corner x value
            size  = size_x[i]                 # obs's width in x direction
            
            upper  = lower + size             # is obs's corner within FOV
            in_FOV = lambda obs: obs >= x_pos and obs <= x_pos + self.FOV

            if (in_FOV(lower) or in_FOV(upper)):   # FOV partly sees obstacle
                self.local_obs.insert((lower, size), (lower_y[i], size_y[i]))
                # add unchanged vals ((  x_items  ), (       y_items       ))
    

    def update_state(self, acc_x, acc_y):
        for i in range(4):               # write curr/start state
            self.state_traj[i].append(self.state[i])

        self.input_traj[0].append(acc_x) # write given input vals
        self.input_traj[1].append(acc_y) # arr[0] = x, arr[1] = y

        t = self.TIME
        pos_x, pos_y, vel_x, vel_y = self.state   # unpack state -> 4 vars

        pos_x += vel_x * t + (0.5 * acc_x * t ** 2)
        pos_y += vel_y * t + (0.5 * acc_y * t ** 2)
        vel_x += acc_x * t # x = x0 + v * t + 0.5(a * t^2)
        vel_y += acc_y * t # v = v0 + a * t
        
        self.state = [pos_x, pos_y, vel_x, vel_y] # assign new state array
        
        print(f"\nDEBUG: update_state() done = {[round(x, 2) for x in self.state]}")


def motion_planning(world, robot, relaxed, N=25):
    """
    Inputs:
    obs_size:   2 x num_obs array, describing width and height of the obstacles, num_obs = # of obstacles
    lower_obs:  2 x num_obs array, describing lower (south-western) corners of obstacles

    Outputs:
    problem:    motion planning problem that can take a starting position and goal position as parameters
    vars:       variables = [state trajectory, input trajectory, binary variables for obstacle avoidance]
    params:     parameters for the motion planning problem = [initial state, goal state]
    """
#### Dynamics model data ####
    dt = robot.TIME

    A = np.matrix(
       [[1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1]])
    
    B = dt * np.matrix(
       [[0.5*dt, 0     ],
        [0,      0.5*dt],
        [1,      0     ],
        [0,      1     ]])
    
    dim_state = A.shape[1]
    dim_input = B.shape[1]
    

#### Robot constraints ####
    Q = 100 * np.identity(dim_state)
    R = 50  * np.identity(dim_input)
    
## Vars & Parameters
    # Declare variables for state and input trajectories
    state = cp.Variable((dim_state, N + 1)) # STATE IS X
    input = cp.Variable((dim_input, N))     # INPUT IS U

    # Declare parameters for state0, goal, and obstacles
    state0 = cp.Parameter(dim_state) # state0, goal have
    goal0  = cp.Parameter(dim_state) # arrays of len = 4

    lower_obs = cp.Parameter((2, world.MAX)) # rows = 2 for x, y arrays
    upper_obs = cp.Parameter((2, world.MAX)) # cols = world.MAX all obs

## State constraints
    x_pos   = state0[0]
    limit_l = world.limit[0] # lower arr[pos_x, pos_y,
    limit_u = world.limit[1] # upper arr vel_x, vel_y]

    lower_x = cp.vstack([x_pos - world.TOL] + limit_l[1:]) # arr[pos-TOL, -5, -1, -1]
    upp_fov = cp.minimum(x_pos + robot.FOV, limit_u[0])    # min(pos +FOV, maximum x)
    upper_x = cp.vstack([upp_fov] + limit_u[1:])           # arr[lim_u +TOL, 5, 1, 1]
    
    lower_x = lower_x[:, 0]      # resize arr shape from
    upper_x = upper_x[:, 0]      # (4, 1) to (4) idk why

    lower_u = np.array([-1, -1]) # input u_t lies within
    upper_u = np.array([ 1,  1]) # low_u <= u_t <= upp_u


#### Obstacle avoidance ####
    # Declaring binary variables for obstacle avoidance paths
    bool_low, bool_upp = [], []

    for _ in range(world.MAX):
        if relaxed:
            bool_low.append(cp.Parameter((2, N), boolean=True))
            bool_upp.append(cp.Parameter((2, N), boolean=True))
        else:
            bool_low.append(cp.Variable((2, N), boolean=True))
            bool_upp.append(cp.Variable((2, N), boolean=True))

    # Big-M hardcoded to 2 * upper_limit_x, 2 * upper_limit_y
    M = np.diag([2 * limit_u[0], 2 * limit_u[1]])
    
    constraints = [state[:, 0] == state0] # state constraints
    objective = 0
    
    for k in range(N):
        ## SEE SCREENSHOT 1 ##
        # @ is matrix (dot) multiplication
        constraints += [state[:, k + 1] == A @ state[:, k] + B @ input[:, k]]   # add dynamics constraints

        constraints += [lower_x <= state[:, k + 1], upper_x >= state[:, k + 1]] # adding state constraints
    
        constraints += [lower_u <= input[:, k], upper_u >= input[:, k]]         # adding input constraints

        # big-M formulation of obstacle avoidance constraints
        for i in range(world.MAX):


            constraints += [
                state[0:2, k + 1] <= lower_obs[:, i] + M @ bool_low[i][:, k],
                state[0:2, k + 1] >= upper_obs[:, i] - M @ bool_upp[i][:, k]]
            
            # IF YOU SATISFY ALL 4 OF OBS'S CONSTRAINTS, YOURE IN THE OBS.
            if not relaxed:
                constraints += [bool_low[i][0, k] + bool_low[i][1, k] +
                                bool_upp[i][0, k] + bool_upp[i][1, k] <= 3]

        ## SEE SCREENSHOT 2 ##
        # calculating cumulative cost
        objective += cp.norm(Q @ (state[:, k] - goal0), 1) + cp.norm(R @ input[:, k], 1) 
    
    # adding extreme penalty on terminal state to encourage getting close to the goal
    objective += 100 * cp.norm(Q @ (state[:, -1] - goal0), 1)

    # Define the motion planning problem
    problem = cp.Problem(cp.Minimize(objective), constraints)

    print(f"\nDEBUG: motion_planning() done. return problem, vars, params")

    if relaxed:
        return problem, (state, input), (bool_low, bool_upp, state0, goal0, lower_obs, upper_obs)
    else:
        return problem, (state, input, bool_low, bool_upp), (state0, goal0, lower_obs, upper_obs)


def run_simulations(num_iters, plot_sol):
    # Create the motion planning problem

    lower_arr = [[ 0.5, 1.7, 2.7, 2.7, 3.8], # x coords
                 [-0.3,-0.7,-1.3, 0.3,-0.5]] # y coords
    
    size_arr  = [[0.7, 0.5, 0.5, 0.5, 0.7],  # width: x
                 [1.0, 0.7, 1.0, 1.0, 1.0]]  # height:y
    
    limit = [[0.0,-1.2,-0.7,-0.7], # lower[pos_x, pos_y,
             [5.0, 1.2, 0.7, 0.7]] # upper vel_x, vel_y]
    goal  =  [5.0, 0.0, 0.0, 0.0]
    
    world_obs = ObsMap(lower_arr, size_arr)
    world = World(limit, goal, world_obs, TOL=0.2)

    # Randomize start, get vars & params
    for iter in range(1019, num_iters):

        start = world.random_state(iters=100, bound=0.9)
        robot = Robot(start, world_obs, TIME=0.1, FOV=1.5)

        print(f"\nDEBUG: world.random_state() done: {[round(x, 2) for x in start]}")

        problem, vars, params = motion_planning(world, robot, relaxed=False)

        state , input, bool_low , bool_upp  = vars
        state0, goal0, lower_obs, upper_obs = params

        dist = lambda x: np.linalg.norm(np.array(robot.state) - np.array(x))
  

        # Initialize all CP.parameter values
        while dist(goal) > world.TOL:

            print(f"DEBUG: abs(distance) to goal: {round(dist(goal), 2)}")

            state0.value = np.array(robot.state)
            goal0.value  = np.array(goal)

            robot.detect_obs()
            lower_cpy = copy.deepcopy(robot.local_obs.lower_arr)
            size_cpy  = copy.deepcopy(robot.local_obs.size_arr)

            while len(lower_cpy[0]) < world.MAX:    # Ensure arr len = MAX
                low = min(limit[0][0], limit[0][1]) # Get low within big-M
                
                for i in [0, 1]:             # Add fake obs to x(0) & y(1)
                    lower_cpy[i].append(low) # Fake obs have lower x,y val
                    size_cpy [i].append(0.0) # outside of world; size: 0.0

            lower_obs.value = np.array(lower_cpy)
            upper_obs.value = np.array(lower_cpy) + np.array(size_cpy)

            # Now collect optimized trajectories
            print(f"\nSolving iter = {iter}")
            problem.solve(verbose=False)

            print(f"Status = {problem.status}")
            print(f"Optimal cost = {round(problem.value, 2)}")
            print(f"Solve time = {round(problem.solver_stats.solve_time, 4)}s")


            state_sol = state.value
            input_sol = input.value

            # Rounds values to X.0; returns list
            arnd = lambda x: np.around(x.value, 1).tolist()

            bl_sol, bu_sol = [], []
            obs = [lower_cpy, size_cpy]

            for i in range(world.MAX):
                bl_sol.append(arnd(bool_low[i]))
                bu_sol.append(arnd(bool_upp[i]))

            world.solutions.append([robot.state, obs, bl_sol, bu_sol])

            robot.update_state(input_sol[0][0], input_sol[1][0])
            # 1st value in arr(    x_accel    ,    y_accel     )

            if plot_sol:
                world.plot_problem(state_sol, start, goal)

        if plot_sol:
            world.plot_problem(np.array(robot.state_traj), start, goal)

        file = open(f"data/sol{iter}.pkl", "wb")
        pickle.dump(world.solutions, file)
        world.solutions = []


if __name__ == "__main__":
    run_simulations(num_iters=2000, plot_sol=False)