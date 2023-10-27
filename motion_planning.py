import os
import copy
import random
import pickle
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Obstacle_Map:
    """
    lower_arr = [[0.0, 2.0, 2.0, 5.0, 7.0],   # x coords
                 [1.0,-5.0, 3.0,-2.0,-5.0]]   # y coords

    size_arr  = [[1.5, 2.5, 2.5, 1.7, 2.0],   # width: x
                 [2.0, 7.0, 2.0, 6.5, 8.0]]   # height:y
    """
    def __init__(self, lower_arr, size_arr):
        self.lower_arr = lower_arr
        self.size_arr  = size_arr

        case = 0 # checks both arrays have x, y
        if not (len(lower_arr) == len(size_arr) == 2):
            case = 1
        else:    # checks all arrays are equal len
            for arr in lower_arr + size_arr:
                if len(arr) != len(lower_arr[0]):
                    case = 2
                    break       
                if not all(isinstance(i, float) for i in arr):
                    case = 3    # checks all values are floats
                    break 
        if case != 0:
            print(f"\nERROR: Obs_Map.init() failed @ case {case}"); exit()


    def insert(self, items_x, items_y): # items_# = (lower_#, size_#)
        lower_x, size_x = items_x
        lower_y, size_y = items_y

        found = False
        for i in range(len(self.lower_arr[0])):
            # check all 4 vals at index i

            if (lower_x == self.lower_arr[0][i] and
                lower_y == self.lower_arr[1][i] and
                size_x == self.size_arr[0][i] and
                size_y == self.size_arr[1][i]):
                found = True

        if not found: # append all 4 vals to arrays
            self.lower_arr[0].append(lower_x)
            self.lower_arr[1].append(lower_y)
            self.size_arr[0].append(size_x)
            self.size_arr[1].append(size_y)

    
    def unwrap(self): # return (x_l, x_s, y_l, y_s)
        lower = self.lower_arr
        size = self.size_arr
        return lower[0], size[0], lower[1], size[1]
    
    def __str__(self):
        return f"lower_arr = {self.lower_arr}\nsize_arr = {self.size_arr}"
    
    def __len__(self):
        return len(self.size_arr[0])


class Environment:
    """
    limit = [[0.0, -4.9,-1.0,-1.0], # lower[pos_x, pos_y,
             [20.0, 4.9, 1.0, 1.0]] # upper vel_x, vel_y]
    goal = [x_pos, y_pos, x_vel, y_vel]

    global_obs = Obstacle_Map()
    MAX = integer, max # of obs
    """
    def __init__(self, limit, goal, global_obs, TOL):
        self.limit = limit
        self.goal  = goal
        self.TOL   = TOL
        self.global_obs = global_obs
        self.MAX   = len(global_obs)

        self.solutions = [] # [[state, bl_sol, bu_sol], ...]
        self.trajects  = [] # [[state_traj, input_traj] ...]


    def random_state(self, iters, bound):
        lower, upper = self.limit # unpack array -> 2 vars

        for _ in range(iters):                             # random starting state:
            x = random.uniform(lower[0], upper[0] * bound) # within x_width * bound
            y = random.uniform(lower[1] + self.TOL, upper[1] - self.TOL)

            lower_x, size_x, lower_y, size_y = self.global_obs.unwrap()

            upper_x = np.array(lower_x) + np.array(size_x) # upper_x, upper_y =
            upper_y = np.array(lower_y) + np.array(size_y) # lower_xy + size_xy
            
            in_obs = False
            for i in range(len(self.global_obs)): # loop through every obs
                if (x >= lower_x[i] and x <= upper_x[i] and
                    y >= lower_y[i] and y <= upper_y[i]):
                    in_obs = True                 # if inside walls of obs
                    break                         # break to get new state
            if not in_obs:                        # else return this state
                return [x, y, 0.0, 0.0]

        print("\nERROR: random_state() couldn't find valid state"); exit()


    def plot_problem(self, x_sol, start, goal):
        obs_lower = self.global_obs.lower_arr
        obs_size = self.global_obs.size_arr

        plt.gca().add_patch(Rectangle((0, -5), 20, 10, linewidth=5.0, 
                            ec='g', fc='w', alpha=0.2, label="boundary"))
    
        plt.plot(x_sol[0, :], x_sol[1, :], 'o', label="trajectory")
        plt.plot(start[0], start[1], "*", linewidth=10, label="start")
        plt.plot(goal[0], goal[1], '*', linewidth=10, label="goal")

        for i in range(len(self.global_obs)):
            label = "obstacle" if i == 0 else ""

            plt.gca().add_patch(Rectangle((obs_lower[0][i], obs_lower[1][i]),
                obs_size[0][i], obs_size[1][i], ec='r', fc='r', label=label))
        
        plt.legend(loc = 4)
        plt.show()


    def export_files(self):
        obs = self.global_obs
        data = [self.limit, self.goal, obs.lower_arr, obs.size_arr]

        if not os.path.exists("data"):
            os.mkdir("data")

        with open("data/solutions.pkl", "wb") as x:
            pickle.dump([data] + self.solutions, x)

        with open("data/trajects.pkl", "wb") as x:
            pickle.dump([data] + self.trajects, x)

        self.solutions = []
        self.trajects  = []


class Robot:
    """
    state = [pos_x, pos_y, vel_x, vel_y]
    global/local_obs = Obstacle_Map()

    TIME = seconds between state updates
    FOV = Field Of View: range in x dir.
    """
    def __init__(self, state, global_obs, TIME, FOV):
        self.state = state
        self.TIME  = TIME
        self.FOV   = FOV

        self.global_obs = global_obs
        self.local_obs  = Obstacle_Map([[], []], [[], []])

        self.state_traj = [[], [], [], []]    # track vars by updating arr
        self.input_traj = [[], []]


    def detect_obs(self):
        x = self.state[0]                     # state = [x_pos, y_pos,...]
        lower_x, size_x, lower_y, size_y = self.global_obs.unwrap()

        for i in range(len(self.global_obs)): # loop through each obstacle
            obs_lower = lower_x[i]            # obs's lower corner x value
            obs_size = size_x[i]              # obs's width in x direction
            
            obs_upper = obs_lower + obs_size              # obs's upper corner x value
            in_FOV = lambda obs: obs >= x and obs <= x + self.FOV  # is obs_corner in robot.FOV

            if (in_FOV(obs_lower) or in_FOV(obs_upper)): # FOV fully capture obstacle
                self.local_obs.insert((obs_lower, obs_size), (lower_y[i], size_y[i]))
                # add unchanged vals ((      x_items      ), (       y_items       ))
        return
        print(f"\nDEBUG: detect_obs() done. local_obs:\n{self.local_obs}")
    

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


def motion_planning(world, robot, relaxed):
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
    ## SEE SCREENSHOT 2 ##
    Q = 100 * np.identity(dim_state)
    R = 50  * np.identity(dim_input)
    N = 50
    
## Vars & Parameters
    # Declare variables for state and input trajectories
    state = cp.Variable((dim_state, N + 1)) # STATE IS X
    input = cp.Variable((dim_input, N))     # INPUT IS U

    # Declare parameters for state0, goal, and obstacles
    state0 = cp.Parameter(dim_state) # state0, goal have
    goal0  = cp.Parameter(dim_state) # arrays of len = 4

    obs_lower = cp.Parameter((2, world.MAX)) # rows = 2 for x and y array
    obs_upper = cp.Parameter((2, world.MAX)) # cols = world.MAX of all obs

## State constraints
    x = state0[0]
    limit_l = world.limit[0] # lower arr[pos_x, pos_y,
    limit_u = world.limit[1] # upper arr vel_x, vel_y]

    lower_x = cp.vstack([x - world.TOL] + limit_l[1:]) # arr[pos-TOL, -5, -1, -1]
    upp_fov = cp.minimum(x + robot.FOV, limit_u[0])    # min(pos+FOV, limit_u[0])
    upper_x = cp.vstack([upp_fov] + limit_u[1:])       # arr[upp_fov, 5, 1, 1]

    lower_x = lower_x[:, 0]      # resize arr shape from
    upper_x = upper_x[:, 0]      # (4, 1) to (4) idk why

    lower_u = np.array([-2, -2]) # input u_t lies within
    upper_u = -1 * lower_u       # low_u <= u_t <= upp_u

    print(f"\nDEBUG: CP variables done.\nlower_x = {lower_x},\nupper_x = {upper_x}")


#### Obstacle avoidance ####
    # Declaring binary variables for obstacle avoidance formulation
    bool_low, bool_upp = [], []

    for _ in range(world.MAX):
        if relaxed:
            bool_low.append(cp.Parameter((2, N), boolean=True))
            bool_upp.append(cp.Parameter((2, N), boolean=True))
        else:
            bool_low.append(cp.Variable((2, N), boolean=True))
            bool_upp.append(cp.Variable((2, N), boolean=True))

    # DONE: Big-M hardcoded to 2 * upper_limit_x, 2 * upper_limit_y
    M = np.diag([2 * limit_u[0], 2 * limit_u[1]])
    
    constraints = [state[:, 0] == state0]# initial state constraint
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
                state[0:2, k + 1] <= obs_lower[:, i] + M @ bool_low[i][:, k],
                state[0:2, k + 1] >= obs_upper[:, i] - M @ bool_upp[i][:, k]]
            
            # IF YOU SATISFY ALL 4 OF OBS'S CONSTRAINTS, YOURE IN THE OBS.
            constraints += [
                bool_low[i][0, k] + bool_low[i][1, k] + bool_upp[i][0, k] + bool_upp[i][1, k] <= 3]

        ## SEE SCREENSHOT 2 ##
        # calculating cumulative cost
        objective += cp.norm(Q @ (state[:, k] - goal0), 'inf') + cp.norm(R @ input[:, k], 'inf') 
    
    # adding extreme penalty on terminal state to encourage getting close to the goal
    objective += 100 * cp.norm(Q @ (state[:, -1] - goal0), 'inf')

    # Define the motion planning problem
    problem = cp.Problem(cp.Minimize(objective), constraints)

    print(f"\nDEBUG: motion_planning() done. return problem, vars, params")

    if relaxed:
        return problem, (state, input), (bool_low, bool_upp, state0, goal0, obs_lower, obs_upper)
    else:
        return problem, (state, input, bool_low, bool_upp), (state0, goal0, obs_lower, obs_upper)


def run_simulations(num_iters, plot_period, plot_steps):
    # Create the motion planning problem

    lower_arr = [[0.0, 2.0, 2.0, 5.0, 7.5, 10.0, 12.0, 12.0, 15.0, 17.5], # x coords
                 [1.0,-5.0, 3.0,-2.0,-5.0, 1.0, -5.0,  3.0, -2.0, -5.0]]  # y coords
    
    size_arr  = [[1.5, 2.5, 2.5, 2.0, 2.0, 1.5, 2.5, 2.5, 2.0, 2.0],      # width: x
                 [2.0, 7.0, 2.0, 6.5, 6.0, 2.0, 7.0, 2.0, 6.5, 6.0]]      # height:y
    
    goal  =  [20.0, 0.0, 0.0, 0.0]
    limit = [[0.0, -4.9,-1.0,-1.0], # lower[pos_x, pos_y,
             [20.0, 4.9, 1.0, 1.0]] # upper vel_x, vel_y]
    
    global_obs = Obstacle_Map(lower_arr, size_arr)
    world = Environment(limit, goal, global_obs, TOL = 0.1)

    # Randomize start, get vars & params
    for iter in range(num_iters):

        start = world.random_state(iters=100, bound=0.5)
        print(f"\nDEBUG: world.random_state() done: {[round(x, 2) for x in start]}")

        robot = Robot(start, global_obs, TIME=0.2, FOV=10.0)
        problem, vars, params = motion_planning(world, robot, relaxed=False)

        state, input, bool_low, bool_upp = vars
        state0, goal0, obs_lower, obs_upper = params

        dist = lambda x: np.linalg.norm(np.array(robot.state) - np.array(x))

  
        # Initialize all CP.parameter values
        while dist(goal) > world.TOL:

            print(f"DEBUG: abs(distance) to goal: {round(dist(goal), 2)}")

            state0.value = np.array(robot.state)
            goal0.value = np.array(goal)

            robot.detect_obs()
            l = copy.deepcopy(robot.local_obs.lower_arr)
            s = copy.deepcopy(robot.local_obs.size_arr)

            while (len(l[0]) < world.MAX):
                for i in range(2):
                    l[i].append(-2.0) # [len(L) to world.MAX] are fake obs
                    s[i].append(0.0)  # fake obs have lower x,y: -2.0,-2.0

            obs_lower.value = np.array(l)
            obs_upper.value = np.array(l) + np.array(s)

            # Now collect optimized trajectories
            print(f"\nSolving iter = {iter}")
            problem.solve(verbose = False)

            print(f"Status = {problem.status}")
            print(f"Optimal cost = {round(problem.value, 2)}")
            print(f"Solve time = {round(problem.solver_stats.solve_time, 2)} secs.")

            x_sol = state.value
            u_sol = input.value
            bl_sol, bu_sol = [], []


            for i in range(world.MAX): # convert np.array to py.lists
                bl_sol.append(bool_low[i].value.tolist())
                bu_sol.append(bool_upp[i].value.tolist())

            if (len(robot.state_traj[0]) - 1) % plot_period == 0:

                # every plot_period steps, collect solutions in world
                world.solutions.append([robot.state, bl_sol, bu_sol])
                if plot_steps:
                    world.plot_problem(x_sol, start, goal)
            
            robot.update_state(u_sol[0][0], u_sol[1][0])
            # 1st value in arr(  x_accel  ,   y_accel  )

        if plot_steps:
            world.plot_problem(np.array(robot.state_traj), start, goal)

        world.trajects.append([robot.state_traj, robot.input_traj])
        
        world.export_files()

if __name__ == "__main__": # Set True to see every plot_period steps
    run_simulations(num_iters=1, plot_period=10, plot_steps=False)