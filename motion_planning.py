import copy
import random
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
        self.size_arr = size_arr

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
    limit = [[0.0, -5.0,-1.0,-1.0], # lower[pos_x, pos_y,
             [20.0, 5.0, 1.0, 1.0]] # upper vel_x, vel_y]
    goal = [x_pos, y_pos, x_vel, y_vel]

    global_obs = Obstacle_Map()
    MAX = integer, max # of obs
    """
    def __init__(self, limit, goal, global_obs, MAX):
        self.limit = limit
        self.goal = goal
        self.MAX = MAX

        self.global_obs = global_obs
        self.solutions = [] # [[start, robot.state, bl_sol, bu_sol], [...] ...]
        self.trajects  = [] # [[start, goal0, state_traj, input_traj], [ ] ...]


    def random_state(self, bound):
        lower, upper = self.limit # unpack list -> 2 vars

        for _ in range(100):                           # random starting state
            x = random.uniform(lower[0], upper[0] * bound) # in x_width * bound
            y = random.uniform(lower[1], upper[1])     # arr[0] = x, arr[1] = y

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
        # Graph the motion planning problem
        # %matplotlib inline

        figure = plt.figure()
        plt.gca().add_patch(Rectangle((0, -5), 20, 10, linewidth=5.0, 
                            ec='g', fc='w', alpha=0.2, label="boundary"))
    
        plt.plot(x_sol[0, :], x_sol[1, :], 'o', label="trajectory")
        plt.plot(start[0], start[1], "*", linewidth=10, label="start")
        plt.plot(goal[0], goal[1], '*', linewidth=10, label="goal")

        obs_lower = self.global_obs.lower_arr
        obs_size = self.global_obs.size_arr

        for i in range(len(self.global_obs)):
            label = "obstacle" if i == 0 else ""

            plt.gca().add_patch(Rectangle((obs_lower[0][i], obs_lower[1][i]),
                obs_size[0][i], obs_size[1][i], ec='r', fc='r', label=label))
        plt.legend(loc = 4)
        plt.show()


class Robot:
    """
    state = [pos_x, pos_y, vel_x, vel_y]
    global/local_obs = Obstacle_Map()

    TIME = seconds between state updates
    FOV = Field Of View: range in x dir.
    """
    def __init__(self, state, global_obs, TIME, FOV):
        self.state = state
        self.TIME = TIME
        self.FOV = FOV

        self.global_obs = global_obs # Obstacle_Map again
        self.local_obs = Obstacle_Map([[], []], [[], []])

        self.state_traj = [[state[0]], [state[1]], [state[2]], [state[3]]]
        self.input_traj = [[], []] # track states, inputs by updating arrs


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
            continue

            # TEMP CODE. Deleted Obs_Map.clean()
            if (in_FOV(obs_lower) or in_FOV(obs_upper)): # FOV partially captures obs

                new_lower = min(obs_lower, x + self.FOV) # new lower and upper coords
                new_upper = min(obs_upper, x + self.FOV) # min(obs_corner, FOVs edge)

                new_size = new_upper - new_lower         # new size_x = upper - lower
                self.local_obs.insert((new_lower, new_size), (lower_y[i], size_y[i]))

        print(f"\nDEBUG: detect_obs() done. local_obs:\n{self.local_obs}")
    

    def update_state(self, acc_x, acc_y):
        t = self.TIME
        pos_x, pos_y, vel_x, vel_y = self.state   # unpack state -> 4 vars

        pos_x += vel_x * t + (0.5 * acc_x * t ** 2)
        pos_y += vel_y * t + (0.5 * acc_y * t ** 2)
        vel_x += acc_x * t # x = x0 + v * t + 0.5(a * t^2)
        vel_y += acc_y * t # v = v0 + a * t
        
        self.state = [pos_x, pos_y, vel_x, vel_y] # assign new state array
        for i in range(4):                        # write calculated state
            self.state_traj[i].append(self.state[i])

        self.input_traj[0].append(acc_x) # write given input vals
        self.input_traj[1].append(acc_y)

        print(f"\nDEBUG: update_state() done. Robot.state:\n{self.state}")


def motion_planning(world, robot):
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

    # State updates described by A and B should match Robot.update_state()
    A = np.matrix(
        [[1, 0, dt,0],
         [0, 1, 0,dt],
         [0, 0, 1, 0],
         [0, 0, 0, 1]])
    
    B = dt * np.matrix(
        [[0.5 * dt, 0],
         [0, 0.5 * dt],
         [1, 0],
         [0, 1]])
    
    dim_state = A.shape[1]
    dim_input = B.shape[1]
    

#### Robot constraints ####
    ## SEE SCREENSHOT 2 ##
    Q = 100 * np.identity(dim_state)  # originally 1000
    R = 50  * np.identity(dim_input)
    N = 50
    
## Vars & Parameters
    
    # Declare variables for state and input trajectories
    state = cp.Variable((dim_state, N + 1)) # STATE IS X
    input = cp.Variable((dim_input, N))     # INPUT IS U

    # Declare parameters for state0, goal, and obstacles
    state0 = cp.Parameter(dim_state) # state0, goal have
    goal   = cp.Parameter(dim_state) # arrays of len = 4

    obs_lower = cp.Parameter((2, world.MAX)) # rows = 2 for x and y array
    obs_upper = cp.Parameter((2, world.MAX)) # cols = world.MAX of all obs

## State constraints

    x = state0[0]            # state = [x_pos, y_pos,]
    limit_l = world.limit[0] # lower arr[pos_x, pos_y,
    limit_u = world.limit[1] # upper arr vel_x, vel_y]

    lower_x = cp.vstack([x - 0.1] + limit_l[1:])      # arr[pos, -5, -1, -1]
    min_arr = [cp.minimum(x + robot.FOV, limit_u[0])] # do min(pos+FOV, u[0])
    upper_x = cp.vstack(min_arr + limit_u[1:])        # arr[pos+FOV, 5, 1, 1]

    lower_x = lower_x[:, 0]      # real scuffed solution
    upper_x = upper_x[:, 0]      # .shape (4, 1) to (4,)

    lower_u = np.array([-2, -2]) # input u_t lies within
    upper_u = -1 * lower_u       # low_u <= u_t <= upp_u

    print(f"\nDEBUG: CP variables done.\nlower_x = {lower_x},\nupper_x = {upper_x}")


#### Obstacle avoidance ####

    # Declaring binary variables for obstacle avoidance formulation
    boxes_low = [cp.Variable((2, N), boolean=True) for _ in range(world.MAX)] # BOXES_LOW IS B_L
    boxes_upp = [cp.Variable((2, N), boolean=True) for _ in range(world.MAX)] # BOXES_UPP IS B_U

    # FIXME: Big-M hardcoded to 2 * upper_limit_x, 2 * upper_limit_y
    M = np.diag([2 * limit_u[0], 2 * limit_u[1]])
    
    constraints = [state[:, 0] == state0] # initial state constraint
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
                state[0:2, k + 1] <= obs_lower[:, i] + M @ boxes_low[i][:, k],
                state[0:2, k + 1] >= obs_upper[:, i] - M @ boxes_upp[i][:, k]]
            

            # IF YOU SATISFY ALL 4 OF BOX'S CONSTRAINTS, YOURE IN THE BOX.
            constraints += [
                boxes_low[i][0, k] + boxes_low[i][1, k] + boxes_upp[i][0, k] + boxes_upp[i][1, k] <= 3]

        ## SEE SCREENSHOT 2 ##
        # calculating cumulative cost
        objective += cp.norm(Q @ (state[:, k] - goal), 'inf') + cp.norm(R @ input[:, k], 'inf') 
    
    # adding extreme penalty on terminal state to encourage getting close to the goal
    objective += 100 * cp.norm(Q @ (state[:, -1] - goal), 'inf')

    # Define the motion planning problem
    problem = cp.Problem(cp.Minimize(objective), constraints)

    print(f"\nDEBUG: motion_planning() done. return problem, vars, params")
    return problem, (state, input, boxes_low, boxes_upp), (state0, goal, obs_lower, obs_upper)


def run_simulations(num_iters, plot_steps, plot_period):
    # Create the motion planning problem

    lower_arr = [[0.0, 2.0, 2.0, 5.0, 7.5, 10.0, 12.0, 12.0, 15.0, 17.5], # x coords
                 [1.0,-5.0, 3.0,-2.0,-5.0, 1.0, -5.0,  3.0, -2.0, -5.0]]  # y coords
    
    size_arr  = [[1.5, 2.5, 2.5, 2.0, 2.0, 1.5, 2.5, 2.5, 2.0, 2.0],      # width: x
                 [2.0, 7.0, 2.0, 6.5, 6.0, 2.0, 7.0, 2.0, 6.5, 6.0]]      # height:y
    
    goal0 =  [20.0, 0.0, 0.0, 0.0]
    limit = [[0.0, -4.9,-1.0,-1.0], # lower[pos_x, pos_y,
             [20.0, 4.9, 1.0, 1.0]] # upper vel_x, vel_y]
    
    global_obs = Obstacle_Map(lower_arr, size_arr)
    world = Environment(limit, goal0, global_obs, MAX = 10)

    # Randomize start, get vars & params
    for _ in range(num_iters):

        start = world.random_state(bound = 0.5)
        print(f"\nDEBUG: world.random_state() done. start = {[round(x, 4) for x in start]}")

        robot = Robot(start, global_obs, TIME=0.2, FOV=10.0)
        problem, vars, params = motion_planning(world, robot)

        state, input, boxes_low, boxes_upp = vars
        state0, goal, obs_lower, obs_upper = params

        diff = np.array(robot.state) - np.array(goal0)
        TOL = 0.1 # 0.01 is too small

  
        # Initialize all CP parameter values
        while np.linalg.norm(diff) > TOL: # while not @ goal

            print(f"\nDEBUG: |distance| to goal = {np.linalg.norm(diff)}")

            state0.value = np.array(robot.state)
            goal.value = np.array(goal0)

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
            print("\nProblem: solving...")
            problem.solve(verbose = False)

            print(f"Status = {problem.status}")
            print(f"Optimal cost = {problem.value}")
            print(f"Solve time = {problem.solver_stats.solve_time} seconds")


            x_sol = state.value
            u_sol = input.value
            bl_sol = [boxes_low[i].value for i in range(world.MAX)]
            bu_sol = [boxes_upp[i].value for i in range(world.MAX)]

            # Collect solutions in world & robot
            world.solutions.append([start, robot.state, bl_sol, bu_sol])

            # 1st value in arr(  x_accel  ,   y_accel  )
            robot.update_state(u_sol[0][0], u_sol[1][0])

            diff = np.array(robot.state) - np.array(goal0)
            sim_time = len(robot.state_traj[0]) - 1

            if plot_steps and (sim_time % plot_period == 0):
                world.plot_problem(x_sol, start, goal0)
        
        world.trajects.append([start, goal0, robot.state_traj, robot.input_traj])
        world.plot_problem(np.array(robot.state_traj), start, goal0)

run_simulations(num_iters=1, plot_steps=True, plot_period=10) # Make this true to see every plot