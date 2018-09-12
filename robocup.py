import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from scipy.linalg import block_diag
import time

solvers.options['show_progress'] = False

# Env
class Soccer:
    def __init__(self):
        self.pos = [np.array([0, 2]), np.array([0, 1])]
        self.goal = [0, 3]
        self.ball = 1 # np.random.choice([0, 1], 1)[0]

    def move(self, actions):
        all_actions = [[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]]    # 0 = N, 1 = E, 2 = S, 3 = W, 4 = Stick
        first_mover = np.random.choice([0, 1], 1)[0]
        second_mover = 1-first_mover
        rewards = np.array([0, 0])
        new_pos = self.pos.copy()
        done = 0

        if actions[0] < 0 | actions[0] > 4 | actions[1] < 0 | actions[1] > 4:
            print('Illegal Action: Has to be one of [0, 1, 2, 3, 4]')
            return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1], self.ball], rewards, done
        else:
            # Move first player
            new_pos[first_mover] = self.pos[first_mover] + all_actions[actions[first_mover]]
            if np.array_equal(new_pos[first_mover], self.pos[second_mover]):    # Collide
                if self.ball == first_mover:    # Lose ball
                    self.ball = second_mover
            elif new_pos[first_mover][0] >= 0 and new_pos[first_mover][0] <= 1 and new_pos[first_mover][1] >= 0 and new_pos[first_mover][1] <= 3:  # Legal move
                self.pos[first_mover] = new_pos[first_mover]
                if self.ball == first_mover and self.pos[first_mover][1] == self.goal[first_mover]:      # Player scored
                    rewards = np.array([100, -100]) * [1, -1][first_mover]
                    done = 1
                    return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1], self.ball], rewards, done
                elif self.ball == first_mover and self.pos[first_mover][1] == self.goal[second_mover]:      # Player scored for opponent
                    rewards = np.array([-100, 100]) * [1, -1][first_mover]
                    done = 1
                    return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1], self.ball], rewards, done
            # Move second player
            new_pos[second_mover] = self.pos[second_mover] + all_actions[actions[second_mover]]
            if np.array_equal(new_pos[second_mover], self.pos[first_mover]):  # Collide
                if self.ball == second_mover:  # Lose ball
                    self.ball = first_mover
            elif new_pos[second_mover][0] >= 0 and new_pos[second_mover][0] <= 1 and new_pos[second_mover][1] >= 0 and new_pos[second_mover][1] <= 3:  # Legal move
                self.pos[second_mover] = new_pos[second_mover]
                if self.ball == second_mover and self.pos[second_mover][1] == self.goal[second_mover]:  # Player scored
                    rewards = np.array([100, -100]) * [1, -1][second_mover]
                    done = 1
                    return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1], self.ball], rewards, done
                elif self.ball == second_mover and self.pos[second_mover][1] == self.goal[first_mover]:  # Player scored for opponent
                    rewards = np.array([-100, 100]) * [1, -1][second_mover]
                    done = 1
                    return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1], self.ball], rewards, done

        return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1], self.ball], rewards, done

    def eval(self):
        self.grid = np.array([['GA ', '   ', '   ', 'GB '], ['GA ', '   ', '   ', 'GB ']])
        if self.ball == 0:
            self.grid[tuple(self.pos[0])] = self.grid[tuple(self.pos[0])][0:2] + 'A'
            self.grid[tuple(self.pos[1])] = self.grid[tuple(self.pos[1])][0:2] + 'b'
        else:
            self.grid[tuple(self.pos[0])] = self.grid[tuple(self.pos[0])][0:2] + 'a'
            self.grid[tuple(self.pos[1])] = self.grid[tuple(self.pos[1])][0:2] + 'B'
        print(self.grid)

def plot_error(errors, title, pos):
    plt.figure(pos)
    plt.clf()
    plt.title(title)
    plt.xlabel('# of Iterations')
    plt.ylabel('Q-value Difference')
    plt.ylim(0, 0.5)
    plt.plot(errors, linestyle='-', color='black', linewidth=0.5)
    plt.pause(0.001)



# Q-learning
def q_learning(n=1000000):
    # Parameters
    np.random.seed(1)
    gamma = 0.9
    epsilon_begin = 0.1
    epsilon_end = 0
    epsilon_periods = n/2
    end_alpha = 0.001
    Q1 = np.zeros((8, 8, 2, 5))
    Q2 = np.zeros((8, 8, 2, 5))
    errors = []

    # Evaluation function
    def take_action(Q, state, i):
        epsilon = epsilon_end + (epsilon_begin - epsilon_end) * np.exp(-1.0 * i / epsilon_periods)
        if np.random.random() > epsilon:
            return np.random.choice(np.where(Q[state[0]][state[1]][state[2]] == max(Q[state[0]][state[1]][state[2]]))[0], 1)[0]
        else:
            return np.random.choice([0,1,2,3,4], 1)[0]

    # Loop for n steps
    begin_time = time.time()
    i = 0
    while i < n:
        soccer = Soccer()
        state = [soccer.pos[0][0] * 4 + soccer.pos[0][1], soccer.pos[1][0] * 4 + soccer.pos[1][1], soccer.ball]
        while True:
            if (i + 1) % 100 == 0: print(i + 1, ', ', np.round(time.time() - begin_time, 0), 'seconds')
            i += 1
            begin_qt = Q1[2][1][1][2]
            # Choose action
            actions = [take_action(Q1, state, i), take_action(Q2, state, i)]
            # Observe R and next state
            state_next, rewards, done = soccer.move(actions)
            alpha = 1 / (i / end_alpha / n + 1)
            # Update Q
            if done:
                Q1[state[0]][state[1]][state[2]][actions[0]] = Q1[state[0]][state[1]][state[2]][actions[0]] + alpha * (rewards[0] - Q1[state[0]][state[1]][state[2]][actions[0]])
                Q2[state[0]][state[1]][state[2]][actions[1]] = Q2[state[0]][state[1]][state[2]][actions[1]] + alpha * (rewards[1] - Q2[state[0]][state[1]][state[2]][actions[1]])
                break
            else:
                Q1[state[0]][state[1]][state[2]][actions[0]] = Q1[state[0]][state[1]][state[2]][actions[0]] + alpha * (rewards[0] + gamma * max(Q1[state_next[0]][state_next[1]][state_next[2]]) - Q1[state[0]][state[1]][state[2]][actions[0]])
                Q2[state[0]][state[1]][state[2]][actions[1]] = Q2[state[0]][state[1]][state[2]][actions[1]] + alpha * (rewards[1] + gamma * max(Q2[state_next[0]][state_next[1]][state_next[2]]) - Q2[state[0]][state[1]][state[2]][actions[1]])
                state = state_next
            # Calculate error
            end_qt = Q1[2][1][1][2]
            errors.append(np.abs(end_qt - begin_qt))

    return errors


# Friend-Q
def friend_q(n = 10000000):
    # Parameters
    np.random.seed(1)
    gamma = 0.9
    epsilon_begin = 0.1
    epsilon_end = 0
    epsilon_periods = n/2
    end_alpha = 0.001
    Q1 = np.zeros((8, 8, 2, 5, 5))
    Q2 = np.zeros((8, 8, 2, 5, 5))
    errors = []

    # Evaluation function
    def take_action(Q, state, i):
        epsilon = epsilon_end + (epsilon_begin - epsilon_end) * np.exp(-1.0 * i / epsilon_periods)
        if np.random.random() > epsilon:
            max_idx = np.where(Q[state[0]][state[1]][state[2]] == np.max(Q[state[0]][state[1]][state[2]]))
            return max_idx[1][np.random.choice(range(len(max_idx[0])), 1)[0]]
        else:
            return np.random.choice([0,1,2,3,4], 1)[0]

    # Loop for n steps
    begin_time = time.time()
    i = 0
    while i < n:
        soccer = Soccer()
        state = [soccer.pos[0][0] * 4 + soccer.pos[0][1], soccer.pos[1][0] * 4 + soccer.pos[1][1], soccer.ball]
        while True:
            if (i + 1) % 100 == 0: print(i + 1, ', ', np.round(time.time() - begin_time, 0), 'seconds')
            i += 1
            begin_qt = Q1[2][1][1][4][2]
            # Choose action
            actions = [take_action(Q1, state, i), take_action(Q2, state, i)]
            # Observe R and next state
            state_next, rewards, done = soccer.move(actions)
            alpha = 1 / (i / end_alpha / n + 1)
            # Update Q
            if done:
                Q1[state[0]][state[1]][state[2]][actions[1]][actions[0]] = Q1[state[0]][state[1]][state[2]][actions[1]][actions[0]] + alpha * (rewards[0] - Q1[state[0]][state[1]][state[2]][actions[1]][actions[0]])
                Q2[state[0]][state[1]][state[2]][actions[0]][actions[1]] = Q2[state[0]][state[1]][state[2]][actions[0]][actions[1]] + alpha * (rewards[1] - Q2[state[0]][state[1]][state[2]][actions[0]][actions[1]])
                break
            else:
                Q1[state[0]][state[1]][state[2]][actions[1]][actions[0]] = Q1[state[0]][state[1]][state[2]][actions[1]][actions[0]] + alpha * (rewards[0] + gamma * np.max(Q1[state_next[0]][state_next[1]][state_next[2]]) - Q1[state[0]][state[1]][state[2]][actions[1]][actions[0]])
                Q2[state[0]][state[1]][state[2]][actions[0]][actions[1]] = Q2[state[0]][state[1]][state[2]][actions[0]][actions[1]] + alpha * (rewards[1] + gamma * np.max(Q2[state_next[0]][state_next[1]][state_next[2]]) - Q2[state[0]][state[1]][state[2]][actions[0]][actions[1]])
                state = state_next
            # Calculate error
            end_qt = Q1[2][1][1][4][2]
            errors.append(np.abs(end_qt - begin_qt))

    return errors


# Foe-Q
def foe_q(n = 1000000):
    # Parameters
    np.random.seed(1)
    gamma = 0.9
    epsilon_end = 0.001
    epsilon_decay = 10**(np.log10(epsilon_end)/n)
    alpha_end = 0.001
    alpha_decay = 10**(np.log10(alpha_end)/n)
    Q1 = np.ones((8, 8, 2, 5, 5)) * 1.0
    Q2 = np.ones((8, 8, 2, 5, 5)) * 1.0
    V1 = np.ones((8, 8, 2)) * 1.0
    V2 = np.ones((8, 8, 2)) * 1.0
    pi1 = np.ones((8, 8, 2, 5)) * 1/5
    pi2 = np.ones((8, 8, 2, 5)) * 1/5
    errors = []

    # Evaluation function
    def take_action(pi, state, i):
        epsilon = epsilon_decay ** i
        if np.random.random() > epsilon:
            return np.random.choice([0,1,2,3,4], 1, p=pi[state[0]][state[1]][state[2]])[0]
        else:
            return np.random.choice([0,1,2,3,4], 1)[0]

    # Linear programming solver
    def linear_program(Q, state):
        c = matrix([0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        G = matrix(np.append(np.append(-Q[state[0]][state[1]][state[2]], np.ones((5,1)), axis=1), np.append(-np.eye(5), np.zeros((5,1)), axis=1), axis=0))
        h = matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        A = matrix([[1.0], [1.0], [1.0], [1.0], [1.0], [0.0]])
        b = matrix(1.0)
        Solution = solvers.lp(c, G, h, A, b)
        return np.abs(Solution['x'][0:5]).reshape((5,)) / sum(np.abs(Solution['x'][0:5])), np.array(Solution['x'][5])

    # Loop for n steps
    begin_time = time.time()
    i = 0
    while i < n:
        soccer = Soccer()
        state = [soccer.pos[0][0] * 4 + soccer.pos[0][1], soccer.pos[1][0] * 4 + soccer.pos[1][1], soccer.ball]
        done = 0
        while not done:
            if (i + 1) % 100 == 0: print(i + 1, ', ', np.round(time.time() - begin_time, 0), 'seconds')
            i += 1
            begin_qt = Q1[2][1][1][4][2]
            # Choose action
            actions = [take_action(pi1, state, i), take_action(pi2, state, i)]
            # Observe R and next state
            state_next, rewards, done = soccer.move(actions)
            alpha = alpha_decay ** i
            # Update Q/V/pi
            # Player 1
            Q1[state[0]][state[1]][state[2]][actions[1]][actions[0]] = (1 - alpha) * Q1[state[0]][state[1]][state[2]][actions[1]][actions[0]] + alpha * (rewards[0] + gamma * V1[state_next[0]][state_next[1]][state_next[2]])
            prob, val = linear_program(Q1, state)
            pi1[state[0]][state[1]][state[2]] = prob
            V1[state[0]][state[1]][state[2]] = val
            # Player 2
            Q2[state[0]][state[1]][state[2]][actions[0]][actions[1]] = (1 - alpha) * Q2[state[0]][state[1]][state[2]][actions[0]][actions[1]] + alpha * (rewards[1] + gamma * V2[state_next[0]][state_next[1]][state_next[2]])
            prob, val = linear_program(Q2, state)
            pi2[state[0]][state[1]][state[2]] = prob
            V2[state[0]][state[1]][state[2]] = val
            state = state_next
            # Calculate error
            end_qt = Q1[2][1][1][4][2]
            errors.append(np.abs(end_qt - begin_qt))

    return errors


# uCE-Q
def ce_q(n = 1000000):
    # Parameters
    np.random.seed(1)
    gamma = 0.9
    epsilon_end = 0.001
    epsilon_decay = 10**(np.log10(epsilon_end)/n)
    alpha_end = 0.001
    alpha_decay = 10**(np.log10(alpha_end)/n)
    Q1 = np.ones((8, 8, 2, 5, 5)) * 1.0
    Q2 = np.ones((8, 8, 2, 5, 5)) * 1.0
    V1 = np.ones((8, 8, 2)) * 1.0
    V2 = np.ones((8, 8, 2)) * 1.0
    pi = np.ones((8, 8, 2, 5, 5)) * 1/25
    errors = []

    # Evaluation function
    def take_action(pi, state, i):
        epsilon = epsilon_decay ** i
        if np.random.random() > epsilon:
            idx = np.random.choice(np.arange(25), 1, p=pi[state[0]][state[1]][state[2]].reshape(25))
            return np.array([idx // 5, idx % 5]).reshape(2)
        else:
            idx = np.random.choice(np.arange(25), 1)
            return np.array([idx // 5, idx % 5]).reshape(2)

    # Linear programming solver
    def linear_program(Q1, Q2, state):
        # Conditions for Player 1
        Qs = Q1[state[0]][state[1]][state[2]]
        s = block_diag(Qs - Qs[0, :], Qs - Qs[1, :], Qs - Qs[2, :], Qs - Qs[3, :], Qs - Qs[4, :])
        row_idx = (1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23)
        param1 = s[row_idx, :]
        # Conditions for Player 2
        Qs = Q2[state[0]][state[1]][state[2]]
        s = block_diag(Qs - Qs[0, :], Qs - Qs[1, :], Qs - Qs[2, :], Qs - Qs[3, :], Qs - Qs[4, :])
        col_idx = (0, 5, 10, 15, 20, 1, 6, 11, 16, 21, 2, 7, 12, 17, 22, 3, 8, 13, 18, 23, 4, 9, 14, 19, 24)  # To make pi sequence matching player 1
        param2 = s[row_idx, :][:, col_idx]
        # Build parameters
        c = matrix((Q1[state[0]][state[1]][state[2]] + Q2[state[0]][state[1]][state[2]].T).reshape(25))
        G = matrix(np.append(np.append(param1, param2, axis=0), -np.eye(25), axis=0))
        h = matrix(np.zeros(65) * 0.0)
        A = matrix(np.ones((1, 25)))
        b = matrix(1.0)
        # Solver
        try:
            Solution = solvers.lp(c, G, h, A, b)
            if Solution['x'] is not None:
                prob = np.abs(np.array(Solution['x']).reshape((5, 5))) / sum(np.abs(Solution['x']))
                val1 = np.sum(prob * Q1[state[0]][state[1]][state[2]])
                val2 = np.sum(prob * Q2[state[0]][state[1]][state[2]].T)
            else:
                prob = None
                val1 = None
                val2 = None
        except:
            prob = None
            val1 = None
            val2 = None
        return prob, val1, val2

    # Loop for n steps
    begin_time = time.time()
    i = 0
    while i < n:
        soccer = Soccer()
        state = [soccer.pos[0][0] * 4 + soccer.pos[0][1], soccer.pos[1][0] * 4 + soccer.pos[1][1], soccer.ball]
        done = 0
        j = 0
        while not done and j <= 100:
            if (i + 1) % 100 == 0: print(i + 1, ', ', np.round(time.time() - begin_time, 0), 'seconds')
            i += 1
            j += 1
            begin_qt = Q1[2][1][1][2][4]
            # Choose action
            actions = take_action(pi, state, i)
            # Observe R and next state
            state_next, rewards, done = soccer.move(actions)
            alpha = alpha_decay ** i
            # Update Q/V/pi
            Q1[state[0]][state[1]][state[2]][actions[0]][actions[1]] = (1 - alpha) * Q1[state[0]][state[1]][state[2]][actions[0]][actions[1]] + alpha * (rewards[0] + gamma * V1[state_next[0]][state_next[1]][state_next[2]])
            Q2[state[0]][state[1]][state[2]][actions[1]][actions[0]] = (1 - alpha) * Q2[state[0]][state[1]][state[2]][actions[1]][actions[0]] + alpha * (rewards[1] + gamma * V2[state_next[0]][state_next[1]][state_next[2]].T)
            prob, val1, val2 = linear_program(Q1, Q2, state)
            if prob is not None:
                pi[state[0]][state[1]][state[2]] = prob
                V1[state[0]][state[1]][state[2]] = val1
                V2[state[0]][state[1]][state[2]] = val2
            state = state_next
            # Calculate error
            end_qt = Q1[2][1][1][2][4]
            errors.append(np.abs(end_qt - begin_qt))

    return errors


# Q-learning
q_learning_errors = q_learning()

# Friend-Q
friend_q_errors = friend_q()

# Foe-Q
foe_q_errors = foe_q()

# CE-Q
ce_q_errors = ce_q()

# Plot
plot_error(np.array(q_learning_errors)[np.where(np.array(q_learning_errors) > 0)], 'Q-learning', 4)
plot_error(np.array(friend_q_errors)[np.where(np.array(friend_q_errors) > 0)], 'Friend-Q', 3)
plot_error(np.array(foe_q_errors)[np.where(np.array(foe_q_errors) > 0)], 'Foe-Q', 2)
plot_error(np.array(ce_q_errors)[np.where(np.array(ce_q_errors) > 0)], 'uCE-Q', 1)
