import numpy as np

# R matrix
# We can traverse to 0 and 100 nodes from the matrix but NOT -1
R = np.matrix([[-1, -1, -1, -1, 0, -1],
               [-1, -1, -1, 0, -1, 100],
               [-1, -1, -1, 0, -1, -1],
               [-1, 0, 0, -1, 0, -1],
               [-1, 0, 0, -1, -1, 100],
               [-1, 0, -1, -1, 0, 100]])

# Q matrix, making it 6 by 6 since we obviously have 6 states starting from 0 to 5, and initializing to zero
Q = np.matrix(np.zeros([6, 6]))

# Gamma (Learning parameter)
gamma = 0.8

# Initial state (Usually to be chosen at random)
initial_state = 1

# This function returns all available actions in the state given as an argument
# Firstly we are checking row number 1 from the R matrix, and we are checking for values greater than or equal to 0
# That represents our available actions, since we can traverse to those nodes
def available_actions(state):
    current_state_row = R[state, ]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act

# Get available actions in the current state
# Pretty much just storing the possible actions in this "available_act" variable.
available_act = available_actions(initial_state)

# This function chooses at random which action to be performed within the range of all the available actions.
# Then once it chooses an action it will store it into the next_action variable.
def Sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_act, 1)[0])
    return next_action

# Sample next action to be performed
action = Sample_next_action(available_act)

# This function updates the Q matrix according to the path selected and the Q learning algorithm
# In the first line we are calculating our maximum index, meaning that we're going to check which of the
# possible actions will give us the maximum Q value.
def update(current_state, action, gamma):
    max_index = np.where(Q[action, ] == np.max(Q[action, ]))[1]

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1)[0])
    else:
        max_index = int(max_index[0])
    max_value = Q[action, max_index]

    # Q learning formula, basically computing the value of Q
    Q[current_state, action] = R[current_state, action] + gamma * max_value

# Now we update our Q matrix
update(initial_state, action, gamma)

# TRAINING

# Train over 10,000 iterations (Re-iterate the process above).
# Meaning that our agent will take 10,000 possible scenarios, and it'll go through them all to find the best policy.
# In the first line I am choosing the current state randomly
# after that I am choosing the available action from that current state, then calculating the next action.
# Finally updating the value in the Q matrix.
for i in range(10000):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = Sample_next_action(available_act)
    update(current_state, action, gamma)

# Normalize the "trained" Q matrix
# Computation is harder on larger numbers such as 500 or 600, so we are taking our calculated value and
# dividing it with the maximum Q value into 100
print("Trained Q matrix: ")
print(Q / np.max(Q) * 100)

# TESTING

# Now since we trained a model, let's test by telling our agent he is in room number 1, and now you need to go to
# room number 5, so start the sequence.
# Goal state = 5
# Best sequence path starting from 2 -> 2, 3, 1, 5

current_state = 2
steps = [current_state]

# This is the same loop we executed earlier, so we are going to do the same iterations again
while current_state != 5:
    next_step_index = np.where(Q[current_state, ] == np.max(Q[current_state, ]))[1]

    if next_step_index[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1)[0])
    else:
        next_step_index = int(next_step_index[0])

    steps.append(next_step_index)
    current_state = next_step_index

# Print selected sequence of steps
print("Selected path: ")
print(steps)

# The output shows the best sequence of states that the agent would take to reach the goal state of 5.
