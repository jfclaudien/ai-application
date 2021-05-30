# Optimisation d'un processus business (Warehouse)
# Importing the libraries
import numpy as np

# Params
alpha = 0.9
gamma = 0.75


# I- Defining the environment

# States
location_to_state = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'F': 5,
        'G': 6,
        'H': 7,
        'I': 8,
        'J': 9,
        'K': 10,
        'L': 11
        }

state_to_location = {state: location for location, state in location_to_state.items()}

# Actions
actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def route(starting_location, ending_location):
    
    # Rewards
    R = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, -500, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]])
    
    ending_state = location_to_state[ending_location]
    R[ending_state, ending_state] = 1000
    
    # Q-Values Initialisation
    Q = np.zeros([12, 12])
    for _ in range(1000):
        current_state = np.random.randint(0, 12)
        
        playable_actions = []
        for j in range(12):
            if R[current_state, j] > 0:
                playable_actions.append(j)
        
        next_state = np.random.choice(playable_actions)
        
        TD = R[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state, ])] - Q[current_state, next_state]
        
        # Equation de Bellman 
        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD

    
    route = [starting_location]
    next_location = starting_location
    while next_location != ending_location:
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state, ])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route
        

route('E', 'G')
route('A', 'G')

def best_route(starting_location, ending_location, intermediary_location):
    return route(starting_location, intermediary_location) + \
           route(intermediary_location, ending_location)[1: ]


best_route('E', 'G', 'K')


