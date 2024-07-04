# Asgn_P3_Mainuddin.py
# Coding solution of Assignment P3
# Submitted by: Mainuddin Alam Irteja (A00446752)
# Original File Provided by: Professor Fatemeh Bayeh
# Original comments are kept intact
# You need to install gymnasium and pygame
# References: 
# https://github.com/Farama-Foundation/Gymnasium
# https://github.com/danpisq/Value-and-Policy-iteration-OpenAI-FrozenLake

import gymnasium as gym
import numpy as np

def Value_Iteration(rows, columns, givenActions, givenEnv, tolerance, discountVal):
    """
    Function to implement value iteration.

    Args:
        rows: The number of rows
        columns: The number of columns
        givenActions: The dictionary with the actions
        givenEnv: The environment of FrozenLake-v1 of the gymnasium module
        tolerance: The stopping tolerance for this algorithm
        discountVal: The discounted value provided by the user
    Returns:
        stateValues: The numpy array with the optimum values of each state 
        iterationCounter: The number of iterations required to stop the while loop
    """
    # Get the number of states
    numStates = rows * columns
    # Create numpy array to rerpesent state values and initalize all elements to 0
    stateValues = np.zeros(numStates, dtype = float)
    # Initialize the iteration counter
    iterationCounter = 0
    # Initiate infinite loop
    while True:
        # Increment the iteration counter
        iterationCounter += 1
        # Set checkConvergence to zero
        checkConvergence = 0
        # Keep track of the previous values of the states
        prevStateValues = np.copy(stateValues)
        # Loop through each state
        for state in range(0, numStates):
            # Initialize qValues to empty
            qValues = []
            # Loop through the actions
            for action in givenActions:
                # Get the value of the action
                actionVal = givenActions[action]
                # Initialize updateVal
                updateVal = 0
                # Get the transtions by taking a state with it's
                # corresponding action
                transitions = givenEnv.P[state][actionVal]
                # Loop through the transitions
                for probability, nextState, reward, isDone in transitions:
                    # Update the value
                    updateVal += probability * (reward + (discountVal * prevStateValues[nextState]))
                # Add the updated value to qValues
                qValues.append(updateVal)
            # Get the optimum value of a state 
            stateValues[state] = max(qValues)
            # Get th new value of checkConvergence
            checkConvergence = np.sum(np.fabs(np.subtract(prevStateValues, stateValues)))
        # Break the infinite while loop if checkConvergence is less than tolerance
        if checkConvergence < tolerance:
            break
    # Return the optimum values of the states and the number of iterations of the while loop
    return stateValues, iterationCounter

def Policy_Evaluation(rows, columns, givenPolicies, givenEnv, tolerance, discountVal):
    """
    Function to implement policy evaluation.

    Args:
        rows: The number of rows
        columns: The number of columns
        givenPolicies: The numpy array with the policies
        givenEnv: The environment of FrozenLake-v1 of the gymnasium module
        tolerance: The stopping tolerance for this algorithm
        discountVal: The discounted value provided by the user
    Returns:
        stateValues: The numpy array with the optimum values of each state 
    """
    # Get the number of states
    numStates = rows * columns
    # Create numpy array to rerpesent state values and initalize all elements to 0
    stateValues = np.zeros(numStates, dtype = float)
    while True:
        # Set checkConvergence to zero
        checkConvergence = 0
        # Keep track of the previous values of the states
        prevStateValues = np.copy(stateValues)
        # Loop through each state
        for state in range(0, numStates):
            # Get the value of the action
            actionVal = givenPolicies[state]
            # Initialize updateVal
            updateVal = 0
            # Get the transtions by taking a state with it's
            # corresponding action
            transitions = givenEnv.P[state][actionVal]
            for probability, nextState, reward, isDone in transitions:
                # Update the value
                updateVal += probability * (reward + (discountVal * prevStateValues[nextState]))
            # Get the optimum value of the state
            stateValues[state] = updateVal
        # Get the new value of checkConvergence
        checkConvergence = np.sum(np.fabs(np.subtract(prevStateValues, stateValues)))
        # Break the infinite while loop if checkConvergence is less than tolerance
        if checkConvergence < tolerance:
            break
    # Return the optimum values of the states
    return stateValues
              
def Policy_Extraction(rows, columns, givenActions, optimumValues, givenEnv, discountVal):
    """
    Function to implement policy extraction.

    Args:
        rows: The number of rows
        columns: The number of columns
        givenActions: The dictionary with the actions
        optimumValues: The numpy array with the optimum values of the states
        givenEnv: The environment of FrozenLake-v1 of the gymnasium module
        discountVal: The discounted value provided by the user
    Returns:
        optimumPolicies: The numpy array with the optimum policies of each state 
    """
    # Get the number of states
    numStates = rows * columns
     # Create numpy array to rerpesent optimum policies and initalize all elements to 0
    optimumPolicies = np.zeros(numStates, dtype = int)
    # Loop through each state
    for state in range(0, numStates):
        # Initialize qValues to empty
        qValues = []
        # Loop through the actions
        for action in givenActions:
            # Get the value of the action
            actionVal = givenActions[action]
            # Initialize updateVal
            updateVal = 0
            # Get the transtions by taking a state with it's
            # corresponding action
            transitions = givenEnv.P[state][actionVal]
            # Loop through the transitions
            for probability, nextState, reward, isDone in transitions:
                # Update the value
                updateVal += probability * (reward + (discountVal * optimumValues[nextState]))
             # Add the updated value to qValues
            qValues.append(updateVal)
        # Convert qValues from a list to a numpy array 
        qValues = np.array(qValues)
        # Get the optimum action
        optimumAction = np.argmax(qValues)
        # Add the optimum action to the optimumPolicies dictionary
        optimumPolicies[state] = optimumAction
    # Return the optimum policies for each state
    return optimumPolicies            
        
def Policy_Iteration(rows, columns, givenActions, givenEnv, tolerance, discountVal):
    """
    Function to implement policy iteration.

    Args:
        rows: The number of rows
        columns: The number of columns
        givenActions: The dictionary with the actions
        givenEnv: The environment of FrozenLake-v1 of the gymnasium module
        tolerance: The stopping tolerance for this algorithm
        discountVal: The discounted value provided by the user
    Returns:
        stateValues: The numpy array with the optimum values of each state
        optimumPolicies: The numpy array with the optimum policies of each state 
        iterationsCounter: The number of iterations required to stop the while loop
    """
    # Get the number of states
    numStates = rows * columns
    # Create numpy array to rerpesent state values and initalize all elements to 0
    stateValues = np.zeros(numStates, dtype = float)
    # Get the number of actions
    numActions = len(givenActions)
    # Create numpy array to represent optimum policies
    # Initalize all elements randomly to form a random policy
    optimumPolicies = np.random.choice(numActions, size = numStates)
    # Initialize iteration counter
    iterationCounter = 0
    # Initiate infinite loop
    while True:
        # Increment the iteration counter
        iterationCounter += 1
        # Boolean value to check if the two arrays are same
        isChanged = True
        # Get the current optimum values using policy evaluation
        currentOptimumValues = Policy_Evaluation(rows, columns, optimumPolicies, givenEnv, tolerance, discountVal)
        # Update the policies using policy extraction
        updatedPolicies = Policy_Extraction(rows, columns, givenActions, currentOptimumValues, givenEnv, discountVal)
        # Update stateValues with currentOptimumValues
        stateValues = np.copy(currentOptimumValues)
        # Check if policies and updated policies are the same
        if (np.all(optimumPolicies == updatedPolicies)):
            # Set isChanged to false
            isChanged = False
        # If isChanged is False, exit the while loop
        if isChanged == False:
            break
        # Set optimumPolicies to updatedPolicies
        optimumPolicies = updatedPolicies 
    # Return the optimum values of each state, the optimum policies for each state 
    # and the number of iterations for the while loop
    return stateValues, optimumPolicies, iterationCounter

def Display_Results(stateValues, givenPolicies, numIter):
    """
    Function to display results.

    Args:
        stateValues: The optimum values of each states
        givenPolicies: The optimum policies of each state
        numIter: The number of iterations required
    """
    # Display the number of iterations to the user
    print(f"\nThe number of iterations was: {numIter}")
    print("\nState\t\tValue of State\t\tPolicy for State")
    #Loop through the stateVals
    for i in range(0, len(stateVals)):
        # Display the state, the optimum value of the state and the optimum policy
        # of each state
        print(f"{i}\t\t{stateValues[i]:.5f}\t\t\t{givenPolicies[i]}")
    print("")

# Initialize worldType to None
worldType = None
# Prompt user to enter the world type
print("\nWhat kind of world do you want to run the program with ?")
checkWorldType = input("Enter 'determinstic' or 'stochastic': ")
# Set worldType to false if checkWorldType is deterministic
if (checkWorldType == "deterministic"):
    worldType = False
# Set checkWorldType to true if checkWorldType is stochastic
if (checkWorldType == "stochastic"):
    worldType = True

# Prompt user to get discount or gamma value
discount = float(input("\nEnter the discount (gamma) value: "))
# Prompt user to enter the iteration type
print("\nDo you want to use value iteration or policy iteration ?")
checkIterationType = input("Enter 'value' or 'policy': ")

# You need this part
# S:Start, F:Frozen, H:Hole, G:Goal
map = ["SFFF", "FHFH", "FFFF", "HFFG"]
# is_slippery=True means stochastic and is_slippery=False means deterministic
env = gym.make('FrozenLake-v1', render_mode="human", desc=map, map_name="4x4", is_slippery=worldType)
env.reset()
env.render()

# You need to find the policy using both value iteration and policy iteration
# You may not need this part!
action = ["left", "down", "right", "up"]
ncols = 4
nrows = 4
e = 0.001

# Create the actions dictionary
actionsDictionary = {}
for a in action:
    if (a == "left"):
        actionsDictionary[a] = 0
    elif (a == "down"):
        actionsDictionary[a] = 1
    elif (a == "right"):
        actionsDictionary[a] = 2
    else:
        actionsDictionary[a] = 3

# A sample policy to make the following while loop works
policy = [1, 2, 1, 0, 1, 0, 1, 0, 2, 1, 1, 1, 0, 2, 2, 0]

# Implement value iteration if checkIterationType is equal to value
if (checkIterationType == "value"):
    # Get the optimum values of the states and the number of iterations
    stateVals, countIterations = Value_Iteration(nrows, ncols, actionsDictionary, env, e, discount)
    # Get the optimum policies of each state
    policy = Policy_Extraction(nrows, ncols, actionsDictionary, stateVals, env, discount)
    # Display the results to the user
    Display_Results(stateVals, policy, countIterations)

# Implement value iteration if checkIterationType is equal to policy  
if (checkIterationType == "policy"):
    # Get the optimum values of the states, the optimum policies of the states and
    # the number of iterations
    stateVals, policy, countIterations = Policy_Iteration(nrows, ncols, actionsDictionary, env, e, discount)
    # Display the results to the user
    Display_Results(stateVals, policy, countIterations)

# This part uses the found policy to interact with the environment.
# You don't need to change anything here.
s = 0
goal = ncols * nrows - 1
while s != goal:
    a = policy[s]
    s, r, t, f, p = env.step(a)
    if t == True and s != goal:
        env.reset()
        s = 0

print("end")
