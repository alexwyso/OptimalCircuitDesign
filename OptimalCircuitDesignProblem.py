# Alex Wysoczanski (alexjwyso@gmail.com)
# University of Pennsylvania 

# Copyright 2020, Alex Wysoczanski, All rights reserved.

# This project utilizes the Constraint Programming library from Google OR-Tools.
from ortools.sat.python import cp_model
import math
import collections
from timeit import default_timer as timer

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                EQUIVALENT RESISTANCE CALCULATION METHOD
                        LIST NOTATION VARIATION

    Takes a valid representation of a list representation of a circuit 
    in Reverse Polish Notation and calculates its equivalent resistance
    in O(n) where n is the number of resistors in the set.

    Reverse Polish Notation negates the need for parentheses by placing
    operators directly after their operands, drastically simplifying
    circuit representation. 

    INPUT: circuit ->   the list of numbers corresponding to 
                        resistors and series/parallel operators
    OUTPUT: (float) the equivalent resistance of the circuit

    [0] = An unused index, for circuits less than the maximum length
    [-1] =  Operator for a series connection between two resistors
    [-2] = Operator for a parallel connection between two resistors
    [i] = Operand representing a resistor with resistance i

    *Each operator turns two operands into one, so the length of a 
    solution is (2n - 1)

    **Operators will always be integers, but operands may be floats. Note
    that both of the operators in this case are commutative, so many resistors
    in series or parallel can be represented by chains of operations

    ***This implementation was only used for testing the accuracy of the 
    solver as the functionality had to be redone in terms of linear 
    constraints for the CP Model. It also provides a more clear understanding 
    of the notation used for circuit representation.

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def getEquivalentResistance(circuit: [float]) -> float:
    
    # Handle a base case of an empty circuit
    if len(circuit) == 0:
        return 0

    # Handle a base case of a single resistor circuit
    if len(circuit) == 1:
        return circuit[0]

    # Initialize a stack and iterate over the circuit list
    stack = []
    for n in circuit:

        # Handle the empty space (0) by terminating the calculation and
        # returning the top of the stack, which should contain only 1 element
        if n == 0:
            return stack[0]

        # Handle the series operator (-1) by taking the two top elements of
        # the stack, adding them, and pushing the result to the stack
        if n == -1:
            r1 = stack.pop()
            r2 = stack.pop()
            stack.append(r1 + r2)

        # Handle the case of a parallel operatore (-x) by taking the x top
        # elements of the stack, adding their inverses, and pushing the 
        # inverse of this sum to the stack
        elif n == -2:
            r1 = stack.pop()
            r2 = stack.pop()
            stack.append(1 / ((1 / r1) + (1 / r2)))
        
        # Push any operand resistor values to the stack
        else:
            stack.append(float(n))

    # With the assumption that the list was well formed in RPN, when
    # the loop terminates, the stack will contain a single value
    # corresponding to the equivalent resistance of the circuit
    return stack[0]

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                    THE OPTIMAL CIRCUIT DESIGN PROBLEM:

    A solver class for the Optimal Circuit Design Problem, in which the user
    seeks the arrangement of a set of resistors into a series-parallel circuit 
    that yields an equivalent resistance closest to a target value.

    INPUT:  target      -> the desired equivalent resistance
            resistors   -> the set of available resistors

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class OptimalCircuitDesignProblem():

    # ------------------------ CONSTRUCTOR -------------------------------- #
    # Initializes the class based on the given information.
    def __init__(self, target: float, resistors: [float]):

        # The desired equivalent resistance
        self.target = target

        # Information based on the set of resistors given
        self.resistors = resistors
        self.numResistors = len(resistors)
        self.maxResistor = max(resistors)
        self.maxLengthCircuit = 2 * self.numResistors - 1

    # ------------------------ CREATE VARIABLES --------------------------- #
    # Establishes variables that store the state of the model in regards
    # to the solution list and lists that correspond to which elements
    # contain operators, series operators, parallel operators, resistors, 
    # or empty space.
    def create_variables(self):
        model: cp_model.CpModel = self.model
        maxLengthCircuit = self.maxLengthCircuit
        maxResistor = self.maxResistor

        # Initialize the solution format to be size 'maxLengthCircuit' with
        # entries ranging from -2 to 'maxResistor'
        solution = {}
        for i in range(maxLengthCircuit):
            solution[i] = model.NewIntVar(-2, maxResistor, f'solution {i}')
        self.solution = solution

        # Enforce that if an element of the solution is 0 that it is empty
        isEmpty = {}
        for i in range(maxLengthCircuit):
            isEmpty[i] = model.NewBoolVar(f'{i} is 0')
            model.Add(solution[i] == 0).OnlyEnforceIf(isEmpty[i])
            model.Add(solution[i] != 0).OnlyEnforceIf(isEmpty[i].Not())
        self.isEmpty = isEmpty
        
        # Enforce that if an element of the solution is negatibe that it is 
        # an operator
        isOp = {}
        for i in range(maxLengthCircuit):
            isOp[i] = model.NewBoolVar(f'{i} is an operator')
            model.Add(solution[i] < 0).OnlyEnforceIf(isOp[i])
            model.Add(solution[i] >= 0).OnlyEnforceIf(isOp[i].Not())
        self.isOp = isOp

        # Enforce that if an element of the solution is -1 that it is a
        # series operator
        isSeries = {}
        for i in range(maxLengthCircuit):
            isSeries[i] = model.NewBoolVar(f'{i} is a series operator')
            model.Add(solution[i] == -1).OnlyEnforceIf(isSeries[i])
            model.Add(solution[i] != -1).OnlyEnforceIf(isSeries[i].Not())
        self.isSeries = isSeries

        # Enforce that if an element of the solution is -1 that it is a
        # parallel operator
        isParallel = {}
        for i in range(maxLengthCircuit):
            isParallel[i] = model.NewBoolVar(f'{i} is a parallel operator')
            model.Add(solution[i] == -2).OnlyEnforceIf(isParallel[i])
            model.Add(solution[i] != -2).OnlyEnforceIf(isParallel[i].Not())
        self.isParallel = isParallel

        # Enforce that if an element of the solution is positive that it is
        # a resistor
        isResistor = {}
        for i in range(maxLengthCircuit):
            isResistor[i] = model.NewBoolVar(f'{i} is a resistor')
            model.Add(solution[i] > 0).OnlyEnforceIf(isResistor[i])
            model.Add(solution[i] <= 0).OnlyEnforceIf(isResistor[i].Not())
        self.isResistor = isResistor

    # ------------------- ADD INITIAL STACK CONSTRAINT -------------------- # 
    # Adds constraints that help ensure the solutions are in well-formed 
    # Reverse Polish Notation.     
    def addInitialStackConstraint(self):
        model: cp_model.CpModel = self.model
        solution = self.solution
        maxLengthCircuit = self.maxLengthCircuit

        # The first entry must be a resistor
        if maxLengthCircuit >= 1:
            model.Add(solution[0] > 0)
        
        # The second entry can be a resistor or an empty space
        if maxLengthCircuit >= 2:
            model.Add(solution[1] >= 0)

        # Elements less than -3 are not allowed, as they do not have any 
        # meaning in the encoding
        for i in range(maxLengthCircuit):
            model.Add(solution[i] >= -2)

    # ------------------- ADD RESISTOR SET CONSTRAINT --------------------- #      
    # Adds constraints to ensure the every value in the solution comes from
    # the input set or corresponds to an operator or empty space.
    def addResistorSetConstraint(self):
        model: cp_model.CpModel = self.model
        solution = self.solution
        resistors = self.resistors
        maxLengthCircuit = self.maxLengthCircuit

        # Make a set of all operators and all input resistor values
        resistorsListOfList = [[i] for i in resistors]
        resistorsListOfList.append([0])
        resistorsListOfList.append([-1])
        resistorsListOfList.append([-2])

        # Ensure each element of the solution comes from this list
        for i in range(maxLengthCircuit):
            model.AddAllowedAssignments([solution[i]], resistorsListOfList)

    # ----------------------- ADD LENGTH CONSTRAINT ----------------------- #     
    # Adds constraints so that a circuit representation of length n
    # fills the first n indices of the solution with resistors/operators 
    # and forces the rest to be 0 for unused space.
    def addLengthConstraint(self):
        model: cp_model.CpModel = self.model
        maxLengthCircuit = self.maxLengthCircuit
        isEmpty = self.isEmpty
        solution = self.solution

        for i in range(1, maxLengthCircuit):
            model.Add(solution[i] == 0).OnlyEnforceIf(isEmpty[i - 1])
        return

    # --------------------- ADD SINGLE USE CONSTRAINT --------------------- #   
    # Adds constraints to ensure each resistor is not used in the solution
    # more times than it appears in the input set.
    def addSingleUseConstraint(self):
        model: cp_model.CpModel = self.model
        maxLengthCircuit = self.maxLengthCircuit
        resistors = self.resistors
        solution = self.solution

        isMatch = {}
        for i in resistors:
            for j in range(maxLengthCircuit):
                isMatch[i, j] = model.NewBoolVar(f'resistor {i} is at index {j} ')
                model.Add(solution[j] == i).OnlyEnforceIf(isMatch[i, j])
                model.Add(solution[j] != i).OnlyEnforceIf(isMatch[i, j].Not())

        for i in resistors:
            model.Add(sum(isMatch[i, j] for j in range(maxLengthCircuit)) <= resistors.count(i))

    # ------------------ ADD VALID OPERATORS CONSTRAINT ------------------- #   
    # Adds constraints to ensure each operator is preceeded by the correct 
    # number of operators and operands, i.e. the Reverse Polish Notation has
    # well formed operators.
    def addValidOperatorsConstraint(self):
        model: cp_model.CpModel = self.model
        maxLengthCircuit = self.maxLengthCircuit
        numResistors = self.numResistors
        isEmpty = self.isEmpty
        isResistor = self.isResistor 
        isSeries = self.isSeries
        isParallel = self.isParallel

        # Since Reverse Polish Notation is evalutated based on a stack,
        # it must be ensured that an empty stack is never popped, and that
        # the stack has one element in it when the calculation terminates.
        # Keep track of a new counter after each new element is encountered to
        # ensure these constraints hold throughout the solution
        counter = {}

        # The first counter will equal 1, as the first element must correspond
        # to a resistor so there will be 1 element in the stack
        counter[0] = model.NewIntVar(1, 1, f'counter {0}')

        # Builds the other constrains based on what type of element was
        # last encountered       
        for i in range(1, maxLengthCircuit):

            # The variable storing the size of the stack by the time element
            # i is reached
            counter[i] = model.NewIntVar(1, numResistors, f'counter {i}')

            # Add 1 to the previous counter when i is a resistor value
            model.Add(
                counter[i] == counter[i - 1] + 1).OnlyEnforceIf(isResistor[i])
            
            # Subtract one for operator values, as they take two elements
            # off the stack and add one back
            model.Add(
                counter[i] == counter[i - 1] - 1).OnlyEnforceIf(isSeries[i])
            model.Add(
                counter[i] == counter[i - 1] - 1).OnlyEnforceIf(isParallel[i])

            # Maintain the same counter for unused space
            model.Add(counter[i] == counter[i - 1]).OnlyEnforceIf(isEmpty[i])

        # Ensure the final counter is 1, meaning the circuit representation 
        # can be evaluated to a single equivalent resistance without errors
        model.Add(counter[maxLengthCircuit - 1] == 1)

    # -------------------------- MINIMIZE ERROR --------------------------- #     
    # Adds constraints to represent the equivalent resistance calcaultation
    # of the circuit representation, and then minimize the difference between
    # this value and the target value.
    def minimizeError(self):
        model: cp_model.CpModel = self.model
        maxLengthCircuit = self.maxLengthCircuit
        maxResistor = self.maxResistor
        numResistors = self.numResistors
        solution = self.solution
        isEmpty = self.isEmpty
        isResistor = self.isResistor 
        isSeries = self.isSeries
        isParallel = self.isParallel
        target = self.target

        # These constraints will be formed by making new variables that 
        # keep track of what the stack looks like after each element 
        stack = {}

        # The stack at the first element will hold the first element of the
        # solution, as it must be a resistor.
        # NOTE: Resistor values are scaled by a factor of 1000 before being
        #       added to the stack because OR-Tools only works with integers,
        #       so the scale factor makes the model accurate to 0.001 Ohms
        stack[0] = model.NewIntVar(
            0, 1000 * maxResistor, f'stack state {0} entry {0}')
        model.Add(stack[0] == 1000 * solution[0])

        # The max size of each stack grows by 1 with each index, so the 
        # total lenght is calculated by summing the numbers from 1 to 
        # maxLengthCircuit
        stackLength = sum([x for x in range(1, maxLengthCircuit + 1)])
        self.stackLength = stackLength

        # Create variables to store the state of the stacks in one flat array
        # such that the ith element in the circuit is allocated i+1 storage 
        # space with the next stack coming right after. This makes the stack
        # structure manageable even with only linear constraints
        index = 1
        for circuitElement in range(1, maxLengthCircuit):
            for stackEntry in range (0, circuitElement + 1):
                stack[index] = model.NewIntVar(
                    0, 1000 * maxResistor * numResistors, f'stack state {circuitElement} entry {stackEntry}')
                index = index + 1

        # EMPTY SPACES #
        # Copy the stack state over from the previous index with an additional
        # blank space at the beginning
        index = 1
        for circuitElement in range(1, maxLengthCircuit):
            model.Add(stack[index] == 0).OnlyEnforceIf(isEmpty[circuitElement])
            index = index + 1
            for stackEntry in range (1, circuitElement + 1):
                model.Add(
                    stack[index] == stack[index - circuitElement - 1]).OnlyEnforceIf(isEmpty[circuitElement])
                index = index + 1

        # RESISTORS #
        # Copy stack from previous index, but shifted down to make room for the
        # new resistor, then add the resistor at the top (technically left) 
        # of the current stack
        index = 1
        for circuitElement in range(1, maxLengthCircuit):
            for stackEntry in range (0, circuitElement):
                model.Add(
                    stack[index] == stack[index - circuitElement]).OnlyEnforceIf(isResistor[circuitElement])
                index = index + 1
            model.Add(
                stack[index] == 1000 * solution[circuitElement]).OnlyEnforceIf(isResistor[circuitElement])
            index = index + 1

        # SERIES OPERATORS #
        # Copy stack from previous index, but shifted down by two elements. The
        # first is to allocate more space to a larger stack, and the second is
        # because two elements of the stack are removed, and only one is pushed
        # back. Get the sum of the top two elements of the previous stack and
        # push it to the current stack
        index = 3
        for circuitElement in range(2, maxLengthCircuit):
            model.Add(stack[index] == 0).OnlyEnforceIf(isSeries[circuitElement])
            index = index + 1
            model.Add(stack[index] == 0).OnlyEnforceIf(isSeries[circuitElement])
            index = index + 1
            for stackEntry in range (2, circuitElement):
                model.Add(
                    stack[index] == stack[index - circuitElement - 2]).OnlyEnforceIf(isSeries[circuitElement])
                index = index + 1
            model.Add(
                stack[index] == ((
                    stack[index - circuitElement - 2] + stack[index - circuitElement - 1]))).OnlyEnforceIf(isSeries[circuitElement])
            index = index + 1

        # PARALLEL OPERATORS #
        # The same logic applies for parallel operators, but the inverse of the 
        # sum of the inverses is pushed instead through the identity 
        # ab / (a+b) = 1 / ((1 / a) + (1 / b))
        index = 3
        for circuitElement in range(2, maxLengthCircuit):
            model.Add(stack[index] == 0).OnlyEnforceIf(isParallel[circuitElement])
            index = index + 1
            model.Add(stack[index] == 0).OnlyEnforceIf(isParallel[circuitElement])
            index = index + 1
            for stackEntry in range (2, circuitElement):
                model.Add(
                    stack[index] == stack[index - circuitElement - 2]).OnlyEnforceIf(isParallel[circuitElement])
                index = index + 1

            sumAB = model.NewIntVar(1, numResistors * maxResistor * 1000, 'sum')
            prodAB = model.NewIntVar(0, (maxResistor * numResistors * 1000 * numResistors * maxResistor * 1000), 'prod')
            divAB = model.NewIntVar(0, (maxResistor * numResistors * 1000 * numResistors * maxResistor * 1000), 'div')
            model.Add(sumAB == stack[index - circuitElement - 2] + stack[index - circuitElement - 1])
            model.AddMultiplicationEquality(prodAB, [stack[index - circuitElement - 2], stack[index - circuitElement - 1]])
            model.AddDivisionEquality(divAB, prodAB, sumAB)
            model.Add(stack[index] == divAB).OnlyEnforceIf(isParallel[circuitElement])
            index = index + 1
        
        # Minimize the difference between the target and equivalent resistance
        # of the circuit, which is now stored in the variable at the top of the
        # final stack. If there is a tie, choose the smaller solution
        self.stack = stack
        numNonZeros = model.NewIntVar(0, maxLengthCircuit * 2, "numNonZeros")
        model.Add(numNonZeros == sum(isEmpty[i].Not() for i in range(maxLengthCircuit)))
        error = model.NewIntVar(-1 * maxResistor * numResistors, maxResistor * numResistors, 'error')
        model.Add(error == (1000 * target) - stack[stackLength - 1])
        errorAbs = model.NewIntVar(0, 1000 * maxResistor * numResistors * maxLengthCircuit, 'error')
        model.AddAbsEquality(errorAbs, error)
        self.model.Minimize(maxLengthCircuit * errorAbs + numNonZeros)

    # -------------------------- BREAK SYMMETRY --------------------------- #          
    # Since series and parallel operators are both commutative, the run time 
    # can be drastically cut down by eliminating topologically equivalent 
    # circuits from the potential list of solutions.
    def breakSymmetry(self):
        model = self.model
        solution = self.solution
        maxLengthCircuit = self.maxLengthCircuit
        isOp = self.isOp

        # Adds constraints to force the second half of the solution to consist
        # of only operators or empty spaces so that 
        for i in range(maxLengthCircuit):
            if i > maxLengthCircuit // 2:
                model.Add(solution[i] <= 0)
        
        # Similarly, ensure that all of the resistors are at the front and
        # once an operator is used, block other resistors from following
        for i in range(1, maxLengthCircuit):
            model.Add(solution[i] <= 0).OnlyEnforceIf(isOp[i - 1])        
        
    # ------------------------------- SOLVE ------------------------------- #        
    # Runs an instance of the solver, combining all the constraints and 
    # finding the optimal solution.
    def solve(self) -> [float]:

        # Create a new model and solver
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        maxLengthCircuit = self.maxLengthCircuit
        model = self.model
        solver = self.solver
        numResistors = self.numResistors

        # Handle base cases for resistor set inputs pf size 0, 1, or 2
        if (numResistors == 0):
            return []
        elif (numResistors == 1):
            return [resistors[0]]
        elif (numResistors == 2):
            seriesRes = abs(target - resistors[0] - resistors[1])
            parallelRes = abs(target - (1 / ((1 / resistors[0]) + (1 / resistors[1]))))
            if seriesRes < parallelRes:
                return [resistors[0], resistors[1], -1]
            else:
                return [resistors[0], resistors[1], -2]

        # Add all of the previously defined constraints
        self.create_variables()
        self.addInitialStackConstraint()
        self.addResistorSetConstraint()
        self.addLengthConstraint()
        self.addSingleUseConstraint()
        self.addValidOperatorsConstraint()
        self.minimizeError()
        self.breakSymmetry()
        solution = self.solution

        # Output the solution
        output = []
        if (solver.Solve(model) == cp_model.OPTIMAL) | (
            solver.Solve(model) == cp_model.FEASIBLE):
            for i in range(maxLengthCircuit):
                output.append(solver.Value(solution[i]))
            return [ x for x in output if x != 0 ]
        else:
            # This should theoretically never happen, as none of the constraints
            # should result in an unfeasible solution. At worst it would just
            # find a poor solution. Please contact me if you ever hit this
            # message.
            return []

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                            TESTING:

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''' 
# ------------------ TEST GET EQUIVALENT RESISTANCE ----------------- #            
# Run a sample of tests to verify the correctness of the algorithm
# when using a list notation
def testGetEquivalentResistance():

    # TEST SINGLE
    assert abs((50 - getEquivalentResistance([50]))) < 0.0001

    # TEST SIMPLE SERIES -> S(5, (P(4, 3))
    assert abs((6.71428 - getEquivalentResistance([3, 4, -2, 5, -1]))) < 0.0001

    # TEST STRING OF SERIES -> S(50, S(50, 50))
    assert abs((175 - getEquivalentResistance([50, 50, -1, 75, -1]))) < 0.0001

    # TEST SIMPLE SERIES-PARALLEL -> P(S(100, 200), 50)
    assert abs((42.85714 - getEquivalentResistance(
        [100, 200, -1, 50, -2]))) < 0.0001

    # TEST COMPLEX SERIES-PARALLEL -> P(P(50, S(100, 200)), 75, S(50, 50, 75))
    assert abs((23.59551 - getEquivalentResistance(
        [100, 200, -1, 50, -2, 75, 50, 50, -1, 75, -1, -2, -2]))) < 0.0001

    return

# --------------------------- GET RESULTS ------------------------- #            
# Run an instance of the sovler and outputs useful information on the results
def getResults(target: int, resistors: [float]):

    t = timer()
    solver = OptimalCircuitDesignProblem(target, resistors)
    solution = solver.solve()
    print("RESISTOR SET: " + str(resistors))
    print("TARGET: " + str(target))
    print("BEST: " + str(solution))
    print("CLOSEST RESISTANCE: " + str(
        round(getEquivalentResistance(solution), 4)))
    print("ERROR: " + str(round(100 * abs((
        target - getEquivalentResistance(solution)) / target), 4)) + "%")
    print(f'Solved in {timer() - t} sec.')
    print()
    return solution

# TEST: List notation calculations
testGetEquivalentResistance()

# TEST: Single resistor
target = 400
resistors = [200]
assert getResults(target, resistors) == [200]

# TEST: Two unique resistors, series best
target = 400
resistors = [200, 100]
assert getResults(target, resistors) == [200, 100, -1]

# TEST: Three of the same resistor
target = 600
resistors = [200, 200, 200]
getResults(target, resistors) == [200, 200, 200, -1, -1]

# TEST: Standard set of 5 unique resistors with complex solution
target = 158
resistors = [100, 200, 300, 400, 500]
getResults(target, resistors) == [400, 100, 200, 300, 500, -1, -2, -1, -2]

# TEST: Standard set of 6 unique resistors with complex solution
target = 439
resistors = [100, 200, 300, 400, 500, 600]
getResults(target, resistors) == [200, 300, 600, 500, 400, 100, -2, -1, -1, -2, -1]

# TEST: Standard set of 7 unique resistors with simple solution
target = 1000
resistors = [100, 200, 300, 400, 500, 600, 700]
getResults(target, resistors) == [600, 400, 500, 200, 300, 800, 100, -2, -1, -2, -1, -2, -1]

# TEST: Complex set of 7 unique resistors with complex solution
target = 999
resistors = [4, 67, 840, 189, 223, 319, 21]
getResults(target, resistors) == [600, 400, 500, 200, 300, 800, 100, -2, -1, -2, -1, -2, -1]

print("DONE")
