# Alex Wysoczanski (alexjwyso@gmail.com)
# University of Pennsylvania 

# Copyright 2020, Alex Wysoczanski, All rights reserved.

# This project utilizes the Constraint Programming library from Google OR-Tools
from ortools.sat.python import cp_model
import math
import collections

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            EQUIVALENT RESISTANCE CALCULATOR ALGORITHM

    Takes a valid representation of a circuit in Reverse Polish Notation
    and calculates its equivalent resistance in O(n) where n is the 
    number of resistors in the set.

    Reverse Polish Notation negates the need for parentheses by placing
    operators directly after their operands, drastically simplifying
    circuit representation. 

    INPUT: circuit ->   the list of numbers corresponding to 
                        resistors and series/parallel operators
    OUTPUT: (float) the equivalent resistance of the circuit

    [0] = An unused index, for circuits less than the max length
    [-1] =  Operator for a series connection between two resistors
    [-i] = Operator for a parallel connection between i resistors
    [i] = Operand representing a resistor with resistance i

    *Each operator turns at least two operands into one,
    so the maximum length of a solution is (2n - 1)

    **Operators will always be integers, but operands may be floats

    ***This implementation was only used for testing, as the function
    had to be redone in terms of linear constraints for the CP Model,
    but it is proof of the underlying logic.

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
            stack.append(r1+r2)

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

# Run a sample of tests to verify the correctness of the algorithm
def testGetEquivalentResistance():

    # TEST 50 ~= 50
    assert abs((50 - getEquivalentResistance([50]))) < 0.0001

    # TEST SIMPLE SERIES -> S(5, (P(4, 3))
    assert abs((6.71428 - getEquivalentResistance([3, 4, -2, 5, -1]))) < 0.0001

    # TEST STRING OF SERIES -> S(50, S(50, 50))
    assert abs((175 - getEquivalentResistance([50, 50, -1, 75, -1]))) < 0.0001

    # TEST SIMPLE SERIES-PARALLEL -> P(S(100, 200), 50)
    assert abs((42.85714 - getEquivalentResistance([100, 200, -1, 50, -2]))) < 0.0001

    # TEST COMPLEX SERIES-PARALLEL -> P(P(50, S(100, 200)), 75, S(50, 50, 75))
    assert abs((23.59551 - getEquivalentResistance([100, 200, -1, 50, -2, 75, 50, 50, -1, 75, -1, -2, -2]))) < 0.0001

    return

class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.solutionsSet = []

    def on_solution_callback(self):
        self.__solution_count += 1
        solution = []
        for v in self.__variables:
            solution.append(self.Value(v))
        self.solutionsSet.append(tuple(solution))

    def solution_count(self):
        return len(self.finalSet)

    def getFinalSet(self) -> set():
        finalSet = set(self.solutionsSet)
        self.finalSet = finalSet
        return finalSet

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                THE OPTIMAL CIRCUIT DESIGN PROBLEM:

 A solver class for the Optimal Circuit Problem, in which the user seeks
 the arrangement of a series-parallel circuit that yields an
 equivalent resistance closest to a target value.

 INPUT: target      -> the desired equivalent resistance
        resistors   -> the set of available resistors

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class OptimalCircuitDesignProblem():

    # Initializes the calculator class inputs
    def __init__(self, target: float, resistors: [float]):

        # The desired equivalent resistance
        self.target = target

        # Information on the set of resistors given
        self.resistors = resistors
        self.numResistors = len(resistors)
        self.maxResistor = max(resistors)
        self.maxLengthCircuit = 2 * self.numResistors - 1

    # Establishes the solution list variable for the model
    def create_variables(self):
        model: cp_model.CpModel = self.model
        numResistors = self.numResistors
        maxLengthCircuit = self.maxLengthCircuit
        maxResistor = self.maxResistor

        solution = {}
        for i in range(maxLengthCircuit):
            solution[i] = model.NewIntVar(-2, maxResistor, ' ')
        self.solution = solution

        isEmpty = {}
        for i in range(maxLengthCircuit):
            isEmpty[i] = model.NewBoolVar(f'{i} is 0')
            model.Add(self.solution[i] == 0).OnlyEnforceIf(isEmpty[i])
            model.Add(self.solution[i] != 0).OnlyEnforceIf(isEmpty[i].Not())
        self.isEmpty = isEmpty
        
        isOp = {}
        for i in range(maxLengthCircuit):
            isOp[i] = model.NewBoolVar(f'{i} is an operator')
            model.Add(self.solution[i] < 0).OnlyEnforceIf(isOp[i])
            model.Add(self.solution[i] >= 0).OnlyEnforceIf(isOp[i].Not())
        self.isOp = isOp

        isSeries = {}
        for i in range(maxLengthCircuit):
            isSeries[i] = model.NewBoolVar(f'{i} is a series operator')
            model.Add(self.solution[i] == -1).OnlyEnforceIf(isSeries[i])
            model.Add(self.solution[i] != -1).OnlyEnforceIf(isSeries[i].Not())
        self.isSeries = isSeries

        isParallel = {}
        for i in range(maxLengthCircuit):
            isParallel[i] = model.NewBoolVar(f'{i} is a parallel operator')
            model.Add(self.solution[i] == -2).OnlyEnforceIf(isParallel[i])
            model.Add(self.solution[i] != -2).OnlyEnforceIf(isParallel[i].Not())
        self.isParallel = isParallel

        isResistor = {}
        for i in range(maxLengthCircuit):
            isResistor[i] = model.NewBoolVar(f'{i} is a resistor')
            model.Add(self.solution[i] > 0).OnlyEnforceIf(isResistor[i])
            model.Add(self.solution[i] <= 0).OnlyEnforceIf(isResistor[i].Not())
        self.isResistor = isResistor

    # Adds constraints to ensure the first two elements are operands
    def addInitialStackConstraint(self):
        model: cp_model.CpModel = self.model
        solution = self.solution

        if self.maxLengthCircuit >= 1:
            model.Add(solution[0] > 0)
        if self.maxLengthCircuit >= 2:
            model.Add(solution[1] >= 0)
        for i in range(self.maxLengthCircuit):
            model.Add(solution[i] >= -2)

    # Adds constraints to block the use of parallel resistors
    def addNoParallelConstraint(self):
        model: cp_model.CpModel = self.model
        solution = self.solution
        for i in range(self.maxLengthCircuit):
            model.Add(solution[i] >= -1)

    # Adds constraints to force a desired result for testing
    def addForceResultConstraint(self):
        model: cp_model.CpModel = self.model
        solution = self.solution
        model.Add(solution[0] == 300)
        model.Add(solution[1] == 200)
        model.Add(solution[2] == 100)
        model.Add(solution[3] == -1)
        model.Add(solution[4] == -1)

    # Adds constraints to ensure the every resistor value comes from the set
    def addResistorSetConstraint(self):
        model: cp_model.CpModel = self.model
        solution = self.solution
        resistors = self.resistors
        maxLengthCircuit = self.maxLengthCircuit

        resistorsListOfList = [[i] for i in resistors]
        resistorsListOfList.append([0])
        resistorsListOfList.append([-1])
        resistorsListOfList.append([-2])

        for i in range(maxLengthCircuit):
            model.AddAllowedAssignments([solution[i]], resistorsListOfList)

    # Adds constraints so that a circuit representation of length n
    # fills the first n indices of the solution with resistors/operators 
    # and designates the rest as -1 for unused space
    def addLengthConstraint(self):
        model: cp_model.CpModel = self.model
        maxLengthCircuit = self.maxLengthCircuit
        isEmpty = self.isEmpty

        for i in range(1, maxLengthCircuit):
            model.Add(self.solution[i] == 0).OnlyEnforceIf(isEmpty[i - 1])

        return

    # Adds constraints to ensure each resistor is used at most once
    # NEED TO REDO TO ALLOW MULTIPLE USES WHEN IN LIST TWICE
    def addSingleUseConstraint(self):
        model: cp_model.CpModel = self.model
        maxLengthCircuit = self.maxLengthCircuit
        isResistor = self.isResistor

        for i in range(maxLengthCircuit):
            for j in range(maxLengthCircuit):
                if (i != j):
                    model.Add(self.solution[i] != self.solution[j]).OnlyEnforceIf(isResistor[i])           

    # Adds constraints to ensure each operator is preceeded by the correct 
    # number of operators and operands, i.e. the Reverse Polish Notation has
    # well formed operators.
    def addValidOperatorsConstraint(self):
        model: cp_model.CpModel = self.model
        maxLengthCircuit = self.maxLengthCircuit
        numResistors = self.numResistors
        isEmpty = self.isEmpty
        isResistor = self.isResistor 
        isOp = self.isOp
        isSeries = self.isSeries
        isParallel = self.isParallel

        # Since Reverse Polish Notation is evalutated based on a stack,
        # it must be ensured that an empty stack is never popped, and that
        # the stack has one element in it when the calculation terminates.
        # Keep track of a counter and evaluate it after each new element
        counter = {}

        # The first counter will be 1, as the first element corresponds to a 
        # resistor
        counter[0] = model.NewIntVar(1, 1, f'counter {0}')

        # Builds the other constrains based on what type of element was
        # last encountered       
        for i in range(1, maxLengthCircuit):

            counter[i] = model.NewIntVar(1, numResistors, f'counter {i}')

            # Add 1 to the previous counter to resistor values
            model.Add(counter[i] == counter[i - 1] + 1).OnlyEnforceIf(isResistor[i])
            
            # Subtract one for operator values, as it takes two elements
            # off the stack and adds one back
            model.Add(counter[i] == counter[i - 1] - 1).OnlyEnforceIf(isSeries[i])
            model.Add(counter[i] == counter[i - 1] - 1).OnlyEnforceIf(isParallel[i])

            # Maintain the same counter for unused space
            model.Add(counter[i] == counter[i - 1]).OnlyEnforceIf(isEmpty[i])

        # Ensure the final counter is 1, meaning the circuit representation 
        # can be evaluated to a single equivalent resistance without errors
        model.Add(counter[maxLengthCircuit - 1] == 1)


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

        stack = {}
        stack[0] = model.NewIntVar(0, 1000 * maxResistor * numResistors, f'stack state {0} entry {0}')
        model.Add(stack[0] == 1000 * solution[0])

        stackLength = sum([x for x in range(1, maxLengthCircuit + 1)])
        self.stackLength = stackLength

        # Create variables to store the state of the stack
        index = 1
        for circuitElement in range(1, maxLengthCircuit):
            for stackEntry in range (0, circuitElement + 1):
                stack[index] = model.NewIntVar(0, 1000 * maxResistor * numResistors, f'stack state {circuitElement} entry {stackEntry}')
                index = index + 1

        # EMPTY SPACES #
        # Copy the stack state over from the previous index
        index = 1
        for circuitElement in range(1, maxLengthCircuit):
            model.Add(stack[index] == 0).OnlyEnforceIf(isEmpty[circuitElement])
            index = index + 1
            for stackEntry in range (1, circuitElement + 1):
                model.Add(stack[index] == stack[index - circuitElement - 1]).OnlyEnforceIf(isEmpty[circuitElement])
                index = index + 1

        # RESISTORS #
        # Copy stack from previous index but add current resistor to it 
        index = 1
        for circuitElement in range(1, maxLengthCircuit):
            for stackEntry in range (0, circuitElement):

                model.Add(stack[index] == stack[index - circuitElement]).OnlyEnforceIf(isResistor[circuitElement])
                index = index + 1

            model.Add(stack[index] == 1000 * solution[circuitElement]).OnlyEnforceIf(isResistor[circuitElement])
            index = index + 1

        # SERIES #
        # Sum the two end values of the previous stack and add their sum to
        # the new stack which was copied from the previous index up to its 
        # two end values
        index = 3
        for circuitElement in range(2, maxLengthCircuit):
            model.Add(stack[index] == 0).OnlyEnforceIf(isSeries[circuitElement])
            index = index + 1
            model.Add(stack[index] == 0).OnlyEnforceIf(isSeries[circuitElement])
            index = index + 1
            for stackEntry in range (2, circuitElement):
                model.Add(stack[index] == stack[index - circuitElement - 2]).OnlyEnforceIf(isSeries[circuitElement])
                index = index + 1

            model.Add(stack[index] == ((stack[index - circuitElement - 2] + stack[index - circuitElement - 1]))).OnlyEnforceIf(isSeries[circuitElement])
            index = index + 1

        # PARALLEL #
        # Adjust the indexes to match a parallel operator of arbitrary length
        index = 3
        for circuitElement in range(2, maxLengthCircuit):
            model.Add(stack[index] == 0).OnlyEnforceIf(isParallel[circuitElement])
            index = index + 1
            model.Add(stack[index] == 0).OnlyEnforceIf(isParallel[circuitElement])
            index = index + 1
            for stackEntry in range (2, circuitElement):
                model.Add(stack[index] == stack[index - circuitElement - 2]).OnlyEnforceIf(isParallel[circuitElement])
                index = index + 1

            sumAB = model.NewIntVar(1, numResistors * maxResistor * 1000, 'sum')
            prodAB = model.NewIntVar(0, (maxResistor * numResistors * 1000 * numResistors * maxResistor * 1000), 'prod')
            divAB = model.NewIntVar(0, (maxResistor * numResistors * 1000 * numResistors * maxResistor * 1000), 'div')

            model.Add(sumAB == stack[index - circuitElement - 2] + stack[index - circuitElement - 1])
            model.AddMultiplicationEquality(prodAB, [stack[index - circuitElement - 2], stack[index - circuitElement - 1]])
            model.AddDivisionEquality(divAB, prodAB, sumAB)
            
            model.Add(stack[index] == divAB).OnlyEnforceIf(isParallel[circuitElement])
            index = index + 1
        
        
        self.stack = stack
        error = model.NewIntVar(-1 * maxResistor * numResistors, maxResistor * numResistors, 'error')
        model.Add(error == (1000 * target) - stack[stackLength - 1])
        errorAbs = model.NewIntVar(0, 1000 * maxResistor * numResistors, 'error')
        model.AddAbsEquality(errorAbs, error)
        self.model.Minimize(errorAbs)
        

    # Runs an instance of the OR-Tools solver for the Optimal Circuit Problem
    def solve(self) -> [float]:
        # Create a new model and solver
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        maxLengthCircuit = self.maxLengthCircuit
        model = self.model
        solver = self.solver
        numResistors = self.numResistors

        if (numResistors == 0):
            return []

        if (numResistors == 1):
            return [resistors[0]]

        if (numResistors == 2):
            seriesRes = abs(target - resistors[0] - resistors[1])
            parallelRes = abs(target - (1 / ((1 / resistors[0]) + (1 / resistors[1]))))

            if seriesRes < parallelRes:
                return [resistors[0], resistors[1], -1]
            else:
                return [resistors[0], resistors[1], -2]

        self.create_variables()
        self.addInitialStackConstraint()
        self.addResistorSetConstraint()
        self.addLengthConstraint()
        self.addSingleUseConstraint()
        self.addValidOperatorsConstraint()

        #self.addNoParallelConstraint()
        #self.addForceResultConstraint()

        self.minimizeError()

        solution = self.solution

        '''
        solution_printer = VarArraySolutionPrinter([solution[0], solution[1], solution[2], solution[3], solution[4]])
        solver.SearchForAllSolutions(model, solution_printer)
        solutionSet = solution_printer.getFinalSet()
        print('Number of solutions found: %i' % solution_printer.solution_count())

        best = []
        bestError = float('inf')
        for i in solutionSet:
            print(i)
            error = abs(target - getEquivalentResistance(list(i)))
            if error < bestError:
                bestError = error
                best = i

        return list(best)
        '''

        output = []
        if (solver.Solve(model) == cp_model.OPTIMAL) | (solver.Solve(model) == cp_model.FEASIBLE):
            for i in range(maxLengthCircuit):
                output.append(solver.Value(solution[i]))

            #stackPrint = []
            #for i in range(self.stackLength):
                #stackPrint.append(solver.Value(self.stack[i]))

            #print(stackPrint)
            return [ x for x in output if x != 0 ]
        else:
            # This should theoretically never happen, as none of the constraints
            # should result in an unfeasible solution. At worst it would just
            # find a poor solution. Please contact me if you ever hit this
            # message.
            return [11111]


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                            TESTING:

 Creates instances of the solver and finds the optimal solution, 
 printing information about the execution.

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
def getResults(target: int, resistors: [float]):
    solver = OptimalCircuitDesignProblem(target, resistors)
    solution = solver.solve()
    print("RESISTOR SET: " + str(resistors))
    print("TARGET: " + str(target))
    print("BEST: " + str(solution))
    print("CLOSEST RESISTANCE: " + str(round(getEquivalentResistance(solution), 4)))
    print("ERROR: " + str(round(100 * abs((target - getEquivalentResistance(solution)) / target), 4)) + "%")
    return solution

# TEST: Single resistor
target = 400
resistors = [200]
assert getResults(target, resistors) == [200]
print()

# TEST: Two unique resistors, series best
target = 400
resistors = [200, 100]
assert getResults(target, resistors) == [200, 100, -1]
print()

# TEST: Standard set of 5 unique resistors with nonobvious solution
target = 158
resistors = [100, 200, 300, 400, 500]
getResults(target, resistors)
print()




