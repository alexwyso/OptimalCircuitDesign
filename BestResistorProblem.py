# Alex Wysoczanski (alexjwyso@gmail.com)
# University of Pennsylvania 
# 
# The same set of resistors can used to generate a wide range of equivalent
# resistance values. 
#
# Predominantly as an educational tool, this project devises a novel method of 
# representing a circuit as a list and then establishes a method for finding 
# the circuit design that is optimally close to a desired value.


# This project utilizes the Constraint Programming library from Google OR-Tools
from ortools.sat.python import cp_model
import math

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
        elif n < -1:
            curr = []
            for _ in range(abs(n)):
                curr.append(stack.pop())
            inverses = [(1 / resistor) for resistor in curr]
            stack.append(1 / sum(inverses))
        
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
    assert abs((23.59551 - getEquivalentResistance([100, 200, -1, 50, -2, 75, 50, 50, -1, 75, -1, -3]))) < 0.0001

    return

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                THE OPTIMAL CIRCUIT DESIGN PROBLEM:

 A solver class for the Best Resistor Problem, in which the user seeks
 the arrangement of a series-parallel circuit that yields an
 equivalent resistance closest to a target value.

 INPUT: target      -> the desired equivalent resistance
        resistors   -> the set of available resistors

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class BestResistorProblem():

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
            solution[i] = model.NewIntVar(-1 * numResistors, maxResistor, '{i}th value of solution circuit')
  
        self.solution = solution

    # Adds constraints to ensure the first two elements are operands
    def addInitialStackConstraint(self):
        model: cp_model.CpModel = self.model
        solution = self.solution
        model.Add(solution[0] > 0)
        model.Add(solution[1] > 0)

    # Adds constraints so that a circuit representation of length n
    # fills the first n indices of the solution with resistors/operators 
    # and designates the rest as -1 for unused space
    def addLengthConstraint(self):
        model: cp_model.CpModel = self.model
        maxLengthCircuit = self.maxLengthCircuit

        isEmpty = {}
        for i in range(maxLengthCircuit):
            isEmpty[i] = model.NewBoolVar('{i} is -1')
            model.Add(self.solution[i] == 0).OnlyEnforceIf(isEmpty[i])
            model.Add(self.solution[i] != 0).OnlyEnforceIf(isEmpty[i].Not())

        for i in range(3, maxLengthCircuit):
            model.Add(self.solution[i] == 0).OnlyEnforceIf(isEmpty[i - 1])

        return

    # Adds constraints to ensure each resistor is used at most once
    def addSingleUseConstraint(self):
        model: cp_model.CpModel = self.model
        maxLengthCircuit = self.maxLengthCircuit
        
        op = {}
        for i in range(maxLengthCircuit):
            op[i] = model.NewBoolVar('{i} is an operator')
            model.Add(self.solution[i] <= 0).OnlyEnforceIf(op[i])
            model.Add(self.solution[i] > 0).OnlyEnforceIf(op[i].Not())

        for i in range(maxLengthCircuit):
            for j in range(maxLengthCircuit):
                if (i != j):
                    model.Add(self.solution[i] != self.solution[j]).OnlyEnforceIf(op[i].Not())           

    # Adds constraints to ensure each operator is preceeded by the correct 
    # number of operators and operands, i.e. the Reverse Polish Notation has
    # well formed operators.
    def addValidOperatorsConstraint(self):

        return

    # Runs an instance of the OR-Tools solver for the Optimal Circuit Problem
    def solve(self) -> [float]:
        # Create a new model and solver
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        maxLengthCircuit = self.maxLengthCircuit
        model = self.model
        solver = self.solver

        self.create_variables()
        self.addInitialStackConstraint()
        self.addLengthConstraint()
        self.addSingleUseConstraint()
        self.addValidOperatorsConstraint()

        # TO-DO (MINIMIZE ERROR)
        #error = model.NewIntVar(0, math.ceil(10000 * numResistors), 'error')
        #error = abs((10000 * target) - (10000 * getEquivalentResistance(self.solution)))
        #self.model.Minimize(error)

        output = []
        if solver.Solve(model) == cp_model.FEASIBLE:
            for i in range(maxLengthCircuit):
                output.append(solver.Value(self.solution[i]))
            return output
        else:
            # This should theoretically never happen, as none of the constraints
            # should result in an unfeasible solution. At worst it would just
            # find a poor solution. Please contact me if you ever hit this
            # message.
            return [11111111111]

# FOR DEBUGGING:
target = 7
resistors = [100, 20, 40, 30, 100, 20]
testGetEquivalentResistance()
solver = BestResistorProblem(target, resistors)
solution = solver.solve()
print(solution)
#print(getEquivalentResistance(solution))

#TO DO : change name to Optimal Circuit Design Problem