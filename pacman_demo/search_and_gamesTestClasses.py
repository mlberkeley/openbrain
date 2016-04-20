# search_and_gamesTestClasses.py
# ------------------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# search_and_gamesTestClasses.py
# ------------------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


################
# Search Tests #
################

import re
import testClasses
import textwrap

# import project specific code
import layout
import pacman
from search import SearchProblem

# helper function for printing solutions in solution files
def wrap_solution(solution):
    if type(solution) == type([]):
        return '\n'.join(textwrap.wrap(' '.join(solution)))
    else:
        return str(solution)




def followAction(state, action, problem):
  for action1 in problem.getActions(state):  
    if action == action1:
        next_state = problem.getResult(state, action1)
        return next_state
  return None

def followPath(path, problem):
  state = problem.getStartState()
  states = [state]
  for action in path:
    state = followAction(state, action, problem)
    states.append(state)
  return states

def checkSolution(problem, path):
  state = problem.getStartState()
  for action in path:
    state = followAction(state, action, problem)
  return problem.goalTest(state)

# Search problem on a plain graph
class GraphSearch(SearchProblem):

    # Read in the state graph; define start/end states, edges and costs
    def __init__(self, graph_text):
        self.expanded_states = []
        lines = graph_text.split('\n')
        r = re.match('start_state:(.*)', lines[0])
        if r == None:
            print "Broken graph:"
            print '"""%s"""' % graph_text
            raise Exception("GraphSearch graph specification start_state not found or incorrect on line:" + l)
        self.start_state = r.group(1).strip()
        r = re.match('goal_states:(.*)', lines[1])
        if r == None:
            print "Broken graph:"
            print '"""%s"""' % graph_text
            raise Exception("GraphSearch graph specification goal_states not found or incorrect on line:" + l)
        goals = r.group(1).split()
        self.goals = map(str.strip, goals)
        self.successors = {}
        all_states = set()
        self.orderedSuccessorTuples = []
        for l in lines[2:]:
            if len(l.split()) == 3:
                start, action, next_state = l.split()
                cost = 1
            elif len(l.split()) == 4:
                start, action, next_state, cost = l.split()
            else:
                print "Broken graph:"
                print '"""%s"""' % graph_text
                raise Exception("Invalid line in GraphSearch graph specification on line:" + l)
            cost = float(cost)
            self.orderedSuccessorTuples.append((start, action, next_state, cost))
            all_states.add(start)
            all_states.add(next_state)
            if start not in self.successors:
                self.successors[start] = []
            self.successors[start].append((next_state, action, cost))
        for s in all_states:
            if s not in self.successors:
                self.successors[s] = []

    # Get start state
    def getStartState(self):
        return self.start_state

    # Check if a state is a goal state
    def goalTest(self, state):
        return state in self.goals

    def getActions(self, state):
        self.expanded_states.append(state)
        actions = []
        successors = self.successors[state]
        for successor in successors:
            actions.append(successor[1])
        return actions

    def getResult(self, state, action):
        successors = self.successors[state]
        for successor in successors:
            if successor[1] == action:
                return successor[0]

    def getCost(self, state, action):
        successors = self.successors[state]
        for successor in successors:
            if successor[1] == action:
                return successor[2]
    
    # Calculate total cost of a sequence of actions
    def getCostOfActions(self, actions):
        total_cost = 0
        state = self.start_state
        for a in actions:
            successors = self.successors[state]
            match = False
            for (next_state, action, cost) in successors:
                if a == action:
                    state = next_state
                    total_cost += cost
                    match = True
            if not match:
                print 'invalid action sequence'
                sys.exit(1)
        return total_cost

    # Return a list of all states on which 'actions' was called
    def getExpandedStates(self):
        return self.expanded_states

    def __str__(self):
        print self.successors
        edges = ["%s %s %s %s" % t for t in self.orderedSuccessorTuples]
        return \
"""start_state: %s
goal_states: %s
%s""" % (self.start_state, " ".join(self.goals), "\n".join(edges))



def parseHeuristic(heuristicText):
    heuristic = {}
    for line in heuristicText.split('\n'):
        tokens = line.split()
        if len(tokens) != 2:
            print "Broken heuristic:"
            print '"""%s"""' % graph_text
            raise Exception("GraphSearch heuristic specification broken:" + l)
        state, h = tokens
        heuristic[state] = float(h)

    def graphHeuristic(state, problem=None):
        if state in heuristic:
            return heuristic[state]
        else:
            pp = pprint.PrettyPrinter(indent=4)
            print "Heuristic:"
            pp.pprint(heuristic)
            raise Exception("Graph heuristic called with invalid state: " + str(state))

    return graphHeuristic


class GraphSearchTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(GraphSearchTest, self).__init__(question, testDict)
        self.graph_text = testDict['graph']
        self.alg = testDict['algorithm']
        self.diagram = testDict['diagram']
        self.exactExpansionOrder = testDict.get('exactExpansionOrder', 'True').lower() == "true"
        if 'heuristic' in testDict:
            self.heuristic = parseHeuristic(testDict['heuristic'])
        else:
            self.heuristic = None

    # Note that the return type of this function is a tripple:
    # (solution, expanded states, error message)
    def getSolInfo(self, search):
        alg = getattr(search, self.alg)
        problem = GraphSearch(self.graph_text)
        if self.heuristic != None:
            solution = alg(problem, self.heuristic)
        else:
            solution = alg(problem)

        if type(solution) != type([]):
            return None, None, 'The result of %s must be a list. (Instead, it is %s)' % (self.alg, type(solution))

        return solution, problem.getExpandedStates(), None

    # Run student code.  If an error message is returned, print error and return false.
    # If a good solution is returned, printn the solution and return true; otherwise,
    # print both the correct and student's solution and return false.
    def execute(self, grades, moduleDict, solutionDict):
        search = moduleDict['search']
        searchAgents = moduleDict['searchAgents']
        gold_solution = [str.split(solutionDict['solution']), str.split(solutionDict['rev_solution'])]
        gold_expanded_states = [str.split(solutionDict['expanded_states']), str.split(solutionDict['rev_expanded_states'])]

        solution, expanded_states, error = self.getSolInfo(search)
        if error != None:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % error)
            return False

        if solution in gold_solution and (not self.exactExpansionOrder or expanded_states in gold_expanded_states):
            grades.addMessage('PASS: %s' % self.path)
            grades.addMessage('\tsolution:\t\t%s' % solution)
            grades.addMessage('\texpanded_states:\t%s' % expanded_states)
            return True
        else:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\tgraph:')
            for line in self.diagram.split('\n'):
                grades.addMessage('\t    %s' % (line,))
            grades.addMessage('\tstudent solution:\t\t%s' % solution)
            grades.addMessage('\tstudent expanded_states:\t%s' % expanded_states)
            grades.addMessage('')
            grades.addMessage('\tcorrect solution:\t\t%s' % gold_solution[0])
            grades.addMessage('\tcorrect expanded_states:\t%s' % gold_expanded_states[0])
            grades.addMessage('\tcorrect rev_solution:\t\t%s' % gold_solution[1])
            grades.addMessage('\tcorrect rev_expanded_states:\t%s' % gold_expanded_states[1])
            return False

    def writeSolution(self, moduleDict, filePath):
        search = moduleDict['search']
        searchAgents = moduleDict['searchAgents']
        # open file and write comments
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)
        handle.write('# This solution is designed to support both right-to-left\n')
        handle.write('# and left-to-right implementations.\n')

        # write forward solution
        solution, expanded_states, error = self.getSolInfo(search)
        if error != None: raise Exception("Error in solution code: %s" % error)
        handle.write('solution: "%s"\n' % ' '.join(solution))
        handle.write('expanded_states: "%s"\n' % ' '.join(expanded_states))

        # reverse and write backwards solution
        search.REVERSE_PUSH = not search.REVERSE_PUSH
        solution, expanded_states, error = self.getSolInfo(search)
        if error != None: raise Exception("Error in solution code: %s" % error)
        handle.write('rev_solution: "%s"\n' % ' '.join(solution))
        handle.write('rev_expanded_states: "%s"\n' % ' '.join(expanded_states))

        # clean up
        search.REVERSE_PUSH = not search.REVERSE_PUSH
        handle.close()
        return True



class PacmanSearchTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(PacmanSearchTest, self).__init__(question, testDict)
        self.layout_text = testDict['layout']
        self.alg = testDict['algorithm']
        self.layoutName = testDict['layoutName']

        # TODO: sensible to have defaults like this?
        self.leewayFactor = float(testDict.get('leewayFactor', '1'))
        self.costFn = eval(testDict.get('costFn', 'None'))
        self.searchProblemClassName = testDict.get('searchProblemClass', 'PositionSearchProblem')
        self.heuristicName = testDict.get('heuristic', None)


    def getSolInfo(self, search, searchAgents):
        alg = getattr(search, self.alg)
        lay = layout.Layout([l.strip() for l in self.layout_text.split('\n')])
        start_state = pacman.GameState()
        start_state.initialize(lay, 0)

        problemClass = getattr(searchAgents, self.searchProblemClassName)
        problemOptions = {}
        if self.costFn != None:
            problemOptions['costFn'] = self.costFn
        problem = problemClass(start_state, **problemOptions)
        heuristic = getattr(searchAgents, self.heuristicName) if self.heuristicName != None else None

        if heuristic != None:
            solution = alg(problem, heuristic)
        else:
            solution = alg(problem)

        if type(solution) != type([]):
            return None, None, 'The result of %s must be a list. (Instead, it is %s)' % (self.alg, type(solution))

        from game import Directions
        dirs = Directions.LEFT.keys()
        if [el in dirs for el in solution].count(False) != 0:
            return None, None, 'Output of %s must be a list of actions from game.Directions' % self.alg

        expanded = problem._expanded
        return solution, expanded, None

    def execute(self, grades, moduleDict, solutionDict):
        search = moduleDict['search']
        searchAgents = moduleDict['searchAgents']
        gold_solution = [str.split(solutionDict['solution']), str.split(solutionDict['rev_solution'])]
        gold_expanded = max(int(solutionDict['expanded_nodes']), int(solutionDict['rev_expanded_nodes']))

        solution, expanded, error = self.getSolInfo(search, searchAgents)
        if error != None:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('%s' % error)
            return False

        # FIXME: do we want to standardize test output format?

        if solution not in gold_solution:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('Solution not correct.')
            grades.addMessage('\tstudent solution length: %s' % len(solution))
            grades.addMessage('\tstudent solution:\n%s' % wrap_solution(solution))
            grades.addMessage('')
            grades.addMessage('\tcorrect solution length: %s' % len(gold_solution[0]))
            grades.addMessage('\tcorrect (reversed) solution length: %s' % len(gold_solution[1]))
            grades.addMessage('\tcorrect solution:\n%s' % wrap_solution(gold_solution[0]))
            grades.addMessage('\tcorrect (reversed) solution:\n%s' % wrap_solution(gold_solution[1]))
            return False

        if expanded > self.leewayFactor * gold_expanded and expanded > gold_expanded + 1:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('Too many node expanded; are you expanding nodes twice?')
            grades.addMessage('\tstudent nodes expanded: %s' % expanded)
            grades.addMessage('')
            grades.addMessage('\tcorrect nodes expanded: %s (leewayFactor %s)' % (gold_expanded, self.leewayFactor))
            return False

        grades.addMessage('PASS: %s' % self.path)
        grades.addMessage('\tpacman layout:\t\t%s' % self.layoutName)
        grades.addMessage('\tsolution length: %s' % len(solution))
        grades.addMessage('\tnodes expanded:\t\t%s' % expanded)
        return True


    def writeSolution(self, moduleDict, filePath):
        search = moduleDict['search']
        searchAgents = moduleDict['searchAgents']
        # open file and write comments
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)
        handle.write('# This solution is designed to support both right-to-left\n')
        handle.write('# and left-to-right implementations.\n')
        handle.write('# Number of nodes expanded must be with a factor of %s of the numbers below.\n' % self.leewayFactor)

        # write forward solution
        solution, expanded, error = self.getSolInfo(search, searchAgents)
        if error != None: raise Exception("Error in solution code: %s" % error)
        handle.write('solution: """\n%s\n"""\n' % wrap_solution(solution))
        handle.write('expanded_nodes: "%s"\n' % expanded)

        # write backward solution
        search.REVERSE_PUSH = not search.REVERSE_PUSH
        solution, expanded, error = self.getSolInfo(search, searchAgents)
        if error != None: raise Exception("Error in solution code: %s" % error)
        handle.write('rev_solution: """\n%s\n"""\n' % wrap_solution(solution))
        handle.write('rev_expanded_nodes: "%s"\n' % expanded)

        # clean up
        search.REVERSE_PUSH = not search.REVERSE_PUSH
        handle.close()
        return True


from game import Actions
def getStatesFromPath(start, path):
    "Returns the list of states visited along the path"
    vis = [start]
    curr = start
    for a in path:
        x,y = curr
        dx, dy = Actions.directionToVector(a)
        curr = (int(x + dx), int(y + dy))
        vis.append(curr)
    return vis

class CornerProblemTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(CornerProblemTest, self).__init__(question, testDict)
        self.layoutText = testDict['layout']
        self.layoutName = testDict['layoutName']

    def solution(self, search, searchAgents):
        lay = layout.Layout([l.strip() for l in self.layoutText.split('\n')])
        gameState = pacman.GameState()
        gameState.initialize(lay, 0)
        problem = searchAgents.CornersProblem(gameState)
        path = search.astar(problem)

        gameState = pacman.GameState()
        gameState.initialize(lay, 0)
        visited = getStatesFromPath(gameState.getPacmanPosition(), path)
        top, right = gameState.getWalls().height-2, gameState.getWalls().width-2
        missedCorners = [p for p in ((1,1), (1,top), (right, 1), (right, top)) if p not in visited]

        return path, missedCorners

    def execute(self, grades, moduleDict, solutionDict):
        search = moduleDict['search']
        searchAgents = moduleDict['searchAgents']
        gold_length = int(solutionDict['solution_length'])
        solution, missedCorners = self.solution(search, searchAgents)

        if type(solution) != type([]):
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('The result must be a list. (Instead, it is %s)' % type(solution))
            return False

        if len(missedCorners) != 0:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('Corners missed: %s' % missedCorners)
            return False

        if len(solution) != gold_length:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('Optimal solution not found.')
            grades.addMessage('\tstudent solution length:\n%s' % len(solution))
            grades.addMessage('')
            grades.addMessage('\tcorrect solution length:\n%s' % gold_length)
            return False

        grades.addMessage('PASS: %s' % self.path)
        grades.addMessage('\tpacman layout:\t\t%s' % self.layoutName)
        grades.addMessage('\tsolution length:\t\t%s' % len(solution))
        return True

    def writeSolution(self, moduleDict, filePath):
        search = moduleDict['search']
        searchAgents = moduleDict['searchAgents']
        # open file and write comments
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)

        print "Solving problem", self.layoutName
        print self.layoutText

        path, _ = self.solution(search, searchAgents)
        length = len(path)
        print "Problem solved"

        handle.write('solution_length: "%s"\n' % length)
        handle.close()




# template = """class: "HeuristicTest"
#
# heuristic: "foodHeuristic"
# searchProblemClass: "FoodSearchProblem"
# layoutName: "Test %s"
# layout: \"\"\"
# %s
# \"\"\"
# """
#
# for i, (_, _, l) in enumerate(doneTests + foodTests):
#     f = open("food_heuristic_%s.test" % (i+1), "w")
#     f.write(template % (i+1, "\n".join(l)))
#     f.close()

class HeuristicTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(HeuristicTest, self).__init__(question, testDict)
        self.layoutText = testDict['layout']
        self.layoutName = testDict['layoutName']
        self.searchProblemClassName = testDict['searchProblemClass']
        self.heuristicName = testDict['heuristic']

    def setupProblem(self, searchAgents):
        lay = layout.Layout([l.strip() for l in self.layoutText.split('\n')])
        gameState = pacman.GameState()
        gameState.initialize(lay, 0)
        problemClass = getattr(searchAgents, self.searchProblemClassName)
        problem = problemClass(gameState)
        state = problem.getStartState()
        heuristic = getattr(searchAgents, self.heuristicName)

        return problem, state, heuristic

    def checkHeuristic(self, heuristic, problem, state, solutionCost):
        h0 = heuristic(state, problem)

        if solutionCost == 0:
            if h0 == 0:
                return True, ''
            else:
                return False, 'Heuristic failed H(goal) == 0 test'

        if h0 < 0:
            return False, 'Heuristic failed H >= 0 test'
        if not h0 > 0:
            return False, 'Heuristic failed non-triviality test'
        if not h0 <= solutionCost:
            return False, 'Heuristic failed admissibility test'

        for action in problem.getActions(state):
            succ = problem.getResult(state, action)
            stepCost = problem.getCost(state, action)
            h1 = heuristic(succ, problem)
            if h1 < 0: return False, 'Heuristic failed H >= 0 test'
            if h0 - h1 > stepCost: return False, 'Heuristic failed consistency test'

        return True, ''

    def execute(self, grades, moduleDict, solutionDict):
        search = moduleDict['search']
        searchAgents = moduleDict['searchAgents']
        solutionCost = int(solutionDict['solution_cost'])
        problem, state, heuristic = self.setupProblem(searchAgents)

        passed, message = self.checkHeuristic(heuristic, problem, state, solutionCost)

        if not passed:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('%s' % message)
            return False
        else:
            grades.addMessage('PASS: %s' % self.path)
            return True

    def writeSolution(self, moduleDict, filePath):
        search = moduleDict['search']
        searchAgents = moduleDict['searchAgents']
        # open file and write comments
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)

        print "Solving problem", self.layoutName, self.heuristicName
        print self.layoutText
        problem, _, heuristic = self.setupProblem(searchAgents)
        path = search.astar(problem, heuristic)
        cost = problem.getCostOfActions(path)
        print "Problem solved"

        handle.write('solution_cost: "%s"\n' % cost)
        handle.close()
        return True






class HeuristicGrade(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(HeuristicGrade, self).__init__(question, testDict)
        self.layoutText = testDict['layout']
        self.layoutName = testDict['layoutName']
        self.searchProblemClassName = testDict['searchProblemClass']
        self.heuristicName = testDict['heuristic']
        self.basePoints = int(testDict['basePoints'])
        self.thresholds = [int(t) for t in testDict['gradingThresholds'].split()]

    def setupProblem(self, searchAgents):
        lay = layout.Layout([l.strip() for l in self.layoutText.split('\n')])
        gameState = pacman.GameState()
        gameState.initialize(lay, 0)
        problemClass = getattr(searchAgents, self.searchProblemClassName)
        problem = problemClass(gameState)
        state = problem.getStartState()
        heuristic = getattr(searchAgents, self.heuristicName)

        return problem, state, heuristic


    def execute(self, grades, moduleDict, solutionDict):
        search = moduleDict['search']
        searchAgents = moduleDict['searchAgents']
        problem, _, heuristic = self.setupProblem(searchAgents)

        path = search.astar(problem, heuristic)

        expanded = problem._expanded

        if not checkSolution(problem, path):
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\tReturned path is not a solution.')
            grades.addMessage('\tpath returned by astar: %s' % expanded)
            return False

        grades.addPoints(self.basePoints)
        points = 0
        for threshold in self.thresholds:
            if expanded <= threshold:
                points += 1
        grades.addPoints(points)
        if points >= len(self.thresholds):
            grades.addMessage('PASS: %s' % self.path)
        else:
            grades.addMessage('FAIL: %s' % self.path)
        grades.addMessage('\texpanded nodes: %s' % expanded)
        grades.addMessage('\tthresholds: %s' % self.thresholds)

        return True


    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)
        handle.write('# File intentionally blank.\n')
        handle.close()
        return True





# template = """class: "ClosestDotTest"
#
# layoutName: "Test %s"
# layout: \"\"\"
# %s
# \"\"\"
# """
#
# for i, (_, _, l) in enumerate(foodTests):
#     f = open("closest_dot_%s.test" % (i+1), "w")
#     f.write(template % (i+1, "\n".join(l)))
#     f.close()

class ClosestDotTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(ClosestDotTest, self).__init__(question, testDict)
        self.layoutText = testDict['layout']
        self.layoutName = testDict['layoutName']

    def solution(self, searchAgents):
        lay = layout.Layout([l.strip() for l in self.layoutText.split('\n')])
        gameState = pacman.GameState()
        gameState.initialize(lay, 0)
        path = searchAgents.ClosestDotSearchAgent().findPathToClosestDot(gameState)
        return path

    def execute(self, grades, moduleDict, solutionDict):
        search = moduleDict['search']
        searchAgents = moduleDict['searchAgents']
        gold_length = int(solutionDict['solution_length'])
        solution = self.solution(searchAgents)

        if type(solution) != type([]):
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\tThe result must be a list. (Instead, it is %s)' % type(solution))
            return False

        if len(solution) != gold_length:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('Closest dot not found.')
            grades.addMessage('\tstudent solution length:\n%s' % len(solution))
            grades.addMessage('')
            grades.addMessage('\tcorrect solution length:\n%s' % gold_length)
            return False

        grades.addMessage('PASS: %s' % self.path)
        grades.addMessage('\tpacman layout:\t\t%s' % self.layoutName)
        grades.addMessage('\tsolution length:\t\t%s' % len(solution))
        return True

    def writeSolution(self, moduleDict, filePath):
        search = moduleDict['search']
        searchAgents = moduleDict['searchAgents']
        # open file and write comments
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)

        print "Solving problem", self.layoutName
        print self.layoutText

        length = len(self.solution(searchAgents))
        print "Problem solved"

        handle.write('solution_length: "%s"\n' % length)
        handle.close()
        return True




class CornerHeuristicSanity(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(CornerHeuristicSanity, self).__init__(question, testDict)
        self.layout_text = testDict['layout']

    def execute(self, grades, moduleDict, solutionDict):
        search = moduleDict['search']
        searchAgents = moduleDict['searchAgents']
        game_state = pacman.GameState()
        lay = layout.Layout([l.strip() for l in self.layout_text.split('\n')])
        game_state.initialize(lay, 0)
        problem = searchAgents.CornersProblem(game_state)
        start_state = problem.getStartState()
        h0 = searchAgents.cornersHeuristic(start_state, problem)
        # cornerConsistencyA
        for action in problem.getActions(start_state):
            next_state = problem.getResult(start_state, action)
            h1 = searchAgents.cornersHeuristic(next_state, problem)
            if h0 - h1 > 1:
                grades.addMessage('FAIL: inconsistent heuristic')
                return False
        heuristic_cost = searchAgents.cornersHeuristic(start_state, problem)
        true_cost = float(solutionDict['cost'])
        # cornerNontrivial
        if heuristic_cost == 0:
            grades.addMessage('FAIL: must use non-trivial heuristic')
            return False
        # cornerAdmissible
        if heuristic_cost > true_cost:
            grades.addMessage('FAIL: Inadmissible heuristic')
            return False
        path = solutionDict['path'].split()
        states = followPath(path, problem)
        heuristics = []
        for state in states:
            heuristics.append(searchAgents.cornersHeuristic(state, problem))
        for i in range(0, len(heuristics) - 1):
            h0 = heuristics[i]
            h1 = heuristics[i+1]
            # cornerConsistencyB
            if h0 - h1 > 1:
                grades.addMessage('FAIL: inconsistent heuristic')
                return False
            # cornerPosH
            if h0 < 0 or h1 <0:
                grades.addMessage('FAIL: non-positive heuristic')
                return False
        # cornerGoalH
        if heuristics[len(heuristics) - 1] != 0:
            grades.addMessage('FAIL: heuristic non-zero at goal')
            return False
        grades.addMessage('PASS: heuristic value less than true cost at start state')
        return True

    def writeSolution(self, moduleDict, filePath):
        search = moduleDict['search']
        searchAgents = moduleDict['searchAgents']
        # write comment
        handle = open(filePath, 'w')
        handle.write('# In order for a heuristic to be admissible, the value\n')
        handle.write('# of the heuristic must be less at each state than the\n')
        handle.write('# true cost of the optimal path from that state to a goal.\n')

        # solve problem and write solution
        lay = layout.Layout([l.strip() for l in self.layout_text.split('\n')])
        start_state = pacman.GameState()
        start_state.initialize(lay, 0)
        problem = searchAgents.CornersProblem(start_state)
        solution = search.astar(problem, searchAgents.cornersHeuristic)
        handle.write('cost: "%d"\n' % len(solution))
        handle.write('path: """\n%s\n"""\n' % wrap_solution(solution))
        handle.close()
        return True



class CornerHeuristicPacman(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(CornerHeuristicPacman, self).__init__(question, testDict)
        self.layout_text = testDict['layout']

    def execute(self, grades, moduleDict, solutionDict):
        search = moduleDict['search']
        searchAgents = moduleDict['searchAgents']
        total = 0
        true_cost = float(solutionDict['cost'])
        thresholds = map(int, solutionDict['thresholds'].split())
        game_state = pacman.GameState()
        lay = layout.Layout([l.strip() for l in self.layout_text.split('\n')])
        game_state.initialize(lay, 0)
        problem = searchAgents.CornersProblem(game_state)
        start_state = problem.getStartState()
        if searchAgents.cornersHeuristic(start_state, problem) > true_cost:
            grades.addMessage('FAIL: Inadmissible heuristic')
            return False
        path = search.astar(problem, searchAgents.cornersHeuristic)
        print "path:", path
        print "path length:", len(path)
        cost = problem.getCostOfActions(path)
        if cost > true_cost:
            grades.addMessage('FAIL: Inconsistent heuristic')
            return False
        expanded = problem._expanded
        points = 0
        for threshold in thresholds:
            if expanded <= threshold:
                points += 1
        grades.addPoints(points)
        if points >= len(thresholds):
            grades.addMessage('PASS: Heuristic resulted in expansion of %d nodes' % expanded)
        else:
            grades.addMessage('FAIL: Heuristic resulted in expansion of %d nodes' % expanded)
        return True

    def writeSolution(self, moduleDict, filePath):
        search = moduleDict['search']
        searchAgents = moduleDict['searchAgents']
        # write comment
        handle = open(filePath, 'w')
        handle.write('# This solution file specifies the length of the optimal path\n')
        handle.write('# as well as the thresholds on number of nodes expanded to be\n')
        handle.write('# used in scoring.\n')

        # solve problem and write solution
        lay = layout.Layout([l.strip() for l in self.layout_text.split('\n')])
        start_state = pacman.GameState()
        start_state.initialize(lay, 0)
        problem = searchAgents.CornersProblem(start_state)
        solution = search.astar(problem, searchAgents.cornersHeuristic)
        handle.write('cost: "%d"\n' % len(solution))
        handle.write('path: """\n%s\n"""\n' % wrap_solution(solution))
        handle.write('thresholds: "2000 1600 1200"\n')
        handle.close()
        return True



####################
# Multiagent Tests #
####################

# A minimax tree which interfaces like gameState
#     state.getNumAgents()
#     state.isWin()
#     state.isLose()
#     state.generateSuccessor(agentIndex, action)
#     state.getScore()
#           used by multiAgents.scoreEvaluationFunction, which is the default
#
import testClasses
import json

from collections import defaultdict
from pprint import PrettyPrinter
pp = PrettyPrinter()

from game import Agent
from pacman import GameState
from ghostAgents import RandomGhost, DirectionalGhost
import random, math, traceback, sys, os
import layout, pacman
import autograder
# import grading

VERBOSE = False

class MultiagentTreeState(object):
    def __init__(self, problem, state):
        self.problem = problem
        self.state = state

    def generateSuccessor(self, agentIndex, action):
        if VERBOSE:
            print "generateSuccessor(%s, %s, %s) -> %s" % (self.state, agentIndex, action, self.problem.stateToSuccessorMap[self.state][action])
        successor = self.problem.stateToSuccessorMap[self.state][action]
        self.problem.generatedStates.add(successor)
        return MultiagentTreeState(self.problem, successor)

    def getScore(self):
        if VERBOSE:
            print "getScore(%s) -> %s" % (self.state, self.problem.evaluation[self.state])
        if self.state not in self.problem.evaluation:
            raise Exception('getScore() called on non-terminal state or before maximum depth achieved.')
        return float(self.problem.evaluation[self.state])

    def getLegalActions(self, agentIndex=0):
        if VERBOSE:
            print "getLegalActions(%s) -> %s" % (self.state, self.problem.stateToActions[self.state])
        #if len(self.problem.stateToActions[self.state]) == 0:
        #    print "WARNING: getLegalActions called on leaf state %s" % (self.state,)
        return list(self.problem.stateToActions[self.state])

    def isWin(self):
        if VERBOSE:
            print "isWin(%s) -> %s" % (self.state, self.state in self.problem.winStates)
        return self.state in self.problem.winStates

    def isLose(self):
        if VERBOSE:
            print "isLose(%s) -> %s" % (self.state, self.state in self.problem.loseStates)
        return self.state in self.problem.loseStates

    def getNumAgents(self):
        if VERBOSE:
            print "getNumAgents(%s) -> %s" % (self.state, self.problem.numAgents)
        return self.problem.numAgents


class MultiagentTreeProblem(object):
    def __init__(self, numAgents, startState, winStates, loseStates, successors, evaluation):
        self.startState = MultiagentTreeState(self, startState)

        self.numAgents = numAgents
        self.winStates = winStates
        self.loseStates = loseStates
        self.evaluation = evaluation
        self.successors = successors

        self.reset()

        self.stateToSuccessorMap = defaultdict(dict)
        self.stateToActions = defaultdict(list)
        for state, action, nextState in successors:
            self.stateToActions[state].append(action)
            self.stateToSuccessorMap[state][action] = nextState

    def reset(self):
        self.generatedStates = set([self.startState.state])


def parseTreeProblem(testDict):
    numAgents = int(testDict["num_agents"])
    startState = testDict["start_state"]
    winStates = set(testDict["win_states"].split(" "))
    loseStates = set(testDict["lose_states"].split(" "))
    successors = []

    evaluation = {}
    for line in testDict["evaluation"].split('\n'):
        tokens = line.split()
        if len(tokens) == 2:
            state, value = tokens
            evaluation[state] = float(value)
        else:
            raise Exception, "[parseTree] Bad evaluation line: |%s|" % (line,)

    for line in testDict["successors"].split('\n'):
        tokens = line.split()
        if len(tokens) == 3:
            state, action, nextState = tokens
            successors.append((state, action, nextState))
        else:
            raise Exception, "[parseTree] Bad successor line: |%s|" % (line,)

    return MultiagentTreeProblem(numAgents, startState, winStates, loseStates, successors, evaluation)



def run(lay, layName, pac, ghosts, disp, nGames=1, name='games'):
    """
    Runs a few games and outputs their statistics.
    """
    starttime = time.time()
    print '*** Running %s on' % name, layName, '%d time(s).' % nGames
    games = pacman.runGames(lay, pac, ghosts, disp, nGames, False, catchExceptions=True, timeout=120)
    print '*** Finished running %s on' % name, layName, 'after %d seconds.' % (time.time() - starttime)
    stats = {'time': time.time() - starttime, 'wins': [g.state.isWin() for g in games].count(True), 'games': games, 'scores': [g.state.getScore() for g in games],
             'timeouts': [g.agentTimeout for g in games].count(True), 'crashes': [g.agentCrashed for g in games].count(True)}
    print '*** Won %d out of %d games. Average score: %f ***' % (stats['wins'], len(games), sum(stats['scores']) * 1.0 / len(games))
    return stats

class GradingAgent(Agent):
    def __init__(self, seed, studentAgent, optimalActions, altDepthActions, partialPlyBugActions):
        # save student agent and actions of refernce agents
        self.studentAgent = studentAgent
        self.optimalActions = optimalActions
        self.altDepthActions = altDepthActions
        self.partialPlyBugActions = partialPlyBugActions
        # create fields for storing specific wrong actions
        self.suboptimalMoves = []
        self.wrongStatesExplored = -1
        # boolean vectors represent types of implementation the student could have
        self.actionsConsistentWithOptimal = [True for i in range(len(optimalActions[0]))]
        self.actionsConsistentWithAlternativeDepth = [True for i in range(len(altDepthActions[0]))]
        self.actionsConsistentWithPartialPlyBug = [True for i in range(len(partialPlyBugActions[0]))]
        # keep track of elapsed moves
        self.stepCount = 0
        self.seed = seed

    def registerInitialState(self, state):
        if 'registerInitialState' in dir(self.studentAgent):
            self.studentAgent.registerInitialState(state)
        random.seed(self.seed)

    def getAction(self, state):
        GameState.getAndResetExplored()
        studentAction = (self.studentAgent.getAction(state), len(GameState.getAndResetExplored()))
        optimalActions = self.optimalActions[self.stepCount]
        altDepthActions = self.altDepthActions[self.stepCount]
        partialPlyBugActions = self.partialPlyBugActions[self.stepCount]
        studentOptimalAction = False
        curRightStatesExplored = False;
        for i in range(len(optimalActions)):
            if studentAction[0] in optimalActions[i][0]:
                studentOptimalAction = True
            else:
                self.actionsConsistentWithOptimal[i] = False
            if studentAction[1] == int(optimalActions[i][1]):
                curRightStatesExplored = True
        if not curRightStatesExplored and self.wrongStatesExplored < 0:
            self.wrongStatesExplored = 1
        for i in range(len(altDepthActions)):
            if studentAction[0] not in altDepthActions[i]:
                self.actionsConsistentWithAlternativeDepth[i] = False
        for i in range(len(partialPlyBugActions)):
            if studentAction[0] not in partialPlyBugActions[i]:
                self.actionsConsistentWithPartialPlyBug[i] = False
        if not studentOptimalAction:
            self.suboptimalMoves.append((state, studentAction[0], optimalActions[0][0][0]))
        self.stepCount += 1
        random.seed(self.seed + self.stepCount)
        return optimalActions[0][0][0]

    def getSuboptimalMoves(self):
        return self.suboptimalMoves

    def getWrongStatesExplored(self):
        return self.wrongStatesExplored

    def checkFailure(self):
        """
        Return +n if have n suboptimal moves.
        Return -1 if have only off by one depth moves.
        Return 0 otherwise.
        """
        if self.wrongStatesExplored > 0:
            return -3
        if self.actionsConsistentWithOptimal.count(True) > 0:
            return 0
        elif self.actionsConsistentWithPartialPlyBug.count(True) > 0:
            return -2
        elif self.actionsConsistentWithAlternativeDepth.count(True) > 0:
            return -1
        else:
            return len(self.suboptimalMoves)


class PolyAgent(Agent):
    def __init__(self, seed, multiAgents, ourPacOptions, depth):
        # prepare our pacman agents
        solutionAgents, alternativeDepthAgents, partialPlyBugAgents = self.construct_our_pacs(multiAgents, ourPacOptions)
        for p in solutionAgents:
            p.depth = depth
        for p in partialPlyBugAgents:
            p.depth = depth
        for p in alternativeDepthAgents[:2]:
            p.depth = max(1, depth - 1)
        for p in alternativeDepthAgents[2:]:
            p.depth = depth + 1
        self.solutionAgents = solutionAgents
        self.alternativeDepthAgents = alternativeDepthAgents
        self.partialPlyBugAgents = partialPlyBugAgents
        # prepare fields for storing the results
        self.optimalActionLists = []
        self.alternativeDepthLists = []
        self.partialPlyBugLists = []
        self.seed = seed
        self.stepCount = 0

    def select(self, list, indices):
        """
        Return a sublist of elements given by indices in list.
        """
        return [list[i] for i in indices]

    def construct_our_pacs(self, multiAgents, keyword_dict):
        pacs_without_stop = [multiAgents.StaffMultiAgentSearchAgent(**keyword_dict) for i in range(3)]
        keyword_dict['keepStop'] = 'True'
        pacs_with_stop = [multiAgents.StaffMultiAgentSearchAgent(**keyword_dict) for i in range(3)]
        keyword_dict['usePartialPlyBug'] = 'True'
        partial_ply_bug_pacs = [multiAgents.StaffMultiAgentSearchAgent(**keyword_dict)]
        keyword_dict['keepStop'] = 'False'
        partial_ply_bug_pacs = partial_ply_bug_pacs + [multiAgents.StaffMultiAgentSearchAgent(**keyword_dict)]
        for pac in pacs_with_stop + pacs_without_stop + partial_ply_bug_pacs:
            pac.verbose = False
        ourpac = [pacs_with_stop[0], pacs_without_stop[0]]
        alternative_depth_pacs = self.select(pacs_with_stop + pacs_without_stop, [1, 4, 2, 5])
        return (ourpac, alternative_depth_pacs, partial_ply_bug_pacs)

    def registerInitialState(self, state):
        for agent in self.solutionAgents + self.alternativeDepthAgents:
            if 'registerInitialState' in dir(agent):
                agent.registerInitialState(state)
        random.seed(self.seed)

    def getAction(self, state):
        # survey agents
        GameState.getAndResetExplored()
        optimalActionLists = []
        for agent in self.solutionAgents:
            optimalActionLists.append((agent.getBestPacmanActions(state)[0], len(GameState.getAndResetExplored())))
        alternativeDepthLists = [agent.getBestPacmanActions(state)[0] for agent in self.alternativeDepthAgents]
        partialPlyBugLists = [agent.getBestPacmanActions(state)[0] for agent in self.partialPlyBugAgents]
        # record responses
        self.optimalActionLists.append(optimalActionLists)
        self.alternativeDepthLists.append(alternativeDepthLists)
        self.partialPlyBugLists.append(partialPlyBugLists)
        self.stepCount += 1
        random.seed(self.seed + self.stepCount)
        return optimalActionLists[0][0][0]

    def getTraces(self):
        # return traces from individual agents
        return (self.optimalActionLists, self.alternativeDepthLists, self.partialPlyBugLists)

class PacmanGameTreeTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(PacmanGameTreeTest, self).__init__(question, testDict)
        self.seed = int(self.testDict['seed'])
        self.alg = self.testDict['alg']
        self.layout_text = self.testDict['layout']
        self.layout_name = self.testDict['layoutName']
        self.depth = int(self.testDict['depth'])
        self.max_points = int(self.testDict['max_points'])

    def execute(self, grades, moduleDict, solutionDict):
        # load student code and staff code solutions
        multiAgents = moduleDict['multiAgents']
        studentAgent = getattr(multiAgents, self.alg)(depth=self.depth)
        allActions = map(lambda x: json.loads(x), solutionDict['optimalActions'].split('\n'))
        altDepthActions = map(lambda x: json.loads(x), solutionDict['altDepthActions'].split('\n'))
        partialPlyBugActions = map(lambda x: json.loads(x), solutionDict['partialPlyBugActions'].split('\n'))
        # set up game state and play a game
        random.seed(self.seed)
        lay = layout.Layout([l.strip() for l in self.layout_text.split('\n')])
        pac = GradingAgent(self.seed, studentAgent, allActions, altDepthActions, partialPlyBugActions)
        # check return codes and assign grades
        disp = self.question.getDisplay()
        stats = run(lay, self.layout_name, pac, [DirectionalGhost(i + 1) for i in range(2)], disp, name=self.alg)
        if stats['timeouts'] > 0:
            self.addMessage('Agent timed out on smallClassic.  No credit')
            return self.testFail(grades)
        if stats['crashes'] > 0:
            self.addMessage('Agent crashed on smallClassic.  No credit')
            return self.testFail(grades)
        code = pac.checkFailure()
        if code == 0:
            return self.testPass(grades)
        elif code == -3:
            if pac.getWrongStatesExplored() >=0:
                self.addMessage('Bug: Wrong number of states expanded.')
                return self.testFail(grades)
            else:
                return self.testPass(grades)
        elif code == -2:
            self.addMessage('Bug: Partial Ply Bug')
            return self.testFail(grades)
        elif code == -1:
            self.addMessage('Bug: Search depth off by 1')
            return self.testFail(grades)
        elif code > 0:
            moves = pac.getSuboptimalMoves()
            state, studentMove, optMove = random.choice(moves)
            self.addMessage('Bug: Suboptimal moves')
            self.addMessage('State:%s\nStudent Move:%s\nOptimal Move:%s' % (state, studentMove, optMove))
            return self.testFail(grades)

    def writeList(self, handle, name, list):
        handle.write('%s: """\n' % name)
        for l in list:
            handle.write('%s\n' % json.dumps(l))
        handle.write('"""\n')

    def writeSolution(self, moduleDict, filePath):
        # load module, set seed, create ghosts and macman, run game
        multiAgents = moduleDict['multiAgents']
        random.seed(self.seed)
        lay = layout.Layout([l.strip() for l in self.layout_text.split('\n')])
        if self.alg == 'ExpectimaxAgent':
            ourPacOptions = {'expectimax': 'True'}
        elif self.alg == 'AlphaBetaAgent':
            ourPacOptions = {'alphabeta': 'True'}
        else:
            ourPacOptions = {}
        pac = PolyAgent(self.seed, multiAgents, ourPacOptions, self.depth)
        disp = self.question.getDisplay()
        run(lay, self.layout_name, pac, [DirectionalGhost(i + 1) for i in range(2)], disp, name=self.alg)
        (optimalActions, altDepthActions, partialPlyBugActions) = pac.getTraces()
        # recover traces and record to file
        handle = open(filePath, 'w')
        self.writeList(handle, 'optimalActions', optimalActions)
        self.writeList(handle, 'altDepthActions', altDepthActions)
        self.writeList(handle, 'partialPlyBugActions', partialPlyBugActions)
        handle.close()



class GraphGameTreeTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(GraphGameTreeTest, self).__init__(question, testDict)
        self.problem = parseTreeProblem(testDict)
        self.alg = self.testDict['alg']
        self.diagram = self.testDict['diagram'].split('\n')
        self.depth = int(self.testDict['depth'])

    def solveProblem(self, multiAgents):
        self.problem.reset()
        studentAgent = getattr(multiAgents, self.alg)(depth=self.depth)
        action = studentAgent.getAction(self.problem.startState)
        generated = self.problem.generatedStates
        return action, " ".join([str(s) for s in sorted(generated)])

    def addDiagram(self):
        self.addMessage('Tree:')
        for line in self.diagram:
            self.addMessage(line)

    def execute(self, grades, moduleDict, solutionDict):
        multiAgents = moduleDict['multiAgents']
        goldAction = solutionDict['action']
        goldGenerated = solutionDict['generated']
        action, generated = self.solveProblem(multiAgents)

        fail = False
        if action != goldAction:
            self.addMessage('Incorrect move for depth=%s' % (self.depth,))
            self.addMessage('    Student move: %s\n    Optimal move: %s' % (action, goldAction))
            fail = True

        if generated != goldGenerated:
            self.addMessage('Incorrect generated nodes for depth=%s' % (self.depth,))
            self.addMessage('    Student generated nodes: %s\n    Correct generated nodes: %s' % (generated, goldGenerated))
            fail = True

        if fail:
            self.addDiagram()
            return self.testFail(grades)
        else:
            return self.testPass(grades)

    def writeSolution(self, moduleDict, filePath):
        multiAgents = moduleDict['multiAgents']
        action, generated = self.solveProblem(multiAgents)
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('action: "%s"\n' % (action,))
            handle.write('generated: "%s"\n' % (generated,))
        return True


import time
from util import TimeoutFunction


class EvalAgentTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(EvalAgentTest, self).__init__(question, testDict)
        self.layoutName = testDict['layoutName']
        self.agentName = testDict['agentName']
        self.ghosts = eval(testDict['ghosts'])
        self.maxTime = int(testDict['maxTime'])
        self.seed = int(testDict['randomSeed'])
        self.numGames = int(testDict['numGames'])

        self.scoreMinimum = int(testDict['scoreMinimum']) if 'scoreMinimum' in testDict else None
        self.nonTimeoutMinimum = int(testDict['nonTimeoutMinimum']) if 'nonTimeoutMinimum' in testDict else None
        self.winsMinimum = int(testDict['winsMinimum']) if 'winsMinimum' in testDict else None

        self.scoreThresholds = [int(s) for s in testDict.get('scoreThresholds','').split()]
        self.nonTimeoutThresholds = [int(s) for s in testDict.get('nonTimeoutThresholds','').split()]
        self.winsThresholds = [int(s) for s in testDict.get('winsThresholds','').split()]

        self.maxPoints = sum([len(t) for t in [self.scoreThresholds, self.nonTimeoutThresholds, self.winsThresholds]])
        self.agentArgs = testDict.get('agentArgs', '')


    def execute(self, grades, moduleDict, solutionDict):
        startTime = time.time()

        agentType = getattr(moduleDict['multiAgents'], self.agentName)
        agentOpts = pacman.parseAgentArgs(self.agentArgs) if self.agentArgs != '' else {}
        agent = agentType(**agentOpts)

        lay = layout.getLayout(self.layoutName, 3)

        disp = self.question.getDisplay()

        random.seed(self.seed)
        games = pacman.runGames(lay, agent, self.ghosts, disp, self.numGames, False, catchExceptions=True, timeout=self.maxTime)
        totalTime = time.time() - startTime

        stats = {'time': totalTime, 'wins': [g.state.isWin() for g in games].count(True),
                 'games': games, 'scores': [g.state.getScore() for g in games],
                 'timeouts': [g.agentTimeout for g in games].count(True), 'crashes': [g.agentCrashed for g in games].count(True)}

        averageScore = sum(stats['scores']) / float(len(stats['scores']))
        nonTimeouts = self.numGames - stats['timeouts']
        wins = stats['wins']

        def gradeThreshold(value, minimum, thresholds, name):
            points = 0
            passed = (minimum == None) or (value >= minimum)
            if passed:
                for t in thresholds:
                    if value >= t:
                        points += 1
            return (passed, points, value, minimum, thresholds, name)

        results = [gradeThreshold(averageScore, self.scoreMinimum, self.scoreThresholds, "average score"),
                   gradeThreshold(nonTimeouts, self.nonTimeoutMinimum, self.nonTimeoutThresholds, "games not timed out"),
                   gradeThreshold(wins, self.winsMinimum, self.winsThresholds, "wins")]

        totalPoints = 0
        for passed, points, value, minimum, thresholds, name in results:
            if minimum == None and len(thresholds)==0:
                continue

            # print passed, points, value, minimum, thresholds, name
            totalPoints += points
            if not passed:
                assert points == 0
                self.addMessage("%s %s (fail: below minimum value %s)" % (value, name, minimum))
            else:
                self.addMessage("%s %s (%s of %s points)" % (value, name, points, len(thresholds)))

            if minimum != None:
                self.addMessage("    Grading scheme:")
                self.addMessage("     < %s:  fail" % (minimum,))
                if len(thresholds)==0 or minimum != thresholds[0]:
                    self.addMessage("    >= %s:  0 points" % (minimum,))
                for idx, threshold in enumerate(thresholds):
                    self.addMessage("    >= %s:  %s points" % (threshold, idx+1))
            elif len(thresholds) > 0:
                self.addMessage("    Grading scheme:")
                self.addMessage("     < %s:  0 points" % (thresholds[0],))
                for idx, threshold in enumerate(thresholds):
                    self.addMessage("    >= %s:  %s points" % (threshold, idx+1))

        if any([not passed for passed, _, _, _, _, _ in results]):
            totalPoints = 0

        return self.testPartial(grades, totalPoints, self.maxPoints)

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)
        handle.write('# File intentionally blank.\n')
        handle.close()
        return True




