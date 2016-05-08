import numpy as np
from IPython.display import clear_output
import random

class State:
        """ resoConsumed - resource consumed (nvm, mem)
            resoConfigured - resource configured (nvm, mem)
            actualLatency - current latency
            expectedLatency - expected latency by SLO """

	def __init__(self, resoConsumed, resoConfigured , actualLatency, expectedLatency):
                self.resoConsumed = resoConsumed
                self.resoConfigured = resoConfigured
                self.actualLatency = actualLatency
                self.expectedLatency = expectedLatency

def initQtable():
	""" The Q table is a 5-dimensional numpy array (10x5x1x1x1
	    First dimension is number of vms (1-10)
	    Second dimension is memory  utilization ranges:  0) 0% - 20%
							     1) 20% - 40%
							     2) 40% - 60%
							     3) 60% - 80%
							     4) 80% - 100%
	    Third dimension is Q value of the action - Do nothing
	    Fourth dimension is Q value of the action - Scale up
	    Fifth dimension is Q value of the action - Scale down """

	state = np.zeros((50,3))
	return state

def scaleAPI(state, action):
	return state

def currentStateAPI():
	return State( np.array([1, random.random(), random.random(), random.random(), random.random()]), 
		      np.array([10, 1, 1, 1, 1]), 2.3333, 2.444)

def getCurrentState():
	return State( np.array([ 2, 0.20, 0.10, 0.23, 0.67 ]), np.array([ 5, 0.40, 0.71, 0.30, 0.922 ]), 2.3333, 2.444)

def makeAction(state, action):
	new_state = state.deepcopy(state)

	# scale up
	if action == 1:
		# maximum number of vms is 10
		if state.resoConsumed < 10:
			#new_state.resoConsumed[0]++
			new_state = scaleAPI(state, action)
	# scale down
	elif action == 2:
		# minimum number of vms is 1
		if state.resoConsumed > 1:
			#new_state.resoConsumed[0]--
			new_state = scaleAPI(state, action)

def getReward(state):
	yNorm = state.actualLatency / float(state.expectedLatency)
	scoreSLO = np.sign( 1 - yNorm ) * np.exp( np.absolute( 1 - yNorm ) )
	vResource = np.divide(state.resoConsumed, state.resoConfigured)
	uConstrained = np.max(vResource)
	scoreU = np.exp( 1 - uConstrained )
	reward = scoreSLO * scoreU
	return reward

def heuristicPolicy(state, y):
	topSLObound = (1 - y) * float(state.expectedLatency)
	lowSLOblound = y * float(state.expectedLatency)
	
	print "Lower SLO bound: %f" % lowSLOblound
	print "Top SLO bound: %f" % topSLObound
	print "Actual Latency: %f" % state.actualLatency

	if lowSLOblound < state.actualLatency < topSLObound:
		action = 0 # no change
	elif state.actualLatency > topSLObound:
		action = 1 # scale up
	else:
		action = 2 # scale down
	
	return action 

def mapRawStateToQtableRow(state):
	# Map number of vms to Qtable vm column
	stateQtableVms = (state.resoConsumed[0].astype(int) - 1) * 5 #resoConsumed first field is number of vms

	# Map memory utilization to Qtable memory column
	if 0.0 <= state.resoConsumed[1] < 0.2:
		stateQtableMemory = 0
	elif 0.2 <= state.resoConsumed[1] < 0.4:
		stateQtableMemory = 1
	elif 0.4 <= state.resoConsumed[1] < 0.6:
                stateQtableMemory = 2
	elif 0.6 <= state.resoConsumed[1] < 0.8:
                stateQtableMemory = 3
	elif 0.2 <= state.resoConsumed[1] <= 1:
                stateQtableMemory = 4
	
	QtableRow = stateQtableVms + stateQtableMemory
	return QtableRow

def getOptimalActionQtable(mappedQtableRow,Qtable):
	
	doNothingQvalue = Qtable[mappedQtableRow, 0]
	scaleUpQvalue = Qtable[mappedQtableRow, 1]
	scaleDownQvalue	= Qtable[mappedQtableRow, 2]
	
	return np.argmax(Qtable[mappedQtableRow])


learningRate = 0.6
gamma = 0.9 
epsilon = 1
epochs = 20

Qtable = initQtable()

state = currentStateAPI()

for i in range(epochs):
	print "\nStep %d" % i 
	stateRow = mapRawStateToQtableRow(state)

	if (random.random() < epsilon): 
		# use heuristics
		action = heuristicPolicy(state, 0.3)
		print ("Performing heuristic action")
	else:
		#choose best action from Q(s,a) values
		action =  getOptimalActionQtable(stateRow, Qtable)
		print ("Performing best action from Q(s,a) values")
	
	print "Action performed %d" % action
	# get Qvalue of the Q(state, action)
	Qvalue = Qtable[stateRow, action]

	#Take action, observe new state S'
	# ---- perfom action
	# get the new state
	new_state = currentStateAPI()

	#Observe reward
	reward = getReward(new_state)
		
	new_stateRow = mapRawStateToQtableRow(new_state)
	new_stateOptimalAction = getOptimalActionQtable(new_stateRow, Qtable)

	#Get max_Q(S',a)
	newQvalue = Qtable[new_stateRow, new_stateOptimalAction]

	update = Qvalue + learningRate * ((reward + (gamma * newQvalue) - Qvalue))

	Qtable[stateRow, action] = update

	state = new_state

	if epsilon > 0.1:
        	epsilon -= (1/epochs)
	print "---------------------"
