import numpy as np
import random
import copy
import json
import urllib2
import httplib
import time
from IPython.display import clear_output

class State:
        """ resoConsumed - resource consumed (nvm, cpu, mem)
            resoConfigured - resource configured (nvm, ,cpu, mem)
            actualLatency - current latency
            expectedLatency - expected latency by SLO """

	def __init__(self, resoConsumed, resoConfigured , actualLatency, expectedLatency):
                self.resoConsumed = resoConsumed
                self.resoConfigured = resoConfigured
                self.actualLatency = actualLatency
                self.expectedLatency = expectedLatency
	
	def prt(self):
		print "-------------------"
		print self.resoConsumed
		print self.resoConfigured
		print self.actualLatency
		print self.expectedLatency

def initQtable():
	""" The Q table is a 3-dimensional numpy array
	    First dimension is number of vms (1-10)
	    Second dimension is memory utilization ranges:  0) 0% - 20%
							     1) 20% - 40%
							     2) 40% - 60%
							     3) 60% - 80%
							     4) 80% - 100%
	    Third dimension is cpu utilization ranges:  0) 0% - 20%
							     1) 20% - 40%
							     2) 40% - 60%
							     3) 60% - 80%
							     4) 80% - 100%
	    Third dimension is Q value of the action - Do nothing
	    Fourth dimension is Q value of the action - Scale up
	    Fifth dimension is Q value of the action - Scale down """

	state = np.zeros((50,3))
	return state

def getMetricsRestAPI():
	
	readSuccess = False

	while True:
		try:
			rest_response = json.load(urllib2.urlopen("http://quickscaling.ml/GetLatestData"))
			break;
		except (httplib.HTTPException, httplib.IncompleteRead, urllib2.URLError):
			print "Reading API has failed, retrying"

        cState = State( np.array([rest_response['replicas'], rest_response['cpu'], rest_response['ram']]),
                        np.array([10, rest_response['MaxCpu'], rest_response['MaxRam']]),
                        rest_response['responseTime'],
                        1500)
	return cState

def getCurrentState():
	resoConsumedArray = np.array([[1, 150, 150],
				      [2, 150, 150],
				      [2, 140, 150],
				      [2, 150, 150],
				      [2, 150, 150],
				      [2, 150, 150],
				      [2, 150, 150],
				      [2, 150, 150],
				      [2, 150, 150],
				      [2, 150, 150],
				      [2, 150, 150],
				      [2, 150, 150],
				      [2, 150, 150],
				      [2, 150, 150],
				      [2, 150, 150],
				      [2, 150, 150],
				      [2, 150, 150],
				      [2, 150, 150],
				      [2, 150, 150],
				      [2, 150, 150],
				      [2, 150, 150]])
	
	resoConfiguredArray = np.array([[10, 200, 200],
				        [10, 200, 200],		
				        [10, 200, 200],		
				        [10, 200, 200],		
				        [10, 200, 200],		
				        [10, 200, 200],		
				        [10, 200, 200],		
				        [10, 200, 200],		
				        [10, 200, 200],		
				        [10, 200, 200],		
				        [10, 200, 200],		
				        [10, 200, 200],		
				        [10, 200, 200],		
				        [10, 200, 200],		
				        [10, 200, 200],		
				        [10, 200, 200],		
				        [10, 200, 200],		
				        [10, 200, 200],		
				        [10, 200, 200],		
				        [10, 200, 200],		
				        [10, 200, 200]])

	actualLatencyArray = np.array([6,
				       5.6,
				       5.2,
				       5.3,
				       5.3,
				       5,
				       4.7,
				       4.6,
				       4.1,
				       3.9,
				       3.0,
				       2.9,
				       2.5,
				       2.4,
				       2.2,
				       2.4,
				       2.5,
				       2.5,
				       2.4,
				       2.3,
				       2.5])
	
	expectedLatency = 2.5
	
	print "Current counter %d" % getCurrentState.counter	

	cState = State( resoConsumedArray[getCurrentState.counter], 
			resoConfiguredArray[getCurrentState.counter], 
			actualLatencyArray[getCurrentState.counter],
			expectedLatency)
	getCurrentState.counter += 1

	return cState

def scaleAPI(state,action):
	numOfReplicas = state.resoConsumed[0]
	
	 # scale up
        if action == 1:
		numOfReplicas += 1
        # scale down
        elif action == 2:
		numOfReplicas -= 1
	
	
	if action != 0:
		data = {'spec': {'replicas': numOfReplicas}}
		req = urllib2.Request('http://kube.quickscaling.ml/api/v1/namespaces/default/replicationcontrollers/geoserver-controller')
		req.get_method = lambda: 'PATCH'
		req.add_header('Content-Type', 'application/merge-patch+json')

		response = urllib2.urlopen(req, json.dumps(data))
	
	print "Performing  %d action" % action

def makeAction(state, action):

	# scale up
	if action == 1:
		# maximum number of vms is 10
		if state.resoConsumed[0] < 10:
			scaleAPI(state, action)
	# scale down
	elif action == 2:
		# minimum number of vms is 1
		if state.resoConsumed[0] > 1:
			scaleAPI(state, action)

def getReward(state):
	yNorm = float(state.actualLatency) / float(state.expectedLatency)
	print "actualLatency:%f / expectedLatency:%f ratio %f" % (state.actualLatency, state.expectedLatency, yNorm)

	scoreSLO = np.sign( 1 - yNorm ) * np.exp( np.absolute( 1 - yNorm ) )
	print "SLO score: %f" % scoreSLO

	vResource = np.true_divide(state.resoConsumed, state.resoConfigured)
	print "state.resoConsumed:"
	print state.resoConsumed
	print "state.resoConfigured:"
	print state.resoConfigured
	print "state.resoConsumed / state.resoConfigured:"
	print vResource

	uConstrained = np.max(vResource)
	print "uConstrained max(vResource): %f" % uConstrained

	scoreU = np.exp( 1 - uConstrained )
	print "scoreU: %f" % scoreU

	reward = scoreSLO * scoreU
	print "Reward= scoreSLO(%f) * scoreU(%f) = %f" % (scoreSLO,scoreU,reward)

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
	
	resourceRetio = np.true_divide(state.resoConsumed, state.resoConfigured)

	# Map CPU utilization to Qtable memory column
	if 0.0 <= resourceRetio[1] < 0.2:
		stateQtableCPU = 0
	elif 0.2 <= resourceRetio[1] < 0.4:
		stateQtableCPU = 1
	elif 0.4 <= resourceRetio[1] < 0.6:
                stateQtableCPU = 2
	elif 0.6 <= resourceRetio[1] < 0.8:
                stateQtableCPU = 3
	elif 0.2 <= resourceRetio[1] <= 1:
                stateQtabeCPU = 4
	else:
		stateQtableCPU = 0
	
	QtableRow = stateQtableVms + stateQtableCPU	
	print "Mapping [%f, %f, %f] to rowNum: %d" % (resourceRetio[0], resourceRetio[1], resourceRetio[2], QtableRow)

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
i = 0

Qtable = initQtable()

getCurrentState.counter = 0

state = getMetricsRestAPI() #getCurrentState()

#for i in range(epochs):
while True:
	i += 1
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
	makeAction(state, action)
	
	# Wait 1 minute
	print "Sleeping for 1 minute"
	time.sleep(60)

	# get the new state
	new_state = getMetricsRestAPI() #getCurrentState()
	print "New state:"
	print new_state.prt()

	#Observe reward
	reward = getReward(new_state)
		
	new_stateRow = mapRawStateToQtableRow(new_state)
	new_stateOptimalAction = getOptimalActionQtable(new_stateRow, Qtable)

	#Get max_Q(S',a)
	newQvalue = Qtable[new_stateRow, new_stateOptimalAction]

	update = Qvalue + learningRate * ((reward + (gamma * newQvalue) - Qvalue))

	Qtable[stateRow, action] = update

	state = new_state
	
	print Qtable

	if epsilon > 0.1:
        	epsilon -= (1/epochs)
	print "---------------------"
