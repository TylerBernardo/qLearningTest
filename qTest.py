import tensorflow as tf;
import keras;
import math;
from random import random,sample;
from tf_agents.environments import suite_gym
from copy import deepcopy
from numpy import max;
import imageio
import matplotlib.pyplot as plt


#Create a layer with the given number of nodes
def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal')
          ,bias_initializer=tf.keras.initializers.Constant(-0.2))

#wrapper to handle constructing network
class QNetwork:
    q_net : keras.Sequential = None;
    def __init__(self,inputHeight : int, hiddenLayers : list[int], outputLayer:int):
        #Build the input layer
        inputLayer = tf.keras.layers.Dense(hiddenLayers[0],activation=None,kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
        ,input_shape=(inputHeight,))
        #make the dense layers based on the "hiddenLayers" list
        dense_layers = [dense_layer(num_units) for num_units in  hiddenLayers[1:]]
        #Make the output layer
        q_values_layer = tf.keras.layers.Dense(outputLayer,activation=None,kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),bias_initializer=tf.keras.initializers.Constant(-0.2))
        #buid a sequential network based on the layers
        self.q_net = keras.Sequential([inputLayer] + dense_layers + [q_values_layer])
    #def __init__(self,filePath : str):
     #   self.q_net = keras.models.load_model(filePath)
    #converts a python list into a tensor in the correct shape, and passes that to the network
    def calc(self, inputs : list[int]) -> tf.Tensor:
        output = self.q_net(tf.reshape(tf.convert_to_tensor(inputs, dtype=tf.float32),shape=(1,len(inputs))));
        return output;
    def save(self,filename : str) -> None:
        self.q_net.save("./" + filename + ".keras")

#A class to hold all the experiences
class ExperienceBuffer:
    #set the size of the buffer and initilize an array to add experiences to
    def __init__(self,size):
        self.size = size;
        self.experiences = []
    #Add an experience to the experience buffer
    def addToExperience(self,toAdd):
        #check if the buffer is full
        if(len(self.experiences) == self.size):
            #if the buffer is full, get rid of the oldest experience
            self.experiences = self.experiences[1:] + [toAdd];
        else:
            #if the buffer isnt full, just append it to the array
            self.experiences.append(toAdd);
    #Sample an ammount of element from the buffer
    def sample(self,ammount):
        return sample(self.experiences,ammount)


#calculate the mean loss for a given set of experiences
def batchLoss(experiences,network,discount,optimalNetwork):
    #extract the relevant information from the experiences
    states = tf.convert_to_tensor([experience[0] for experience in experiences])
    actions = tf.convert_to_tensor([experience[1] for experience in experiences])
    rewards = tf.convert_to_tensor([experience[2] for experience in experiences])
    nextStates = tf.convert_to_tensor([experience[3] for experience in experiences])
    finalStates = tf.convert_to_tensor([experience[4] for experience in experiences],tf.float32)
    #use the current network to calculate the q_values for all states in the experience list
    q_Values = network.q_net(states)
    #make a list of the q values for all actions taken
    experienced_action_values =tf.convert_to_tensor([q_Values[i,actions[i]] for i in range(len(actions))]);
    #make a list of the expected q_values for each state, removing the reward of the next state if the current state is a final state
    expected_values = rewards + discount * (max(optimalNetwork.q_net(nextStates), axis = 1) *  (1-finalStates))
    #calculate the loss for each experience from the experience list, clamping the error to the interval [-1,1]
    losses = tf.math.square(tf.clip_by_value(expected_values - experienced_action_values,-1,1))
    #return the squared sum of the losses
    return tf.math.reduce_sum((losses))/len(experiences);

#update the weights of the network by sampling from the experience buffer and updating the network based on the loss
def updateWeights(experience,network,discount,optimalNetwork,optimizer):
    meanLoss = 0
    #pull a sample of experience from the buffer
    experienceToTrain = experience.sample(128)
    #calculate the loss, tracking the process with a GradientTape
    with tf.GradientTape() as tape:
        meanLoss = batchLoss(experienceToTrain,network,discount,optimalNetwork)
    #calculate the gradient with the mean loss
    grads = tape.gradient(meanLoss,network.q_net.trainable_weights)
    #apply the gradients with the given optimizer
    optimizer.apply_gradients(zip(grads, network.q_net.trainable_weights))
    #return the mean loss
    return meanLoss;
#Function to evaluate the network using a 100% greedy policy for a given number of episodes
def evalF(env,maxStepsPerEval,network,num_episodes=5):
    meanReward = 0;
    #Go through the correct number of episodes
    for _ in range(num_episodes):
        #keep track of variables
        reward = 0;
        time_step = env.reset()
        #add the current frame to the video
        for s in range(maxStepsPerEval):
            toAdd = [None] * 4;
            #Get the current observation from the env
            toAdd[0] = time_step.observation
            #Calculate the q_values for this state
            q_values = network.calc(time_step.observation)
            #find out which action has a higher q_value
            if(q_values[0,0] > q_values[0,1]):
                toAdd[1] = 0;
            else:
                toAdd[1] = 1;
            #take the action
            time_step = env.step(toAdd[1])
            #get the reward
            toAdd[2] = time_step.reward;
            reward = reward + toAdd[2]
            #quit if this is the last time step
            if(time_step.is_last()):
                break;
        meanReward += reward/num_episodes; 
    #print the average reward for debugging
    print("Eval mean reward: " + str(meanReward));
    #return the mean rewwad
    return meanReward
#run a test of the network only using the greedy policy and save it as a video for manual review
def makeVideo(env,maxStepsPerEval,network,num_episodes=5,fps=30,filename = "eval.mp4" ):
    meanReward = 0;
    #Create a video writer
    with imageio.get_writer(filename, fps=fps) as video:
        #Go through the correct number of episodes
        for _ in range(num_episodes):
            #keep track of variables
            reward = 0;
            time_step = env.reset()
            #add the current frame to the video
            video.append_data(env.render())
            for s in range(maxStepsPerEval):
                toAdd = [None] * 4;
                #Get the current observation from the env
                toAdd[0] = time_step.observation
                        #calc here
                #Calculate the q_values for this state
                q_values = network.calc(time_step.observation)
                #find out which action has a higher q_value
                if(q_values[0,0] > q_values[0,1]):
                    toAdd[1] = 0;
                else:
                    toAdd[1] = 1;
                #take the action
                time_step = env.step(toAdd[1])
                #get the reward
                toAdd[2] = time_step.reward;
                reward = reward + toAdd[2]
                #Add the next frame to the video
                video.append_data(env.render())
                #quit if this is the last time step
                if(time_step.is_last()):
                    break;
            meanReward += reward/num_episodes; 
    #print the average reward for debugging
    print("Eval mean reward: " + str(meanReward));
    return meanReward

def trainTest(epochs : int, evalsPerEpoch : int, discount : float, learnRate : float, maxStepsPerEval : int, videoInterval : int):
    #training variables
    losses = []
    rewards = []
    optimizer = keras.optimizers.Adam(learning_rate=learnRate)
    e = 1;
    decay = .0005;
    minE = 0.01;
    env_name = 'CartPole-v0'
    env = suite_gym.load(env_name)
    network = QNetwork(4,[64],2)#QNetwork("./best.keras")#QNetwork(4,[128,64],2)
    optimalNetwork = deepcopy(network)
    changeOptimal = 50;
    steps = 0
    experience = ExperienceBuffer(10000)
    best = 0;
    #go for a given number of epochs
    for epoch in range(epochs):
        meanReward = 0;
        #collect experience
        for eval in range(evalsPerEpoch):
            #reset the enviornment
            time_step = env.reset()
            episodeReward = 0;
            #Keep going until we hit the max number of steps
            for s in range(maxStepsPerEval):
                #create a list to store this step's experience
                toAdd = [None] * 5;
                #save the observation to the experiene list
                toAdd[0] = time_step.observation
                #calculate the q_values given the current state.
                q_values = network.calc(time_step.observation)
                #use epsilon greedy policy to decide whether to take a random action or act based on the q_values
                if(random() <= e):
                    toAdd[1] = round(random())
                else:
                    if(q_values[0,0] > q_values[0,1]):
                        toAdd[1] = 0;
                    else:
                        toAdd[1] = 1;
                #take the action we decided on earlier
                time_step = env.step(toAdd[1])
                #save the reward for that action to the experience list
                toAdd[2] = time_step.reward;
                episodeReward += toAdd[2];
                #save the next state to the experience list
                toAdd[3] = time_step.observation
                #increment steps
                steps = steps + 1;
                #calculate the new epsilon value based on the number of steps that have taken place
                e = minE + (1 - minE) * math.exp(-decay * steps)
                #by default, set the "finished" value to 0(false)
                toAdd[4] = 0
                #check if enough experience has been gathered
                if(steps > 128):
                    #update the weights by sampling from the experience
                    loss = updateWeights(experience,network,discount,optimalNetwork,optimizer)
                    #Append the current loss to the list of losses for graphing later
                    if(steps%20 == 0):
                        losses.append(loss)
                    #change the target network every 10000 steps
                    if((steps)%10000 == 0):
                        optimalNetwork = deepcopy(network)
                #check if this is the last time step
                if(time_step.is_last()):
                    #set "finished" value to 1(true)
                    toAdd[4] = 1;
                    #add the experience to the experience buffer
                    experience.addToExperience(toAdd);
                    #exit the loop
                    break;
                #add the experience to the experience buffer
                experience.addToExperience(toAdd);
            meanReward += episodeReward/evalsPerEpoch
        #if the current epoch is divisible by the videoInterval, save a video of the agent's performance
        if((epoch+1) % videoInterval == 0):
            rewards.append(evalF(env,maxStepsPerEval,network,num_episodes=10))
            #check if this network performed the best so far
            if(rewards[-1] > best):
                #save the best network
                best = rewards[-1]
                network.save("best")
        #print an update to the console
        print(str(epoch+1) + ")" + "Mean Reward:" + str(meanReward) + " Total steps: " + str(steps))
    #make one final video at the end of evaluation
    makeVideo(env,maxStepsPerEval,network,num_episodes=20,fps=30,filename="eval.mp4")
    #plot the loss versus steps
    xs = [20*x for x in range(len(losses))]
    p1 = plt.figure(1)
    plt.plot(xs, losses)
    #plot the rewards vs epochs
    p2 = plt.figure(2)
    xs2 = [videoInterval * x for x in range(len(rewards))]
    plt.plot(xs2,rewards)
    #save the network
    network.save("final")
    plt.show()
    # Make sure to close the plt object once done
    plt.close()
    


trainTest(100,20,.9,.0001,1000,5)
