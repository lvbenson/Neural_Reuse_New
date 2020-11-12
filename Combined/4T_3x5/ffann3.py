import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def relu(x):
    return x * (x > 0)

class ANN:

    def __init__(self, NIU, NH1U, NH2U, NH3U, NOU):
        self.nI = NIU
        self.nH1 = NH1U
        self.nH2 = NH2U
        self.nH3 = NH3U
        self.nO = NOU
        self.wIH1 = np.zeros((NIU,NH1U))
        self.wH1H2 = np.zeros((NH1U,NH2U))
        self.wH2H3 = np.zeros((NH2U,NH3U))
        self.wH3O = np.zeros((NH3U,NOU))
        self.bH1 = np.zeros(NH1U)
        self.bH2 = np.zeros(NH2U)
        self.bH3 = np.zeros(NH3U)
        self.bO = np.zeros(NOU)
        self.Hidden1Activation = np.zeros(NH1U)
        self.Hidden2Activation = np.zeros(NH2U)
        self.Hidden3Activation = np.zeros(NH3U)
        self.OutputActivation = np.zeros(NOU)
        self.Input = np.zeros(NIU)

    def setParameters(self, genotype, WeightRange, BiasRange):
         k = 0
         for i in range(self.nI):
             for j in range(self.nH1):
                 self.wIH1[i][j] = genotype[k]*WeightRange
                 k += 1
         for i in range(self.nH1):
             for j in range(self.nH2):
                 self.wH1H2[i][j] = genotype[k]*WeightRange
                 k += 1
         for i in range(self.nH2):
             for j in range(self.nH3):
                 self.wH2H3[i][j] = genotype[k]*WeightRange
                 k += 1
         for i in range(self.nH3):
             for j in range(self.nO):
                 self.wH3O[i][j] = genotype[k]*WeightRange
                 k += 1
         for i in range(self.nH1):
             self.bH1[i] = genotype[k]*BiasRange
             k += 1
         for i in range(self.nH2):
             self.bH2[i] = genotype[k]*BiasRange
             k += 1
         for i in range(self.nH3):
             self.bH3[i] = genotype[k]*BiasRange
             k += 1
         for i in range(self.nO):
             self.bO[i] = genotype[k]*BiasRange
             k += 1

    def step(self,Input):
        self.Input = np.array(Input)
        self.Hidden1Activation = relu(np.dot(self.Input.T,self.wIH1)+self.bH1)
        self.Hidden2Activation = relu(np.dot(self.Hidden1Activation,self.wH1H2)+self.bH2)
        self.Hidden3Activation = relu(np.dot(self.Hidden2Activation,self.wH2H3)+self.bH3)
        self.OutputActivation = sigmoid(np.dot(self.Hidden3Activation,self.wH3O)+self.bO)
        return self.OutputActivation
    """

    def step_lesioned(self,Input,Neuron,Layer,Activation):
        self.Input = np.array(Input)
        self.Hidden1Activation = relu(np.dot(self.Input.T,self.wIH1)+self.bH1)
        if (Layer==1):
            self.Hidden1Activation[Neuron] = Activation
        self.Hidden2Activation = relu(np.dot(self.Hidden1Activation,self.wH1H2)+self.bH2)
        if (Layer==2):
            self.Hidden2Activation[Neuron] = Activation
        self.OutputActivation = sigmoid(np.dot(self.Hidden2Activation,self.wH2O)+self.bO)
        return self.OutputActivation
    """

    def output(self):
        return self.OutputActivation*2 - 1

    def states(self):
        return np.concatenate((self.Input,self.Hidden1Activation,self.Hidden2Activation,self.Hidden3Activation,self.OutputActivation))
