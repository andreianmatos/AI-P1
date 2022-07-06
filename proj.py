#89484 Joao Ribeiro Dias, 89413 Andreia Nogueira de Matos, Grupo 74

import random

# LearningAgent to implement
# no knowledeg about the environment can be used
# the code should work even with another environment
class LearningAgent:

        # init
        # nS maximum number of states
        # nA maximum number of action per state
        def __init__(self,nS,nA):
                self.nS = nS
                self.nA = nA
                self.Q = [[]]*nS
                self.gamma = 0.90
                self.alpha_min = 0.1
                self.epsilon = 0.3
              
        # Select one action, used when learning  
        # st - is the current state        
        # aa - is the set of possible actions
        # for a given state they are always given in the same order
        # returns
        # a - the index to the action in aa
        def selectactiontolearn(self,st,aa):
                q_count = len(aa)

                if not self.Q[st]:
                    self.Q[st] = [[0]*q_count , [0]*q_count , []]
                    self.Q[st][2].extend(range(q_count))

                if self.Q[st][2]:
                    a = self.Q[st][2].index(random.choice(self.Q[st][2]))
                    del self.Q[st][2][a]
                elif(random.uniform(0,1) > self.epsilon):
                    maximum = max(self.Q[st][0])
                    max_lst = []
                    for i in range(q_count):
                        if self.Q[st][0][i] == maximum:
                            max_lst.append(i)
                    a = random.choice(max_lst)
                else:
                    a = aa.index(random.choice(aa))

                return a

        # Select one action, used when evaluating
        # st - is the current state        
        # aa - is the set of possible actions
        # for a given state they are always given in the same order
        # returns
        # a - the index to the action in aa
        def selectactiontoexecute(self,st,aa):
                if not self.Q[st]:
                    return aa.index(random.choice(aa))

                a = self.Q[st][0].index(max(self.Q[st][0]))

                return a


        # this function is called after every action
        # st - original state
        # nst - next state
        # a - the index to the action taken
        # r - reward obtained
        def learn(self,ost,nst,a,r):
                if self.Q[nst]:
                    nst_max = max(self.Q[nst][0])
                else:
                    nst_max = 0

                self.Q[ost][1][a] += 1

                new_alpha = 1 / (1 + 0.2*self.Q[ost][1][a])
                if(new_alpha < self.alpha_min):
                    new_alpha = self.alpha_min

                self.Q[ost][0][a] = self.Q[ost][0][a] + new_alpha*(r + self.gamma*nst_max - self.Q[ost][0][a])
                return