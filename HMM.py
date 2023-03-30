# Hidden Markov Model (HMM)
# Author: Zell Wu
# Date: 3/28/2023

import numpy as np


V = [1, 2, 3]
Q = [1, 2, 3]
Pi = np.array([[.2], [.4], [.4]])
A = np.array([[.5, .2, .3], [.3, .5, .2], [.2, .3, .5]])
B = np.array([[.3, .5, .2], [.2, .3, .5], [.5, .2, .3]])
O = [1, 2, 1, 3, 2, 3, 3, 3, 2, 1, 1, 1, 2, 3, 2, 2]


class HMM:
    """ Hidden Markov Model """
    def __init__(self, V, Q, Pi, A, B, O, iteration=1000):
        """ HMM: The forward algorithm
            V: a set of observations
            Q: a set of hidden states
            Pi: an initial probability distribution over states
            A:  a transition probability matrix
            B:  a sequence of observation likelihoods, also called emission probabilities
            O:  a sequence of observations
            (observations of len T, state-graph of len N)
        """
        self.Pi, self.A, self.B, self.O = Pi, A, B, O
        self.V, self.Q = V, Q
        self.M = self.B.shape[1]                        # number of set of observations
        self.N = self.A.shape[0]                        # number of states
        self.T = len(self.O)                            # number of observations
        self.iter = iteration

    def states_back(self, bestpathpointer, backpointer, T):
        """ The path starting at state bestpathpointer, that follows
            backpointer[] to states back in time.
        """
        # Set recursion exit condition
        if T >= backpointer.shape[1] - 1:
            bestpath = [int(bestpathpointer)]
            return bestpath
        T += 1                                                          # iteration
        # Get the back point
        bestpath = self.states_back(bestpathpointer, backpointer, T)    # recursion
        bestpath.append(int(backpointer[bestpath[-1], T]))              # update
        return bestpath

    def forward_prob(self):
        """ Get forward probabilities """
        # create a probability matrix forward[N,T]
        alpha = np.zeros((self.N, self.T))
        for ind in range(self.N):                                       # initiation
            alpha[ind, 0] = self.Pi[ind] * self.B[ind, self.V.index(self.O[0])]
        # time loop
        for time in range(1, self.T):                                   # recursion
            # state-graph loop
            for ind in range(self.N):
                alpha[ind, time] = alpha[:, time - 1].T.dot(self.A[:, ind]) * self.B[ind, self.V.index(self.O[time])]
        prob = sum(alpha[:, self.T - 1])                                # termination
        return alpha, prob

    def backward_prob(self):
        """ Get backward probabilities """
        # create a probability matrix backward[N,T]
        beta = np.zeros((self.N, self.T))
        for ind in range(self.N):                                       # initiation
            beta[ind, -1] = 1
        # time loop
        for time in range(self.T-2, -1, -1):                            # recursion
            # state-graph loop
            for ind in range(self.N):
                beta[ind, time] = beta[:, time + 1].T.dot(self.A[:, ind] * self.B[:, self.V.index(self.O[time+1])])
        prob = (self.Pi[:, ] * beta[:, 0]).T.dot(self.B[:, self.V.index(self.O[0])])        # termination
        return beta, prob

    def decoding(self):
        """ Get the sequence of states """
        # create a path probability matrix viterbi[N,T]
        viterbi = np.zeros((self.N, self.T))
        backpointer = np.zeros((self.N, self.T))
        for ind in range(self.N):                                       # initiation
            viterbi[ind, 0] = self.Pi[ind] * B[ind, self.V.index(O[0])]
        # time loop
        for time in range(1, self.T):                                   # recursion
            # state loop
            for ind in range(self.N):
                viterbi[ind, time] = max(viterbi[:, time - 1] * self.A[:, ind] * self.B[ind, self.V.index(self.O[time])])
                backpointer[ind, time] = np.argmax(viterbi[:, time - 1] * self.A[:, ind] * self.B[ind, self.V.index(self.O[time])])
        bestpathprob = np.max(viterbi[:, self.T - 1], axis=0)           # termination
        bestpathpointer = np.argmax(viterbi[:, self.T - 1], axis=0)
        bestpath = self.states_back(bestpathpointer, backpointer, T=0)
        return bestpath, bestpathprob, viterbi

    def gamma_prob(self, alpha, beta, forward_val):
        """ Get gamma probabilities """
        # initiation
        prob = np.zeros((self.N, self.T))
        # Computation
        for time in range(self.T):
            for state in range(self.N):
                prob[state, time] = ( alpha[state, time] * beta[state, time] ) / forward_val
        return prob

    def xi_prob(self, alpha, beta, forward_val):
        """ Get xi probabilities """
        # initiation
        prob = np.zeros((self.T-1, self.N, self.N))
        # Computation
        for time in range(self.T-1):
            for i in range(self.N):
                for j in range(self.N):
                    prob[time, i, j] = (alpha[i, time] * self.A[i, j] * self.B[j, self.V.index(self.O[time+1])] * beta[j, time+1]) / forward_val
        return prob

    def forward_backward(self, *args):
        """ Learning the parameters of an HMM """
        # initialize A and B
        if len(args) == 2:
            self.A, self.B = args[0], args[1]
        elif len(args) == 0:
            pass
        else:
            raise SyntaxError
        self.A, self.B = self.A, self.B
        # Iteration
        for iteration in range(self.iter):
            print('\nIteration No: ', iteration + 1)
            # E-step
            alpha, forward_val = self.forward_prob()
            beta, _ = self.backward_prob()
            gamma = self.gamma_prob(alpha, beta, forward_val)
            xi = self.xi_prob(alpha, beta, forward_val)
            # M-step
            # calculating A and B matrices
            A = np.zeros((self.N, self.N))
            B = np.zeros((self.N, self.M))
            # Matrix A
            for i in range(self.N):
                for j in range(self.N):
                    for time in range(self.T-1):
                        A[i, j] = A[i, j] + xi[time, i, j]
                    denominator_a = [xi[t_x, i, j_x] for t_x in range(self.T-1) for j_x in range(self.N)]
                    denominator_a = sum(denominator_a)
                    if denominator_a == 0:
                        A[i, j] = 0
                    else:
                        A[i, j] = A[i, j] / denominator_a
            # Matrix B
            for v in range(len(self.V)):
                for state in range(self.N):
                    indices = [ind for ind, val in enumerate(self.O) if val == self.V[v]]
                    numerator_b = sum(gamma[state, indices])
                    denominator_b = sum(gamma[state, :])
                    if denominator_b == 0:
                        B[state, v] = 0
                    else:
                        B[state, v] = numerator_b / denominator_b
            # Judgement
            self.A, self.B = A, B
            _, new_forward_val = self.forward_prob()
            print('New forward probability: ', new_forward_val)
            diff = np.abs(forward_val - new_forward_val)
            print('Difference in forward probability: ', diff)

            if diff < 0.00001:
                return self.A, self.B
        # return self.A, self.B
