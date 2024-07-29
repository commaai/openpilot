import numpy as np

class StateSpace:
    def __init__(self, A, B, C, D):
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        self.D = np.array(D)

    def sample(self, dt):
        """
        Discretize the continuous-time state-space system using zero-order hold.
        """
        n = self.A.shape[0]
        Ad = np.eye(n) + dt * self.A
        Bd = dt * self.B
        return StateSpace(Ad, Bd, self.C, self.D)

    def update(self, x, u):
        """
        Update the state `x` with input `u`.
        x: current state
        u: input
        returns: next state
        """
        return self.A @ x + self.B @ u

    def output(self, x, u):
        """
        Calculate the output `y` from state `x` and input `u`.
        x: current state
        u: input
        returns: output
        """
        return self.C @ x + self.D @ u
