import numpy as np

class StateSpace:
    def __init__(self, A, B, C, D):
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        self.D = np.array(D)

    def simulate(self, x0, u, t):
        """
        Simulates the state-space system.
        
        Parameters:
        x0: Initial state vector
        u: Input vector (or matrix if multiple inputs over time)
        t: Time vector
        
        Returns:
        x: State vector over time
        y: Output vector over time
        """
        x = np.zeros((len(t), len(x0)))
        y = np.zeros((len(t), self.C.shape[0]))
        
        x[0, :] = x0
        
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            x[i, :] = x[i-1, :] + dt * (self.A @ x[i-1, :] + self.B @ u[i-1, :])
            y[i, :] = self.C @ x[i, :] + self.D @ u[i, :]
        
        return x, y

    def sample(self, dt):
        """
        Convert the system to a discrete-time system using zero-order hold.
        """
        n = self.A.shape[0]
        m = self.B.shape[1]

        M = np.zeros((n + m, n + m))
        M[:n, :n] = self.A * dt
        M[:n, n:] = self.B * dt
        M[n:, n:] = np.eye(m)

        Md = np.linalg.matrix_power(M, 1)
        Ad = Md[:n, :n]
        Bd = Md[:n, n:]

        return StateSpace(Ad, Bd, self.C, self.D)
