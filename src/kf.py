import numpy as np

# Reference implementation taken from: https://github.com/zziz/kalman-filter
class KalmanFilter:
    def __init__(self, F=None, B=None, H=None, Q=None, R=None, P=None, x0=None, state_size=None, observation_size=None):
        """
        Constructs a new instance.

        :param      F:    State transition matrix, models linear dynamics and advances state vector from t to t+1
        :type       F:    np.array
        :param      B:    Control input model, maps control vectors into state space
        :type       B:    np.array
        :param      H:    Observation matrix, maps state space into observation space
        :type       H:    np.array
        :param      Q:    Process noise covariance, models uncertainty in the state dynamics
        :type       Q:    np.array
        :param      R:    Observation noise covariance, models uncertainty in the observations
        :type       R:    np.array
        :param      P:    State estimate covariance, models uncertainty in the state estimate
        :type       P:    np.array
        :param      x0:   State vector capturing initial conditions
        :type       x0:   np.array
        :param      state_size:   Size of the state vector
        :type       state_size:   int
        :param      observation_size:   Size of the observation vector
        :type       observation_size:   int
        """
        if state_size is not None and observation_size is not None:
            # For users who would rather specify matrices later
            self.n = state_size
            self.m = observation_size
            self.F = np.eye(self.n) if F is None else F  # state transition matrix
            self.H = np.eye(self.m) if H is None else H  # observation matrix
        elif F is None or H is None:
            raise ValueError("Set proper system dynamics.")
        else:
            self.n = F.shape[1]  # state size
            self.m = H.shape[0]  # observation size

        self.F = F  # state transition matrix
        self.H = H  # observation matrix
        self.B = 0 if B is None else B  # control input model
        self.Q = np.eye(self.n) if Q is None else Q  # process noise covariance
        self.R = np.eye(self.n) if R is None else R  # observation noise covariance
        self.P = np.eye(self.n) if P is None else P  # state estimate covariance
        self.x = np.zeros((self.n, 1)) if x0 is None else x0  # state

    def predict(self, u=0):
        self.x = (self.F @ self.x) + (self.B @ u)
        self.P = ((self.F @ self.P) @ self.F.T) + self.Q
        return self.x

    def update(self, z):
        self.y_pre = z - (self.H @ self.x)  # pre-fit residual
        S = self.R + (self.H @ (self.P @ self.H.T))  # pre-fit cov residual
        K = (self.P @ self.H.T) @ np.linalg.inv(S)  # kalman gain
        self.x = self.x + (K @ self.y_pre)
        I = np.eye(self.n)  # noqa: E741
        self.P = (I - (K @ self.H)) @ self.P
        self.y_post = z - (self.H @ self.x)  # post-fit residual
        return self.x