"""
This file defines the Extended Kalman Filter for state estimation for the rocket's
suite of sensors and attributes.
"""

import numpy as np
import sympy
from pyquaternion import Quaternion
from filterpy.kalman import ExtendedKalmanFilter as EKF

import rocket_math as rm
import sensors

# ---------------------NUMPY PRINT OPTIONS---------------------
# 4 decimal variable precision and disable scientific notation:
np.set_printoptions(precision=4, suppress=True)
# -------------------------------------------------------------

# ---------------------EKF DIMENSIONS---------------------
# Number of fields for all of the attributes of the rocket
ROCKET_STATES_DIM = 18
# Number of fields for all of the sensor measurements
SENSOR_FIELDS_DIM = 11

# -------------------------------------------------------------


class RocketEKF(EKF):
    def __init__(self, timestep=0.01, std_vel=1, std_steer=1):
        EKF.__init__(dim_x=ROCKET_STATES_DIM, dim_z=SENSOR_FIELDS_DIM)
        self.timestep = timestep
        self.std_vel = std_vel
        self.std_steer = std_steer

        a, x, y, v, w, theta, time = sympy.symbols(
            'a, x, y, v, w, theta, t')
        d = v * time
        beta = (d / w) * sympy.tan(a)
        r = w / sympy.tan(a)

        self.fxu = np.array(
            [[x - r * sympy.sin(theta) + r * sympy.sin(theta + beta)],
             [y + r * sympy.cos(theta) - r * sympy.cos(theta + beta)],
             [theta + beta]])

        self.F_j = self.fxu.jacobian(np.array([x, y, theta]))
        self.V_j = self.fxu.jacobian(np.array([v, a]))

        # save dictionary and it's variables for later use
        self.subs = {x: 0, y: 0, v: 0, a: 0,
                     time: timestep, theta: 0}
        self.x_x, self.x_y, = x, y
        self.v, self.a, self.theta = v, a, theta

    def predict(self, u):
        # include predict step here

        self.subs[self.theta] = self.x[2, 0]
        self.subs[self.v] = u[0]
        self.subs[self.a] = u[1]

        F = np.array(self.F_j.evalf(subs=self.subs)).astype(float)
        V = np.array(self.V_j.evalf(subs=self.subs)).astype(float)

        # covariance of motion noise in control space
        M = np.array([[self.std_vel * u[0] ** 2, 0],
                      [0, self.std_steer ** 2]])

        self.P = F @ self.P @ F.T + V @ M @ V.T
