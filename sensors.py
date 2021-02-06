"""
This file defines multiple classes, one for each sensor. Each class has an
update function that takes a rocket as a parameter, which it performs updates on.
"""

import numpy as np
from pyquaternion import Quaternion


class Sensor:
    """
    A class used to represent a Sensor object.

    ...

    Attributes
    ----------

    calibration: float
    """

    def __init__(self, calibration=1):
        self.calibration = calibration


class Accelerometer(Sensor):
    def update(self, rocket):
        """
        Calculates the body (proper) acceleration of the rocket

        Parameters
        ----------
        rocket: A rocket object
            An rocket instance. Calculations will be done according to this rocket's state

        Returns
        -------
        body_acceleration
            Numpy array representing the body (proper) acceleration
            of the rocket in m/s^2.
        """

        quaternion = Quaternion(rocket.orientation)
        body_acceleration = quaternion.rotate(rocket.world_acceleration * self.calibration)
        return body_acceleration


class Baro_Pressure_Sensor(Sensor):
    def update(self, rocket):
        """
        Calculates the barometric pressure of the atmosphere around the Rocket.
        Uses NASA formulas:
        https://www.grc.nasa.gov/WWW/K-12/rocket/atmosmet.html

        Parameters
        ----------
        rocket: A rocket object
            An rocket instance. Calculations will be done according to this rocket's state

        Returns
        ---------
        baro_pressure: float
            Barometric pressure of the atmosphere around the rocket, in kilopascals.
        """
        # Use NASA formula to calculate barometric pressure
        if rocket.altitude < 11000:
            baro_pressure = 101.29 * (
                    (rocket.temperature + 273.1) / 288.08) ** 5.256
        else:
            baro_pressure = 22.65 * np.exp(1.73 - 0.000157 * rocket.altitude)

        return baro_pressure * self.calibration


class Gyro(Sensor):
    def update(self, rocket, angular_rates, delta_time) -> np.array([]):
        """
        Calculates the orientation quaternion of the Rocket object based on
        fixed angular rates.

        Parameters
        ----------
        rocket: A rocket object
            An rocket instance. Calculations will be done according to this rocket's state

        delta_time: float
            The change in time since the last update of the Rocket flight in
            seconds.
        angular_rates: numpy.array
            The angular (pitch, yaw, and roll) rates of the Rocket object.

        Returns
        -------
        numpy.array
            Numpy array (containing data with float type) representing the
            orientation of the Rocket object.
        """
        orientation_quaternion = Quaternion(rocket.orientation)
        orientation_quaternion.integrate(angular_rates * self.calibration, delta_time)
        return orientation_quaternion.elements


class Magnetometer(Sensor):
    def update(self, rocket):
        """
        Calculates the magnetic field around the rocket (Tesla)

        Parameters
        ----------
        rocket: A rocket object
            An rocket instance. Calculations will be done according to this rocket's state

        Returns
        -------
        body_magnetic_field
            Numpy array representing the magnetic field around
            the rocket
        """
        quaternion = Quaternion(rocket.orientation)
        body_magnetic_field = quaternion.rotate(rocket.world_mag_field * self.calibration)
        return body_magnetic_field


class Thermistor(Sensor):
    def update(self, rocket):
        """
        Calculates the temperature around the rocket, in celsius.
        Uses NASA formulas:
        https://www.grc.nasa.gov/WWW/K-12/rocket/atmosmet.html

        Parameters
        ----------
        rocket: A rocket object
            An rocket instance. Calculations will be done according to this rocket's state

        Returns
        -------
        temperature: float
            Temperature of the air around the rocket, in celsius.
        """
        # Use NASA formula to calculate temperature
        if rocket.altitude < 11000:
            temperature = 15.04 - 0.00649 * rocket.altitude
        else:
            temperature = -56.46

        return temperature * self.calibration
