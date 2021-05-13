"""
This file defines multiple classes, one for each sensor. Each class has an
update function that takes a rocket as a parameter, which it performs updates on.
"""

import numpy as np
from pyquaternion import Quaternion


class Sensor:  # TODO: add sensor noise to sensor class
    """
    A class used to represent a Sensor object.

    ...

    Attributes
    ----------

    calibration: float
    """

    def __init__(self, calibration):
        self.calibration = calibration


class Accelerometer(Sensor):
    """
    A class used to represent a Accelerometer object.

    ...

    Attributes
    ----------

    calibration: float
    body_acceleration: numpy.array
    """
    def __init__(self, calibration=1):
        self.body_acceleration = np.array([0.0, 0.0, 0.0])  # [m/s^2]
        super().__init__(calibration)

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
        # world_mag_field is in NED so it needs to be converted to ENU first (x=y, y=x, z=-z)
        body_acceleration = quaternion.rotate(
            rocket.world_acceleration * self.calibration)
        return body_acceleration


class Baro_Pressure_Sensor(Sensor):
    """
    A class used to represent a Barometric Pressure Sensor object.

    ...

    Attributes
    ----------

    calibration: float
    baro_pressure: float
    """
    def __init__(self, calibration=1):
        self.baro_pressure = 0  # [KPa]
        super().__init__(calibration)

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
    """
    A class used to represent a Gyroscopic Sensor object.

    ...

    Attributes
    ----------

    calibration: float
    orientation: numpy.array

    Notes
    -----
    orientation is in the format of [w, x, y, z], where [w] is the scalar part
    of the quaternion and [x, y, z] are the vector parts.
    """
    def __init__(self, calibration=1):
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # identity quat.
        super().__init__(calibration)

    def update(self, rocket, angular_rates, delta_time) -> np.array([]):
        """
        Calculates the orientation quaternion of the Rocket object based on
        fixed angular rates.

        Parameters
        ----------
        rocket: A rocket object
            An rocket instance. Calculations will be done according to this rocket's state

        angular_rates: numpy.array
            The angular (pitch, yaw, and roll) rates of the Rocket object.

        delta_time: float
            The change in time since the last update of the Rocket flight in
            seconds.

        Returns
        -------
        numpy.array
            Numpy array (containing data with float type) representing the
            orientation of the Rocket object.
        """
        orientation_quaternion = Quaternion(rocket.orientation)
        orientation_quaternion.integrate(angular_rates * self.calibration,
                                         delta_time)
        return orientation_quaternion.elements


class Magnetometer(Sensor):
    """
    A class used to represent a Magnetometer object.

    ...

    Attributes
    ----------

    calibration: float
    body_mag_field: numpy.array

    Notes
    -----
    world_mag_field is in the format of [D, I, H, X, Y, Z, F], where:
        D = Geomagnetic declination [deg]
        I = Geomagnetic inclination [deg]
        H = Horizontal geomagnetic field intensity [nT]
        X = North component of geomagnetic field intensity [nT]
        Y = East component of geomagnetic field intensity [nT]
        Z = Vertical component of geomagnetic field intensity [nT]
        F = Total geomagnetic field intensity [nT]

    However, body_mag_field only includes the X,Y,Z components.
    """
    def __init__(self, calibration=1):
        self.body_mag_field = np.array(
            [0.0, 0.0, 0.0])  # See class docstring
        super().__init__(calibration)

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
            the rocket (Note: in ENU coordinate frame).
        """
        quaternion = Quaternion(rocket.orientation)
        body_magnetic_field = quaternion.rotate(
            np.array([rocket.world_mag_field[4], rocket.world_mag_field[3],
                      -rocket.world_mag_field[5]]) * self.calibration)
        return body_magnetic_field


class Thermistor(Sensor):
    """
    A class used to represent a Thermistor object.

    ...

    Attributes
    ----------

    calibration: float
    temperature: float
    """
    def __init__(self, calibration=1):
        self.temperature = 0  # [Celsius]
        super().__init__(calibration)

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
