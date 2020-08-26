"""
This file defines the Rocket class and the general math functions needed to
calculate parameters for data generation.
"""

import numpy as np

# from math import sin, cos, radians

# TODO: investigate quaternion implementation
# Using pyquaternion instead of numpy-quaternion because the later was harder
# to install (could change later if need be)
# import numpy-quaternion as quat


# -----------------------CONSTANTS---------------------------
# TODO: add computed/determined drag coefficient
# Temporary drag coefficient for calculating drag force
COEFF_DRAG = 0.3

# Current radius of the rocket (as if July 1, 2020) [ft]
TUBE_RADIUS = 0.25

# Current cross sectional area of the rocket (as of July 1, 2020) [ft^2]
CROSS_SEC_AREA = np.pi * (TUBE_RADIUS ** 2)  # 0.1963495408...

# Estimated gravity at ground level in Spaceport America [ft/s^2]
# Websites for finding the local gravity:
# https://www.latlong.net/convert-address-to-lat-long.html
# https://www.sensorsone.com/local-gravity-calculator/#local-gravity
# Websites give gravity in [m/s^2], which was 9.75594. Units were converted.
SA_GRAV_GROUND = 32.138911

# Radius of the earth [ft]
RAD_EARTH = 20.902e6

# Body mass of the rocket (AKA non-fuel related mass) [lb]
BODY_MASS = 55  # currently arbitrary

# Mass lost per timestep [lb]
# TODO: implement the proper loss
MASS_LOSS = 0.05

# The rocket's fixed angular velocities
YAW_RATE = 0
PITCH_RATE = 1
ROLL_RATE = 180  # in deg/s

# Launch site parameters
TOWER_ANGLE = 90
LAUNCH_SITE_ALTITUDE = 0
LOCAL_MAGNETIC_FIELD = 0  # TODO: figure out what we need to actually store

# Decimal places for final answers
DECIMALS = 4


# ------------------------------------------------------------

# TODO: add function docstrings
# TODO: finish filling in class docstrings
# TODO: determine when rounding should occur
class Rocket:
    """
    A class used to represent a Rocket object.

    ...

    Attributes
    ----------
    mass: dict of {str : float}
    thrust: numpy.ndarray
    burn_time: float
    sensor_noise: dict of {str : float}
    position: numpy.ndarray
    position_enu: numpy.ndarray
    velocity: numpy.ndarray
    acceleration: numpy.ndarray
    orientation: None
    baro_pressure: float
    temperature: float
    altitude: float

    """
    def __init__(self, mass=None, thrust=np.array([0, 0, 0]), burn_time=0,
                 sensor_noise=None):
        """
        Initialize a Rocket.

        Parameters
        ----------
        mass: dict of {str : float}
        thrust: numpy.ndarray
        burn_time: float
        sensor_noise: dict of {str : float}
        """
        if mass is None:
            mass = {"total_mass": 0, "body_mass": 0, "comb_mass": 0}
        if sensor_noise is None:
            sensor_noise = {"press_noise": 0, "temp_noise": 0,
                            "accel_noise": 0, "gyro_noise": 0,
                            "mag_noise": 0}
        self.mass = mass  # [lb]  # TODO: add explanation for dict
        self.thrust = thrust  # [lbf]
        self.burn_time = burn_time  # [s]
        self.sensor_noise = sensor_noise  # TODO: add explanation for dict
        self.position = np.array([0.0, 0.0, 0.0])  # [ft] (cartesian vector)
        self.position_enu = np.array([0.0, 0.0, 0.0])  # [ft]
        self.velocity = np.array([0.0, 0.0, 0.0])  # [ft/s]
        self.acceleration = np.array([0.0, 0.0, 0.0])  # [ft/s^2]
        self.orientation = None
        self.baro_pressure = 0  # [atm]
        self.temperature = 0  # in fahrenheit
        self.altitude = 0  # [ft]

    def __repr__(self):
        """
        'Normal' string representation of Rocket object in concise form.

        Returns
        -------
        str
            Bare bones Rocket object.
        """
        return str(self.__class__) + ":" + str(self.__dict__)

    def __str__(self):
        """
        'Pretty' string representation of Rocket object.

        Returns
        -------
        str
            Human-readable/formatted Rocket object.
        """
        return f"Rocket object:\n\t\tINPUTS\n\tMass:\n\t{{\n\t\t" \
               f"Total Mass:\t{self.mass['total_mass']}\n\t\tBody Mass:\t" \
               f"{self.mass['body_mass']}\n\t\tComb Mass:\t" \
               f"{self.mass['comb_mass']}\n\t}}\n\tThrust:\t\t" \
               f"{self.thrust}\n\tBurn Time:\t" \
               f"{self.burn_time}\n\tSensor Noise:\n\t{{\n\t\tPressure:\t" \
               f"{self.sensor_noise['press_noise']}\n\t\tTemperature:\t" \
               f"{self.sensor_noise['temp_noise']}\n\t\tAcceleration:\t" \
               f"{self.sensor_noise['accel_noise']}\n\t\tGyro:\t\t" \
               f"{self.sensor_noise['gyro_noise']}\n\t\tMagnetic:\t" \
               f"{self.sensor_noise['mag_noise']}\n\t}}\n\n\t\tOUTPUTS\n\t" \
               f"Position:\t{self.position}\n\tPosition ENU:\t" \
               f"{self.position_enu}\n\tVelocity:\t" \
               f"{self.velocity}\n\tAcceleration:\t" \
               f"{self.acceleration}\n\tOrientation:\t" \
               f"{self.orientation}\n\tBaro Pressure:\t" \
               f"{self.baro_pressure}\n\tTemperature:\t" \
               f"{self.temperature}\n\tAltitude:\t" \
               f"{self.altitude}"   # Note: still unsure about formatting.

    def __eq__(self, other):
        """
        Equality method for comparing if two Rocket objects are the equal.

        Parameters
        ----------
        other
                Object to be compared against.

        Returns
        -------
        bool
            True if both objects are Rocket objects and equal, false if
            otherwise.
        """
        if type(self) == type(
                other):
            return self.mass == other.mass \
                   and np.all(self.thrust == other.thrust) \
                   and self.burn_time == other.burn_time \
                   and self.sensor_noise == other.sensor_noise \
                   and np.all(self.position == other.position) \
                   and np.all(self.position_enu == other.position_enu) \
                   and np.all(self.velocity == other.velocity) \
                   and np.all(self.acceleration == other.acceleration) \
                   and self.orientation == other.orientation \
                   and self.baro_pressure == other.baro_pressure \
                   and self.temperature == other.temperature \
                   and self.altitude == other.altitude
        return False

    def rocket_gravity(self) -> float:
        """
        Calculates the magnitude of gravity experienced by the Rocket object
        based on altitude above sea level.

        Returns
        -------
        float
            Number representing magnitude of gravity in ft/s^2.
        """
        return SA_GRAV_GROUND * (
                (RAD_EARTH / (RAD_EARTH + self.altitude)) ** 2)

    def rocket_mass(self) -> dict:
        """
        Calculates the mass of the Rocket object based on the estimated loss
        in mass.

        Returns
        -------
        self.mass: dict
            Updated mass dict of {str: float} in lbs.
        """
        if self.mass["comb_mass"] != 0:
            if self.mass["comb_mass"] - MASS_LOSS < 0:
                self.mass["comb_mass"] = 0
                self.mass["total_mass"] = self.mass["body_mass"]
            else:
                self.mass["comb_mass"] = round(
                    self.mass["comb_mass"] - MASS_LOSS, DECIMALS)
                self.mass["total_mass"] = self.mass["body_mass"] + self.mass[
                    "comb_mass"]
        return self.mass

    def rocket_velocity_mag(self) -> float:
        """
        Calculates the magnitude of the velocity vector.

        Returns
        -------
        float
            Number representing the magnitude of the velocity vector in ft/s.
        """
        return np.linalg.norm(self.velocity)

    def rocket_velocity_uv(self) -> np.array([]):
        """
        Calculates the velocity unit vector.

        Returns
        -------
        numpy.ndarray
            Numpy array (containing data with float type) representing the
            unit vector for the velocity vector.
        """
        if not self.rocket_velocity_mag():
            return np.array([0, 0, 0])
        return self.velocity / np.linalg.norm(self.velocity)

    # TODO: check if what the output air density and fix if the input should
    #  be in meters or feet (possibly in meters currently)
    # TODO: determine where this equation was determined
    # Alternative to other air density calculation [soon to be deprecated]
    def rocket_air_density(self) -> float:
        alt = self.altitude
        return 1.22 * (0.9 ** (alt / 1000))

    # TODO [Later]: if speeds of rocket are transonic/supersonic, wave drag
    #  may be a thing to consider
    def rocket_drag_force(self) -> np.array([]):
        """
        Calculates the drag force experienced by the Rocket object in the local
        cartesian vector form.

        Returns
        -------
        numpy.ndarray
            Numpy array (containing data with float type) representing the
            drag force experienced by the Rocket object in lbf.
        """
        drag_force_mag = 0.5 * COEFF_DRAG * (
                self.rocket_velocity_mag() ** 2) * self.rocket_air_density() \
                         * CROSS_SEC_AREA
        return np.around(self.rocket_velocity_uv() * drag_force_mag, DECIMALS)

    def rocket_thrust(self, current_time: float) -> np.array([]):
        """
        Calculates the thrust vector of the Rocket object based on the
        velocity unit vector.

        Parameters
        ----------
        current_time : float
            The current time of the rocket flight in seconds.

        Returns
        -------
        numpy.ndarray
            Numpy array (containing data with float type) representing the
            thrust generated by the Rocket object in lbf.
        """
        if np.all(self.rocket_velocity_mag() == np.array([0, 0, 0])):
            return self.thrust
        elif current_time <= self.burn_time:
            thrust_magnitude = np.sqrt(
                self.thrust[0] ** 2 + self.thrust[1] ** 2
                + self.thrust[2] ** 2)
            return np.around(self.rocket_velocity_uv() * thrust_magnitude,
                             DECIMALS)  # TODO: account for launch angle
        return np.array([0, 0, 0])

    def rocket_acceleration(self) -> np.array([]):
        """
        Calculates the accleration of the Rocket object in local cartesian
        vector form.

        Returns
        -------
        numpy.ndarray
            Numpy array (containing data with float type) representing the
            acceleration of the Rocket object in ft/s^2.
        """
        resultant_force = np.array([0.0, 0.0, 0.0])
        drag_force = self.rocket_drag_force()
        resultant_force[0] = self.thrust[0] - drag_force[0]
        resultant_force[1] = self.thrust[1] - drag_force[1]
        resultant_force[2] = self.thrust[2] - drag_force[2] - (
                self.mass["total_mass"] * self.rocket_gravity())
        if any(resultant_force):
            return np.around(resultant_force / self.mass["total_mass"],
                             DECIMALS)
        return np.around(self.acceleration, DECIMALS)

    def rocket_velocity(self, current_time, previous_time) -> np.array([]):
        """
        Calculates the velocity vector of the Rocket object in local cartesian
        vector form.

        Parameters
        ----------
        current_time : float
            The current time of the rocket flight in seconds.
        previous_time : float
            The previous time (by 1 timestep) of the rocket flight in seconds.

        Returns
        -------
        numpy.ndarray
            Numpy array (containing data with float type) representing the
            velocity of the Rocket object in ft/s.
        """
        return np.around(
            self.velocity + self.acceleration * (current_time - previous_time),
            DECIMALS)

    def rocket_position(self, current_time, previous_time) -> np.array([]):
        """
        Calculates the position vector of the Rocket object in local cartesian
        vector form.

        Parameters
        ----------
        current_time : float
            The current time of the rocket flight in seconds.
        previous_time : float
            The previous time (by 1 timestep) of the rocket flight in seconds.

        Returns
        -------
        numpy.ndarray
            Numpy array (containing data with float type) representing the
            position of the Rocket object in ft.
        """
        position = np.around(
            self.position + self.velocity * (current_time - previous_time),
            DECIMALS)

        # Check if position is negative (since rocket launch is from ground,
        # it should always be >= 0)
        if position[2] < 0:
            position[2] = 0
        return position

    # Converts the position of the rocket for local cartesian to ENU [ft]
    def cart_to_enu(self) -> np.array([]):
        pass
