"""
This file defines the Rocket class and the general math functions needed to
calculate parameters for data generation.
"""

import numpy as np

# TODO: investigate quaternion implementation
# Using pyquaternion instead of numpy-quaternion because the later was harder
# to install (could change later if need be)
# import numpy-quaternion as quat


# -----------------------CONSTANTS---------------------------
# TODO: add computed/determined drag coefficient
# Temporary drag coefficient for calculating drag force
COEFF_DRAG = 0.3

# Current radius of the rocket (as of July 1, 2020) [ft]
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

# Mass lost per second [lb/s]
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
            mass is a dict of {"total_mass": float, "body_mass": float,
            "prop_mass": float} where "total_mass" represents the total mass
            of the rocket, "body_mass" represents the mass of the body of the
            rocket (without the combustible mass), and "prop_mass" represents
            the mass of the combustible materials in the rocket (i.e. oxidizer,
            fuel).

        thrust: numpy.ndarray
            thrust represents the average thrust the Rocket generates during
            a flight.

        burn_time: float
            burn_time is the amount of time the Rocket generates thrust.

        sensor_noise: dict of {str : float}
            sensor_noise is a dict of {"press_noise": float, "temp_noise":
            float, "accel_noise": float, "gyro_noise": float, "mag_noise":
            float} where "press_noise" is the pressure noise, "temp_noise" is
            the temperature noise, "accel_noise" is the accelerometer noise,
            "gyro_noise" is the gyroscope noise, and "mag_noise" is the
            magnetometer noise.
        """
        if mass is None:
            mass = {"total_mass": 0, "body_mass": 0, "prop_mass": 0}
        if sensor_noise is None:
            sensor_noise = {"press_noise": 0, "temp_noise": 0,
                            "accel_noise": 0, "gyro_noise": 0,
                            "mag_noise": 0}
        self.mass = mass  # [lb]
        self.thrust = thrust  # [lbf]
        self.burn_time = burn_time  # [s]
        self.sensor_noise = sensor_noise
        self.position = np.array([0.0, 0.0, 0.0])  # [ft] (cartesian vector)
        self.position_enu = np.array([0.0, 0.0, 0.0])  # [ft]
        self.velocity = np.array([0.0, 0.0, 0.0])  # [ft/s]
        self.acceleration = np.array([0.0, 0.0, 0.0])  # [ft/s^2]
        self.orientation = None
        self.baro_pressure = 0  # [psi]
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
               f"{self.mass['prop_mass']}\n\t}}\n\tThrust:\t\t" \
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
               f"{self.altitude}"  # Note: still unsure about formatting.

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
        if type(self) == type(other):
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

    def gravity(self) -> float:
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

    def update_mass(self, timestep) -> dict:
        """
        Calculates the mass of the Rocket object based on the estimated loss
        in mass.

        Parameters
        ----------
        timestep: float
            The time interval for which generator will progress during next
            "step".

        Returns
        -------
        dict
            Updated mass dict of {str: float} in lbs.
        """
        mass = self.mass
        mass["prop_mass"] = max(0, mass["prop_mass"] - (MASS_LOSS * timestep))
        mass["total_mass"] = mass["body_mass"] + mass["prop_mass"]

        return mass

    def speed(self) -> float:
        """
        Calculates the magnitude of the velocity vector.

        Returns
        -------
        float
            Number representing the magnitude of the velocity vector in ft/s.
        """
        return np.linalg.norm(self.velocity)

    def velocity_uv(self) -> np.array([]):
        """
        Calculates the velocity unit vector.

        Returns
        -------
        numpy.ndarray
            Numpy array (containing data with float type) representing the
            unit vector for the velocity vector.
        """
        if not self.speed():
            return np.array([0, 0, 0])
        return self.velocity / np.linalg.norm(self.velocity)

    # TODO: check if what the output air density and fix if the input should
    #  be in meters or feet (possibly in meters currently)
    # TODO: determine where this equation was determined
    # Alternative to other air density calculation [soon to be deprecated]
    def air_density(self) -> float:
        alt = self.altitude
        return 1.22 * (0.9 ** (alt / 1000))

    # TODO [Later]: if speeds of rocket are transonic/supersonic, wave drag
    #  may be a thing to consider
    def drag_force(self) -> np.array([]):
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
                self.speed() ** 2) * self.air_density() * CROSS_SEC_AREA
        return self.velocity_uv() * drag_force_mag

    # TODO: account for launch angle
    # TODO: determine thrust unit vector through orientation
    def update_thrust(self, current_time: float) -> np.array([]):
        """
        Calculates the thrust vector of the Rocket object based on the
        velocity unit vector.

        Parameters
        ----------
        current_time: float
            The current time of the rocket flight in seconds.

        Returns
        -------
        numpy.ndarray
            Numpy array (containing data with float type) representing the
            thrust generated by the Rocket object in lbf.
        """
        if np.all(self.speed() == np.array([0, 0, 0])):
            return self.thrust
        elif current_time <= self.burn_time:
            return self.velocity_uv() * np.linalg.norm(self.thrust)
        return np.array([0, 0, 0])

    def update_acceleration(self) -> np.array([]):
        """
        Calculates the acceleration of the Rocket object in local cartesian
        vector form.

        Returns
        -------
        numpy.ndarray
            Numpy array (containing data with float type) representing the
            acceleration of the Rocket object in ft/s^2.
        """
        resultant_force = np.array([0.0, 0.0, 0.0])
        drag_force = self.drag_force()
        resultant_force[0] = self.thrust[0] - drag_force[0]
        resultant_force[1] = self.thrust[1] - drag_force[1]
        resultant_force[2] = self.thrust[2] - drag_force[2] - (
                self.mass["total_mass"] * self.gravity())
        if any(resultant_force):
            return resultant_force / self.mass["total_mass"]
        return self.acceleration

    def update_velocity(self, delta_time) -> np.array([]):
        """
        Calculates the velocity vector of the Rocket object in local cartesian
        vector form.

        Parameters
        ----------
        delta_time: float
            The change in time since the last update of rocket flight in
            seconds.

        Returns
        -------
        numpy.ndarray
            Numpy array (containing data with float type) representing the
            velocity of the Rocket object in ft/s.
        """
        return self.velocity + self.acceleration * delta_time

    def update_position(self, delta_time) -> np.array([]):
        """
        Calculates the position vector of the Rocket object in local cartesian
        vector form.

        Parameters
        ----------
        delta_time: float
            The change in time since the last update of rocket flight in
            seconds.

        Returns
        -------
        numpy.ndarray
            Numpy array (containing data with float type) representing the
            position of the Rocket object in ft.
        """
        position = self.position + self.velocity * delta_time

        # Check if position is negative (since rocket launch is from ground,
        # it should always be >= 0)
        if position[2] < 0:
            position[2] = 0
        return position

    # Converts the position of the rocket for local cartesian to ENU [ft]
    def cart_to_enu(self) -> np.array([]):
        pass
