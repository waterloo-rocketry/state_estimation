"""
This file defines the Rocket class and the general math functions needed to
calculate parameters for data generation.
"""

import numpy as np
from pyquaternion import Quaternion

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

# The rocket's fixed angular velocities in rad/s
X_ANGULAR_RATE = 0  # Current coordinate system yaw rate
Y_ANGULAR_RATE = np.pi/180  # Current coordinate system pitch rate (1 degree/s)
Z_ANGULAR_RATE = np.pi  # Current coordinate system roll rate (180 degree/s)
ANGULAR_RATES = np.array([X_ANGULAR_RATE, Y_ANGULAR_RATE, Z_ANGULAR_RATE])

# Launch site parameters
TOWER_ANGLE = 90
LAUNCH_SITE_ALTITUDE = 0
LOCAL_MAGNETIC_FIELD = 0  # TODO: figure out what we need to actually store

# Tolerance for equality checks
TOLERANCE = 0.001


# ------------------------------------------------------------

class Rocket:
    """
    A class used to represent a Rocket object.

    ...

    Attributes
    ----------

    mass: dict of {str : float}
    thrust: numpy.array
    burn_time: float
    sensor_noise: dict of {str : float}
    position: numpy.array
    position_enu: numpy.array
    velocity: numpy.array
    acceleration: numpy.array
    orientation: numpy.array
    baro_pressure: float
    temperature: float
    altitude: float

    Notes
    -----
    orientation is in the format of [w, x, y, z], where [w] is the scalar part
    of the quaternion and [x, y, z] are the vector parts.
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

        thrust: numpy.array
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
        self.world_acceleration = np.array([0.0, 0.0, 0.0])  # [ft/s^2]
        self.body_acceleration = np.array([0.0, 0.0, 0.0]) # [ft/s^2]
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # identity quat.
        self.baro_pressure = 0  # [KPa]
        self.temperature = 0  # [Celsius]
        self.altitude = 0  # [ft]
        self.body_mag_field = np.array([0.0, 0.0, 0.0]) # Waiting on model choice
        self.world_mag_field = np.array([0.0, 0.0, 0.0]) # Waiting on model choice

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
               f"{self.velocity}\n\tWorld Acceleration:\t" \
               f"{self.world_acceleration}\n\tBody Acceleration:\t" \
               f"{self.body_acceleration}\n\tOrientation:\t" \
               f"{self.orientation}\n\tBaro Pressure:\t" \
               f"{self.baro_pressure}\n\tTemperature:\t" \
               f"{self.temperature}\n\tAltitude:\t" \
               f"{self.altitude}\n\tBody Magnetic Field:\t" \
               f"{self.body_mag_field}\n\tWorld Magnetic Field:\t" \
               f"{self.world_mag_field}"    # Note: still unsure about formatting.

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
                   and all((self.thrust - other.thrust) <= TOLERANCE) \
                   and self.burn_time == other.burn_time \
                   and self.sensor_noise == other.sensor_noise \
                   and all((self.position - other.position) <= TOLERANCE) \
                   and all((self.position_enu - other.position_enu)
                           <= TOLERANCE) \
                   and all((self.velocity - other.velocity) <= TOLERANCE) \
                   and all((self.world_acceleration - other.world_acceleration)
                           <= TOLERANCE) \
                   and all((self.orientation - other.orientation)
                           <= TOLERANCE) \
                   and self.baro_pressure == other.baro_pressure \
                   and self.temperature == other.temperature \
                   and self.altitude == other.altitude \
                   and all((self.body_mag_field - other.body_mag_field)
                           <= TOLERANCE) \
                   and all((self.world_mag_field - other.world_mag_field)
                           <= TOLERANCE)
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
        numpy.array
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
        numpy.array
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
        numpy.array
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
        numpy.array
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
        return self.world_acceleration

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
        numpy.array
            Numpy array (containing data with float type) representing the
            velocity of the Rocket object in ft/s.
        """
        return self.velocity + self.world_acceleration * delta_time

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
        numpy.array
            Numpy array (containing data with float type) representing the
            position of the Rocket object in ft.
        """
        position = self.position + self.velocity * delta_time

        # Check if position is negative (since rocket launch is from ground,
        # it should always be >= 0)
        if position[2] < 0:
            position[2] = 0
        return position

    def update_orientation(self, angular_rates, delta_time) -> np.array([]):
        """
        Calculates the orientation quaternion of the Rocket object based on
        fixed angular rates.

        Parameters
        ----------
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
        orientation_quaternion = Quaternion(self.orientation)
        orientation_quaternion.integrate(angular_rates, delta_time)
        return orientation_quaternion.elements

    def update_baro_pressure(self):
        """
        Calculates the barometric pressure of the atmosphere around the Rocket.
        Uses NASA formulas:
        https://www.grc.nasa.gov/WWW/K-12/airplane/atmosmet.html

        Returns
        ---------
        baro_pressure: float
            Barometric pressure of the atmosphere around the rocket, in kilopascals.
        """
        # Use NASA formula to calculate barometric pressure
        if self.altitude < 11000:
            baro_pressure = 101.29 * ((self.temperature + 273.1) / 288.08) ** 5.256
        else:
            baro_pressure = 22.65 * np.exp(1.73 - 0.000157 * self.altitude)

        return baro_pressure

    def update_temperature(self):
        """
        Calculates the temperature around the rocket, in celsius.
        Uses NASA formulas:
        https://www.grc.nasa.gov/WWW/K-12/airplane/atmosmet.html

        Returns
        -------
        temperature: float
            Temperature of the air around the rocket, in celsius.
        """
        # Use NASA formula to calculate temperature
        if self.altitude < 11000:
            temperature = 15.04 - 0.00649 * self.altitude
        else:
            temperature = -56.46
        
        return temperature

    def update_body_acceleration(self):
        """
        Calculates the body (proper) acceleration of the rocket,
        that is what the accelerometer measures.

        Returns
        -------
        proper_acceleration
            Numpy array representing the body (proper) acceleration 
            of the rocket
        """
        quaternion = Quaternion(self.orientation)
        proper_acceleration = quaternion.rotate(self.world_acceleration)
        return proper_acceleration

    def update_magnetic_field(self):
        """
        Calculates the magnetic field around the rocket (Tesla)

        Returns
        -------
        body_magnetic_field
            Numpy array representing the magnetic field around
            the rocket
        """
        quaternion = Quaternion(self.orientation)
        body_magnetic_field = quaternion.rotate(self.world_mag_field)
        return body_magnetic_field

    # Converts the position of the rocket for local cartesian to ENU [ft]
    def cart_to_enu(self) -> np.array([]):
        pass
