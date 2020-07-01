# This file defines the Rocket class and the general math functions needed to calculate parameters for data generation

import numpy as np
import pymap3d
from math import sin, cos, radians

# Using pyquaternion instead of numpy-quaternion because the later was harder to install (could change later if need be)
# import numpy-quaternion as quat


# -----------------------CONSTANTS---------------------------
# TODO: add computed/determined drag coefficient
# Temporary drag coefficient for calculating drag force
COEFF_DRAG = 0.3

# Current radius of the rocket (SOTS) [ft]
TUBE_RADIUS = 0.25

# Gravity at ground level in Spaceport America [ft/s^2]
# Websites for finding the local gravity:
# https://www.latlong.net/convert-address-to-lat-long.html
# https://www.sensorsone.com/local-gravity-calculator/#local-gravity
# Websites give gravity in [m/s^2], which was 9.75594. Units were converted.
SA_GRAV_GROUND = 32.138911

# Radius of the earth [ft]
RAD_EARTH = 20.902e6

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
LOCAL_MAGNETIC_FIELD = 0  # TODO: figure out what we need to actually store here


# ------------------------------------------------------------

# TODO: add function definitions and remove explicit return type
# TODO: add class definition comments
class Rocket:
    def __init__(self, mass, thrust, burn_time, sensor_noise):
        self.mass = mass  # in pounds (lb)
        self.thrust = thrust  # in pound-force (lbf)
        self.burn_time = burn_time  # in seconds (s)
        self.sensor_noise = sensor_noise  # TODO: add in explanation for this dictionary
        self.position_cart = np.array([0.0, 0.0, 0.0])  # in feet (ft)
        self.position_enu = np.array([0.0, 0.0, 0.0])  # in feet (ft)
        self.velocity = np.array([0.0, 0.0, 0.0])  # in feet/second (ft/s)
        self.acceleration = np.array([0.0, 0.0, 0.0])  # in feet/second^2 (ft/s^2)
        self.orientation = 0  # not determined
        self.baro_pressure = 0.0  # in atmospheres (atm)
        self.temperature = 0.0  # in fahrenheit
        self.altitude = 0.0  # in feet (ft)
        self.air_speed = 0.0  # in feet/second probably (ft/s)

    # Calculates gravity based on altitude above sea level (in ft/s^2)
    def get_gravity(self) -> float:
        return SA_GRAV_GROUND * ((RAD_EARTH / (RAD_EARTH + self.altitude)) ** 2)

    # Calculates mass based on estimated loss in mass (in lb)
    def get_mass(self) -> float:
        if self.mass["comb_mass"] != 0:
            if self.mass["comb_mass"] - MASS_LOSS < 0:
                self.mass["comb_mass"] = 0
                self.mass["total_mass"] = self.mass["body_mass"] + (
                        MASS_LOSS - self.mass["comb_mass"])
            self.mass["comb_mass"] = self.mass["comb_mass"] - MASS_LOSS
            self.mass["total_mass"] = self.mass["body_mass"] + self.mass["comb_mass"]
        return self.mass

    # Calculates the magnitude of the velocity vector (in lbf)
    def get_velocity_mag(self) -> float:
        return np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2 + self.velocity[2] ** 2)

    # Calculates the velocity unit vector
    # Launch angle is in degrees and is from the x-axis to the z-axis
    # TODO: investigate numpy unit vector  (unit_vector = current_rocket.velocity / np.linalg.norm(current_rocket.velocity))
    # TODO: fix the fact the unit vector is negative and switches sign
    def get_velocity_uv(self) -> np.array([]):
        if not self.get_velocity_mag(self):
            multiplier_arr = np.array([round(cos(radians(TOWER_ANGLE)), 4), 0.0, round(sin(radians(TOWER_ANGLE)), 4)])
            return np.array([1.0, 0.0, 1.0]) * multiplier_arr
        return self.velocity / self.get_velocity_mag(self)

    # TODO: replace this function with one from sensor measurements (temp and pressure)
    # Calculates air density from 3rd-degree polynomial based on plot of air density vs altitude data obtained
    # from [VERSION 1]
    # https://www.engineeringtoolbox.com/standard-atmosphere-d_604.html
    def get_air_density2(self) -> float:
        alt = self.altitude
        return (-1.2694e-14 * (alt ** 3)) + (1.9121e-09 * (alt ** 2)) + (-8.7147e-05 * alt) + 1.1644e+00

    # TODO: check if what the output air density and fix if the input should be in meters or feet
    #  (possibly in meters currently)
    # Alternative to other air density calculation [VERSION 2]
    def get_air_density(self) -> float:
        alt = self.altitude
        return 1.22 * 0.9 ** (alt / 1000)

    # TODO: make function into constant since no variability
    # Calculates the rocket's cross-sectional area [ft^2]
    def get_cross_sec_area(self) -> float:
        return np.pi * (TUBE_RADIUS ** 2)

    # TODO [Later]: if speeds of rocket are transonic/supersonic, wave drag may be a thing to consider
    # Calculates the drag force experienced by the rocket in local cartesian vector form [x,y,z] (in ft)
    def get_drag_force(self) -> np.array([]):
        drag_force_mag = 0.5 * COEFF_DRAG * (self.get_velocity_mag(self) ** 2) * self.get_air_density(self) * \
                         self.get_cross_sec_area()
        return self.get_velocity_uv(self) * drag_force_mag  # TODO: make the launch angle not hard-coded

    # TODO: implement burn time
    # Calculates the thrust vector based on velocity unit vector [x,y,z] (in lbf)
    def get_thrust(self, current_time: float) -> np.array([]):
        if current_time <= self.burn_time:
            thrust_magnitude = np.sqrt(self.thrust[0] ** 2 + self.thrust[1] ** 2
                                       + self.thrust[2] ** 2)
            return self.get_velocity_uv(self) * thrust_magnitude  # TODO: account for launch angle
        return np.array([0, 0, 0])

    # Calculates acceleration of the rocket in local cartesian vector form [x,y,z]
    # TODO: check units for everything
    # TODO: investigate time component?
    def rocket_acceleration(self, previous_time, current_time) -> np.array([]):
        resultant_force = np.array([0.0, 0.0, 0.0])
        drag_force = self.get_drag_force(self)
        resultant_force[0] = self.thrust[0] - drag_force[0]
        resultant_force[1] = self.thrust[1] - drag_force[1]
        resultant_force[2] = self.thrust[2] - drag_force[2] - (self.mass * self.get_gravity(self.altitude))
        if any(resultant_force):
            return resultant_force / self.mass
        return self.acceleration

    # TODO: add bounds checks
    # Calculates velocity of the rocket by integrating acceleration in local cartesian vector form [x,y,z]
    def rocket_velocity(self, previous_time, current_time) -> np.array([]):
        return self.velocity + self.acceleration * (current_time - previous_time)

    # TODO: add functionality
    # Calculated position of the rocket by integrating velocity in the local cartesian vector form [x,y,z]
    def rocket_position(self, previous_time, current_time) -> np.array([]):
        pass
