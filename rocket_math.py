# This file defines the Rocket class and the general math functions needed to calculate parameters for data generation

import numpy as np
import pymap3d
from math import sin, cos, radians
# Using pyquaternion instead of numpy-quaternion because the later was harder to install (could change later if need be)
#import numpy-quaternion as quat


# -----------------------CONSTANTS---------------------------
# TODO: add computed/determined drag coefficient
# Temporary drag coefficient for calculating drag force
COEFF_DRAG = 0.3

# Current radius of the rocket (SOTS) [m]
TUBE_RADIUS = 0.25

# Gravity at ground level in Spaceport America [m/s^2]
SA_GRAV_GROUND = 9.79121

# Radius of the earth [m]
RAD_EARTH = 6.371e6

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

# TODO: add class definition comments
class Rocket:
    def __init__(self, mass, thrust, burn_time, sensor_noise):
        self.mass = mass  # in pounds (lb)
        self.thrust = thrust  # in pound-force (lbf)
        self.burn_time = burn_time  # in seconds (s)
        self.sensor_noise = sensor_noise  # not determined
        self.position_loc_cart = np.array([0.0, 0.0, 0.0])  # in feet (ft)
        self.position_enu = np.array([0.0, 0.0, 0.0])  # in feet (ft)
        self.velocity = np.array([0.0, 0.0, 0.0])  # in feet/second (ft/s)
        self.acceleration = np.array([0.0, 0.0, 0.0])  # in feet/second^2 (ft/s^2)
        self.orientation = 0  # not determined
        self.baro_pressure = 0.0  # in atmospheres (atm)
        self.temperature = 0.0  # in fahrenheit
        self.altitude = 0.0  # in feet (ft)
        self.air_speed = 0.0  # in feet/second probably (ft/s)


# TODO: Add in units manager
# TODO: Add precision
# Converts a distance in feet to a distance in meters
def ft_to_meters(distance_ft: np.array([])) -> np.array([]):
    return distance_ft / 3.2808


# TODO: Add in units manager
# TODO: Add precision
# Converts a distance in meters to a distance in feet
def meters_to_ft(distance_meters: np.array([])) -> np.array([]):
    return distance_meters * 3.2808


# Calculates gravity based on altitude above sea level (in ft/s^2)
def get_gravity(altitude: float) -> float:
    return meters_to_ft(SA_GRAV_GROUND * ((RAD_EARTH / (RAD_EARTH + ft_to_meters(altitude))) ** 2))


# Calculates mass based on estimated loss in mass (in lb)
def get_mass(current_rocket: Rocket) -> float:
    if current_rocket.mass["comb_mass"] != 0:
        if current_rocket.mass["comb_mass"] - MASS_LOSS < 0:
            current_rocket.mass["comb_mass"] = 0
            current_rocket.mass["total_mass"] = current_rocket.mass["body_mass"] + (MASS_LOSS - current_rocket.mass["comb_mass"])
        current_rocket.mass["comb_mass"] = current_rocket.mass["comb_mass"] - MASS_LOSS
        current_rocket.mass["total_mass"] = current_rocket.mass["body_mass"] + current_rocket.mass["comb_mass"]
    return current_rocket.mass


# Calculates the magnitude of the velocity vector (in lbf)
def get_vel_magnitude(current_rocket: Rocket) -> float:
    return np.sqrt(current_rocket.velocity[0] ** 2 + current_rocket.velocity[1] ** 2 + current_rocket.velocity[2] ** 2)


# Calculates the velocity unit vector
# Launch angle is in degrees and is from the x-axis to the z-axis
# TODO: investigate numpy unit vector  (unit_vector = current_rocket.velocity / np.linalg.norm(current_rocket.velocity))
# TODO: fix the fact the unit vector is negative and switches sign
def get_vel_unit_vector(current_rocket: Rocket) -> np.array([]):
    if not get_vel_magnitude(current_rocket):
        print("Magnitude of velocity is 0.")
        multiplier_arr = np.array([round(cos(radians(TOWER_ANGLE)), 4), 0.0, round(sin(radians(TOWER_ANGLE)), 4)])
        print("This is cos: " + str(round(cos(radians(TOWER_ANGLE)), 4)))
        print("This is sin: " + str(round(sin(radians(TOWER_ANGLE)), 4)))
        return np.array([1.0, 0.0, 1.0]) * multiplier_arr
    print("In get vector & This is velocity: " + str(current_rocket.velocity))
    print("This is velocity_magnitude: " + str(get_vel_magnitude(current_rocket)))
    return current_rocket.velocity / get_vel_magnitude(current_rocket)


# TODO: replace this function with one from sensor measurements (temp and pressure)
# Calculates air density from 3rd-degree polynomial based on plot of air density vs altitude data obtained
# from [VERSION 1]
# https://www.engineeringtoolbox.com/standard-atmosphere-d_604.html
def get_air_density2(current_rocket: Rocket) -> float:
    alt = current_rocket.altitude
    return (-1.2694e-14 * (alt ** 3)) + (1.9121e-09 * (alt ** 2)) + (-8.7147e-05 * alt) + 1.1644e+00


# TODO: check if what the output air density and fix if the input should be in meters or feet
#  (possibly in meters currently)
# Alternative to other air density calculation [VERSION 2]
def get_air_density(current_rocket: Rocket) -> float:
    alt = current_rocket.altitude
    return 1.22 * 0.9 ** (alt / 1000)


# TODO: make function into constant since no variability
# Calculates the rocket's cross-sectional area [ft^2]
def get_cross_sec_area() -> float:
    return np.pi * (TUBE_RADIUS ** 2)


# TODO [Later]: if speeds of rocket are transonic/supersonic, wave drag may be a thing to consider
# Calculates the drag force experienced by the rocket in local cartesian vector form [x,y,z] (in ft)
def get_drag_force(current_rocket: Rocket) -> np.array([]):
    drag_force_mag = 0.5 * COEFF_DRAG * (get_vel_magnitude(current_rocket) ** 2) * get_air_density(current_rocket) * \
                     get_cross_sec_area()
    return get_vel_unit_vector(current_rocket) * drag_force_mag  # TODO: make the launch angle not hard-coded


# TODO: implement burn time
# Calculates the thrust vector based on velocity unit vector [x,y,z] (in lbf)
def get_thrust(current_rocket: Rocket, current_time: float) -> np.array([]):
    if current_time <= current_rocket.burn_time:
        thrust_magnitude = np.sqrt(current_rocket.thrust[0] ** 2 + current_rocket.thrust[1] ** 2
                                   + current_rocket.thrust[2] ** 2)
        print("This is vel_unit_vec: " + str(get_vel_unit_vector(current_rocket)))
        return get_vel_unit_vector(current_rocket) * thrust_magnitude  # TODO: make the launch angle not hard-coded
    print("Made it here")
    return np.array([0, 0, 0])


# Calculates acceleration of the rocket in local cartesian vector form [x,y,z]
# TODO: check units for everything
# TODO: investigate time component?
def rocket_acceleration(current_rocket: Rocket, previous_time: float, current_time: float) -> np.array([]):
    resultant_force = np.array([0.0, 0.0, 0.0])
    drag_force = get_drag_force(current_rocket)
    resultant_force[0] = current_rocket.thrust[0] - drag_force[0]
    resultant_force[1] = current_rocket.thrust[1] - drag_force[1]
    resultant_force[2] = current_rocket.thrust[2] - drag_force[2] - ((current_rocket.mass *
                                                                      get_gravity(current_rocket.altitude)) / 32.174049)
    print("This is thrust: " + str(current_rocket.thrust[2]))
    print("This is Fg: " + str((current_rocket.mass * get_gravity(current_rocket.altitude)) / 32.174049))
    print("This is Fd: " + str(drag_force[2]))
    print("This is altitude: " + str(current_rocket.altitude))

    if any(resultant_force):
        return resultant_force / current_rocket.mass
    print("we be here :/")
    return current_rocket.acceleration


# TODO: edit this due to the fact that methodology changed
# Calculates the current cartesian position of the rocket [x,y,z]
# Use this: https://apps.dtic.mil/dtic/tr/fulltext/u2/a484864.pdf
# And this: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
def rocket_position_local_cartesian(current_rocket: Rocket, previous_time: float, current_time: float) -> np.array([]):
    return current_rocket.position_loc_cart + current_rocket.velocity * (current_time - previous_time) + 0.5 * \
           current_rocket.acceleration * (current_time - previous_time) ** 2


# TODO: delete this once new function is done
# Converting cartesian to enu for position of rocket
def cartesian_to_enu(current_rocket: Rocket, position_cartesian) -> np.array([]):
    position_cartesian = ft_to_meters(position_cartesian)
    geodectic_coords = pymap3d.ecef2geodetic(position_cartesian[0], position_cartesian[1], position_cartesian[2], None,
                                             True)
    current_rocket.position_ENU = meters_to_ft(pymap3d.ecef2enu(position_cartesian[0], position_cartesian[1],
                                                                position_cartesian[2], geodectic_coords[0],
                                                                geodectic_coords[1], geodectic_coords[2], None, True))


# TODO: add new function for converting from cartesian to enu/changing library
# Converting local cartesian to ENU for position of rocket
# Use this for quaternion rotation: https://kieranwynn.github.io/pyquaternion/
def loc_cart_to_enu_2(current_rocket: Rocket, previous_time: float, current_time: float) -> np.array([]):
    return np.array([0.0, 0.0, 0.0])
    # rot_quaternion = Quaternion(axis=[0, 0, 0], angle=0)  # TODO: make this the true scalar and vector
    # return rot_quaternion.rotate(current_rocket.position_loc_cart)


# TODO: add bounds chckes
# Calculates velocity of the rocket by integrating acceleration in local cartesian vector form [x,y,z]
def rocket_velocity(current_rocket: Rocket, previous_time: float, current_time: float) -> np.array([]):
    return current_rocket.velocity + current_rocket.acceleration * (current_time - previous_time)
