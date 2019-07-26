# This file defines the Rocket class and the general math functions needed to calculate parameters for state estimation

import numpy as np
import pymap3d
# Using pyquaternion instead of numpy-quaternion because the later was harder to install (could change later if need be)
from pyquaternion import Quaternion


# Simulator Inputs:
#   Rate/timestep
#   Rocket parameters: mass, average thrust, burn time
#   Sensor models: noise, sensitivity, bias, hysteresis, etc.
#   Other parameters: launch tower angle, launch site altitude (
#   we need this for later pressure calculations), local magnetic field data.

# Desired calculations:
# -Absolute position (ENU) --> East, North, Up coordinates
# -Velocity
# -Acceleration
# -Orientation
# -Barometric pressure
# -Temperature
# -Acceleration (x, y, z) --> need to add
# -Angular rates (x, y, z) --> need to add
# -Magnetic field (B field) components (x, y, z) --> need to add

# General assumptions made:
# -The rocket's yaw rate is constant and zero.
# -The rocket's pitch rate is constant and relatively small (~1 degree/s).
# -The angle of attack is zero.

# Desired calculations:
# -Absolute position (ENU) --> East, North, Up coordinates
# -Velocity
# -Acceleration
# -Orientation
# -Barometric pressure
# -Temperature
# -Acceleration (x, y, z)
# -Angular rates (x, y, z)
# -Magnetic field (B field) components (x, y, z)


class Rocket:
    def __init__(self, mass, thrust, burn_time, pressure_noise, temp_noise, accel_noise, gyro_noise, mag_noise,
                 position_loc_cart, position_enu, velocity, vel_unit_vec, acceleration, orientation, baro_pressure,
                 temperature, position_e, position_n, altitude, orien_q0, orien_q1, orien_q2, orien_q3, air_speed):
        self.mass = mass
        self.thrust = thrust
        self.burn_time = burn_time
        self.pressure_noise = pressure_noise
        self.temp_noise = temp_noise
        self.accel_noise = accel_noise
        self.gyro_noise = gyro_noise
        self.mag_noise = mag_noise
        self.position_loc_cart = position_loc_cart
        self.position_enu = position_enu
        self.velocity = velocity
        self.vel_unit_vec = vel_unit_vec
        self.acceleration = acceleration
        self.orientation = orientation
        self.baro_pressure = baro_pressure
        self.temperature = temperature  # in Celsius
        self.position_e = position_e
        self.position_n = position_n
        self.altitude = altitude
        self.orien_q0 = orien_q0
        self.orien_q1 = orien_q1
        self.orien_q2 = orien_q2
        self.orien_q3 = orien_q3
        self.air_speed = air_speed


# Temporary drag coefficient for calculating drag force
coeff_drag = 0.3

# Mass lost per timestep
# TODO: implement the proper loss
mass_loss_per_timestep = 0


# Converts a distance in ft to a distance in meters (VERSION 1)
def ft_to_meters(distance_ft: np.array([])) -> np.array([]):
    for i, val in enumerate(distance_ft):
        distance_ft[i] = val / 3.2808
    return distance_ft


# (VERSION 2) Consider using this version
def ft_to_meters_2(distance_ft: np.array([])) -> float:
    return distance_ft / 3.2808


# Converts a distance in meters to a distance in ft (VERSION 1)
def meters_to_ft(distance_meters: np.array([])) -> np.array([]):
    for i, val in enumerate(distance_meters):
        distance_meters[i] = val / 3.2808
    return distance_meters


# (VERSION 2) Consider using this version
def meters_to_ft_2(distance_meters: np.array([])) -> np.array([]):
    return distance_meters * 3.2808


# Calculates gravity based on altitude above sea level (in ft)
def get_gravity(altitude: float) -> float:
    return float(9.79121 * ((20902231 / (20902231 + altitude)) ** 2))  # Gravity constant specific to Spaceport America


# Calculates mass based on estimated loss in mass
# TODO: implement for variable mass system
def get_mass(current_rocket: Rocket) -> float:
    return current_rocket.mass - mass_loss_per_timestep


# Calculates the magnitude of the velocity vector
def get_vel_magnitude(current_rocket: Rocket) -> np.array([]):
    return np.sqrt(current_rocket.velocity[0] ** 2 + current_rocket.velocity[1] ** 2 +
                   current_rocket.velocity[2] ** 2)


# Calculates the velocity unit vector
# Launch angle is in degrees
# TODO: fix the fact the unit vector is negative and switches sign
def get_vel_unit_vector(current_rocket: Rocket, launch_angle: float) -> np.array([]):
    if not get_vel_magnitude(current_rocket):
        # print("Magnitude of velocity is 0.\n")
        return current_rocket.velocity
    return current_rocket.velocity / get_vel_magnitude(current_rocket)


# Calculates air density from 3rd-degree polynomial based on plot of air density vs altitude data obtained from [VERSION 1]
# https://www.engineeringtoolbox.com/standard-atmosphere-d_604.html
def get_air_density(current_rocket: Rocket) -> float:
    alt = current_rocket.altitude
    return (-1.2694e-14 * (alt ** 3)) + (1.9121e-09 * (alt ** 2)) + (-8.7147e-05 * alt) + 1.1644e+00


# Alternative to other air density calculation [VERSION 2]
def get_air_density2(current_rocket: Rocket) -> float:
    alt = current_rocket.altitude
    return meters_to_ft(1.22 * 0.9**(alt / 1000))  # TODO: meters_to_ft doesn't work without an array so this doesn't work


# Calculates the rocket's cross-sectional area. The 0.25 is the current radius (in ft) of the rocket (SOTS).
def get_cross_sec_area() -> float:
    temp = np.array([0.25])
    rocket_radius = ft_to_meters(temp)
    return np.pi * (rocket_radius[0] ** 2)


# Calculates the drag force experienced by the rocket in local cartesian vector form [x,y,z]
def get_drag_force(current_rocket: Rocket) -> np.array([]):
    drag_force_mag = 0.5 * coeff_drag * (get_vel_magnitude(current_rocket) ** 2) * get_air_density(current_rocket) * \
                 get_cross_sec_area()
    return get_vel_unit_vector(current_rocket, 90) * drag_force_mag  # TODO: make the launch angle not hard-coded


# TODO: implement burn time
# Calculates the thrust vector based on velocity unit vector
def get_thrust(current_rocket: Rocket, current_time: float) -> np.array([]):
    if current_time <= current_rocket.burn_time:
        thrust_magnitude = np.sqrt(current_rocket.thrust[0] ** 2 + current_rocket.thrust[1] ** 2
                                                          + current_rocket.thrust[2] ** 2)
        print("This is vel_unit_vec: " + str(get_vel_unit_vector(current_rocket, 90)))
        return get_vel_unit_vector(current_rocket, 90) * thrust_magnitude  # TODO: make the launch angle not hard-coded
    print("Made it here")
    return np.array([0, 0, 0])


# Calculates acceleration of the rocket in local cartesian vector form [x,y,z]
# TODO: check units for everything
def rocket_acceleration(current_rocket: Rocket, previous_time: float, current_time: float) -> np.array([]):
    resultant_force = np.array([0.0, 0.0, 0.0])
    drag_force = get_drag_force(current_rocket)
    thrust_force = get_thrust(current_rocket, current_time)
    resultant_force[0] = thrust_force[0] - drag_force[0]
    resultant_force[1] = thrust_force[1] - drag_force[1]
    resultant_force[2] = thrust_force[2] - ((current_rocket.mass * get_gravity(current_rocket.altitude)) +
                                            drag_force[2])
    print("This is thrust: " + str(thrust_force[2]))
    print("This is Fg: " + str(current_rocket.mass * get_gravity(current_rocket.altitude)))
    print("This is Fd: " + str(drag_force[2]))

    if any(resultant_force):  # TODO: check for better way
        return current_rocket.acceleration + resultant_force / current_rocket.mass
    return current_rocket.acceleration


# TODO: edit this due to the fact that methodology changed
# Calculates the current cartesian position of the rocket [x,y,z]
# Use this: https://apps.dtic.mil/dtic/tr/fulltext/u2/a484864.pdf
# And this: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
def rocket_position_local_cartesian(current_rocket: Rocket, previous_time: float, current_time: float) -> np.array([]):
    return current_rocket.position_loc_cart + current_rocket.velocity * (current_time - previous_time) + 0.5 * \
           current_rocket.acceleration * (current_time - previous_time)**2


# TODO: delete this once new function is done
# Converting cartesian to enu for position of rocket
def cartesian_to_enu(current_rocket: Rocket, position_cartesian) -> np.array([]):
    position_cartesian = ft_to_meters(position_cartesian)
    geodectic_coords = pymap3d.ecef2geodetic(position_cartesian[0], position_cartesian[1], position_cartesian[2], None,
                                             True)
    current_rocket.position_ENU = meters_to_ft(pymap3d.ecef2enu(position_cartesian[0], position_cartesian[1],
                                                                position_cartesian[2], geodectic_coords[0],
                                                                geodectic_coords[1], geodectic_coords[2], None, True))


# TODO: add new function for converting from cartesian to enu
# Converting local cartesian to ENU for position of rocket
# Use this for quaternion rotation: https://kieranwynn.github.io/pyquaternion/
def loc_cart_to_enu_2(current_rocket: Rocket, previous_time: float, current_time: float) -> np.array([]):
    return np.array([0.0, 0.0, 0.0])
    # rot_quaternion = Quaternion(axis=[0, 0, 0], angle=0)  # TODO: make this the true scalar and vector
    # return rot_quaternion.rotate(current_rocket.position_loc_cart)


# Calculates velocity of the rocket by integrating acceleration in local cartesian vector form [x,y,z]
def rocket_velocity(current_rocket: Rocket, previous_time: float, current_time: float) -> np.array([]):
    return current_rocket.velocity + current_rocket.acceleration * (current_time - previous_time)


