# This file defines the Rocket class and the general math functions needed to calculate parameters for data generation

import numpy as np
import pymap3d
from math import sin, cos, radians
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
        self.mass = mass  # in pounds (lb)
        self.thrust = thrust  # in pound-force (lbf)
        self.burn_time = burn_time  # in seconds (s)
        self.pressure_noise = pressure_noise  # not determined
        self.temp_noise = temp_noise  # not determined
        self.accel_noise = accel_noise  # not determined
        self.gyro_noise = gyro_noise  # not determined
        self.mag_noise = mag_noise  # not determined
        self.position_loc_cart = position_loc_cart  # in feet (ft)
        self.position_enu = position_enu  # in feet (ft)
        self.velocity = velocity  # in feet/second (ft/s)
        self.vel_unit_vec = vel_unit_vec  # no units because unit vector
        self.acceleration = acceleration  # in feet/second^2 (ft/s^2)
        self.orientation = orientation  # not determined
        self.baro_pressure = baro_pressure  # in atmospheres (atm)
        self.temperature = temperature  # in fahrenheit (F)
        self.position_e = position_e  # in feet probably (ft)
        self.position_n = position_n  # in feet probably (ft)
        self.altitude = altitude  # in feet (ft)
        self.orien_q0 = orien_q0  # not determined
        self.orien_q1 = orien_q1  # not determined
        self.orien_q2 = orien_q2  # not determined
        self.orien_q3 = orien_q3  # not determined
        self.air_speed = air_speed  # in feet/second probably (ft/s)


# TODO: add computed/determined drag coefficient
# Temporary drag coefficient for calculating drag force
coeff_drag = 0.3
# Current radius (in ft) of the rocket (SOTS)
tube_radius = 0.25

# Mass lost per timestep
# TODO: implement the proper loss
mass_loss_per_timestep = 0.05


# Converts a distance in feet to a distance in meters
def ft_to_meters(distance_ft: np.array([])) -> float:
    return distance_ft / 3.2808


# Converts a distance in meters to a distance in feet
def meters_to_ft(distance_meters: np.array([])) -> np.array([]):
    return distance_meters * 3.2808


# Calculates gravity based on altitude above sea level (in ft/s^2)
def get_gravity(altitude: float) -> float:
    print("This is gravity coefficient: " + str((20902231 / (20902231 + ft_to_meters(altitude))) ** 2))
    print("This is gravity: " + str(meters_to_ft(9.79121 * ((20902231 / (20902231 + ft_to_meters(altitude))) ** 2))))
    return meters_to_ft(9.79121 * ((20902231 / (20902231 + ft_to_meters(altitude))) ** 2))  # Gravity constant specific to Spaceport America


# Calculates mass based on estimated loss in mass (in lb)
# TODO: implement for variable mass system
def get_mass(current_rocket: Rocket) -> float:
    return current_rocket.mass - mass_loss_per_timestep


# Calculates the magnitude of the velocity vector (in lbf)
def get_vel_magnitude(current_rocket: Rocket) -> float:
    print("This is vel mag: " + str(np.sqrt(current_rocket.velocity[0] ** 2 + current_rocket.velocity[1] ** 2 +
                   current_rocket.velocity[2] ** 2)))
    return np.sqrt(current_rocket.velocity[0] ** 2 + current_rocket.velocity[1] ** 2 + current_rocket.velocity[2] ** 2)


# Calculates the velocity unit vector
# Launch angle is in degrees and is from the x-axis to the z-axis
# TODO: fix the fact the unit vector is negative and switches sign
def get_vel_unit_vector(current_rocket: Rocket, launch_angle: float) -> np.array([]):
    if not get_vel_magnitude(current_rocket):
        print("Magnitude of velocity is 0.")
        multiplier_arr = np.array([round(cos(radians(launch_angle)), 4), 0.0, round(sin(radians(launch_angle)), 4)])
        print("This is cos: " + str(round(cos(radians(launch_angle)), 4)))
        print("This is sin: " + str(round(sin(radians(launch_angle)), 4)))
        return np.array([1.0, 0.0, 1.0]) * multiplier_arr
    print("In get vector & This is velocity: " + str(current_rocket.velocity))
    print("This is velocity_magnitude: " + str(get_vel_magnitude(current_rocket)))
    return current_rocket.velocity / get_vel_magnitude(current_rocket)


# Other way of doing function:
# return rocket.current_rocket.velocity / get_vel_magnitude(current_rocket)


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
    return 1.22 * 0.9**(alt / 1000)


# Calculates the rocket's cross-sectional area [m^2]
def get_cross_sec_area() -> float:
    temp = np.array([tube_radius])
    rocket_radius = ft_to_meters(temp)
    return np.pi * (rocket_radius[0] ** 2)


# Calculates the drag force experienced by the rocket in local cartesian vector form [x,y,z] (in ft)
def get_drag_force(current_rocket: Rocket) -> np.array([]):
    drag_force_mag = 0.5 * coeff_drag * (get_vel_magnitude(current_rocket) ** 2) * get_air_density(current_rocket) * \
                 get_cross_sec_area()
    print("This is currenty vel_mag in Fd: " + str(get_vel_magnitude(current_rocket)))
    print("This is vel_mag**2 + air density + area: " + str(get_vel_magnitude(current_rocket) ** 2) + " " + str(get_air_density(current_rocket)) + " " + str(get_cross_sec_area()))
    return current_rocket.vel_unit_vec * drag_force_mag  # TODO: make the launch angle not hard-coded


# TODO: implement burn time
# Calculates the thrust vector based on velocity unit vector [x,y,z] (in lbf)
def get_thrust(current_rocket: Rocket, current_time: float) -> np.array([]):
    if current_time <= current_rocket.burn_time:
        thrust_magnitude = np.sqrt(current_rocket.thrust[0] ** 2 + current_rocket.thrust[1] ** 2
                                                          + current_rocket.thrust[2] ** 2)
        print("This is vel_unit_vec: " + str(current_rocket.vel_unit_vec))
        return current_rocket.vel_unit_vec * thrust_magnitude  # TODO: make the launch angle not hard-coded
    print("Made it here")
    return np.array([0, 0, 0])


# Calculates acceleration of the rocket in local cartesian vector form [x,y,z]
# TODO: check units for everything
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
