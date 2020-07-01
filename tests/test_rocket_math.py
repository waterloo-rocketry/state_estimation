import pytest

import rocket_math as rm
import numpy as np


def test_get_gravity():
    test_rocket = rm.Rocket()
    assert test_rocket.get_gravity()
    # skipping for now because requires hand calcs


def test_get_mass():
    # Test for mass above mass loss from one timestep
    mass = {"total_mass": 110, "body_mass": 55, "comb_mass": 55}
    test_rocket = rm.Rocket(mass=mass)
    assert test_rocket.get_mass() == 110 - rm.MASS_LOSS
    assert test_rocket.get_mass() == 110 - (2 * rm.MASS_LOSS)
    assert test_rocket.mass["comb_mass"] == 55 - (2 * rm.MASS_LOSS)
    assert test_rocket.mass["body_mass"] == 55
    assert test_rocket.mass["total_mass"] == 110 - (2 * rm.MASS_LOSS)

    # Test for mass below mass loss from one timestep
    mass = {"total_mass": 100 + rm.MASS_LOSS - 0.01, "body_mass": 100, "comb_mass": rm.MASS_LOSS - 0.01}
    test_rocket.mass = mass
    assert test_rocket.get_mass() == 100 - rm.MASS_LOSS - 0.01
    assert test_rocket.get_mass() == 100 - rm.MASS_LOSS - 0.01  # only 1 mass loss because comb_mass = 0
    assert test_rocket.mass["comb_mass"] == 0  # 0 because the mass loss should only be subtracted twice
    assert test_rocket.mass["body_mass"] == 100
    assert test_rocket.mass["total_mass"] == 100 - rm.MASS_LOSS - 0.01  # only 1 mass loss because comb_mass = 0


def test_get_vel_magnitude():
    # Test for no velocity (velocity vector is zero vector)
    test_rocket = rm.Rocket(velocity=np.array([0, 0, 0]))
    assert test_rocket.get_velocity_mag() == 0

    # Test for velocity vector with positive integers
    test_rocket.velocity = np.array([1, 1, 1])
    assert test_rocket.get_velocity_mag() == np.sqrt(3)

    # Test for velocity vector with negative integers
    test_rocket.velocity = np.array([-1, -1, -1])
    assert test_rocket.get_velocity_mag() == np.sqrt(3)

    # Test for velocity vector with positive floats
    test_rocket.velocity = np.array([2.5, 3.5, 4.5])
    assert test_rocket.get_velocity_mag() == np.sqrt(38.75)


def test_get_vel_unit_vector():
    # Test for no velocity (velocity vector is zero vector)
    test_rocket = rm.Rocket(velocity=np.array([0, 0, 0]))
    assert test_rocket.get_velocity_uv() == np.array([0, 0, 0])

    # Test for velocity vector with positive integers
    test_rocket.velocity = np.array([1, 1, 1])
    assert test_rocket.get_velocity_uv() == np.array([(1/np.sqrt(3)), (1/np.sqrt(3)), (1/np.sqrt(3))])

    # Test for velocity vector with negative integers
    test_rocket.velocity = np.array([-1, -1, -1])
    assert test_rocket.get_velocity_uv() == np.array([(-1 / np.sqrt(3)), (-1 / np.sqrt(3)), (-1 / np.sqrt(3))])

    # Test for velocity vector with positive floats
    test_rocket.velocity = np.array([2.5, 3.5, 4.5])
    assert test_rocket.get_velocity_uv() == np.array([(2.5/np.sqrt(38.75)), (3.5/np.sqrt(38.75)), (4.5/np.sqrt(38.75))])


def test_get_air_density():
    test_rocket = rm.Rocket()
    test_rocket.altitude = 0
    assert test_rocket.get_air_density(test_rocket) == 23.77
    # skipping for now because requires hand calcs


def test_get_drag_force():
    # Test for no velocity (velocity vector is zero vector)
    test_rocket = rm.Rocket(velocity=np.array([0, 0, 0]))
    test_rocket.altitude = 0  # parameter for air density
    assert test_rocket.get_drag_force() == np.array([0, 0, 0])

    # Test for velocity vector with positive integers
    test_rocket.velocity = np.array([1, 1, 1])
    test_rocket.altitude = 5000
    assert test_rocket.get_drag_force() == np.array([0.3482, 0.3482, 0.3482])     # hand calc

    # Test for velocity vector with negative integers
    test_rocket.velocity = np.array([-1, -1, -1])
    test_rocket.altitude = 5000
    assert test_rocket.get_drag_force() == np.array([-0.3482, -0.3482, -0.3482])  # hand calc

    # Test for velocity vector with positive floats
    test_rocket.velocity = np.array([32.5, 42.5, 52.5])
    test_rocket.altitude = 10000
    assert test_rocket.get_drag_force() == np.array([0.2242, 0.2932, 0.3622])     # hand calc


def test_get_thrust():
    # Test for no thrust (thrust vector is zero vector)
    test_rocket = rm.Rocket(thrust=np.array[0, 0, 0])
    current_time = 0
    assert test_rocket.get_thrust(current_time) == np.array([0, 0, 0])

    # Test for thrust and velocity vectors with positive integers
    test_rocket.thrust = np.array([1, 1, 1])
    test_rocket.velocity = np.array([1, 1, 1])
    current_time = 0
    assert test_rocket.get_thrust(current_time) == np.array([1, 1, 1])

    # Test for thrust with negative integers, velocity with positive integers
    test_rocket.thrust = np.array([-1, -1, -1])
    test_rocket.velocity = np.array([1, 1, 1])
    current_time = 0
    assert test_rocket.get_thrust(current_time) == np.array([-1, -1, -1])

    # Test for current time greater than burn time
    test_rocket.thrust = np.array([1, 1, 1])
    test_rocket.velocity = np.array([1, 1, 1])
    test_rocket.burn_time = 5
    current_time = 10
    assert test_rocket.get_thrust(current_time) == np.array([0, 0, 0])

    # Test for thrust and velocity with positive floats
    test_rocket.thrust = np.array([10.5, 11.5, 12.5])
    test_rocket.velocity = np.array([10.5, 11.5, 12.5])
    test_rocket.burn_time = 5
    current_time = 4
    assert test_rocket.get_thrust(current_time) == np.array([10.5, 11.5, 12.5])

    # Need a test with random floats for thrust and velocity but can't do the hand calc rn.


def test_rocket_acceleration():
    # Test for no thrust or velocity (thrust and velocity vectors are zero vector)
    mass = {"total_mass": 110, "body_mass": 55, "comb_mass": 55}
    thrust = np.array([0, 0, 0])
    test_rocket = rm.Rocket(mass=mass, thrust=thrust)
    test_rocket.velocity = np.array([0, 0, 0])
    current_time, previous_time = 0
    assert test_rocket.rocket_acceleration(current_time, previous_time) == -test_rocket.get_gravity()

    # Test for constant acceleration/no resultant force (z component of thrust vector equals mass * gravity with no velocity)
    mass = {"total_mass": 110, "body_mass": 55, "comb_mass": 55}
    test_acceleration, test_rocket.acceleration = np.array([0, 0, 0])
    thrust = np.array([0, 0, test_rocket.get_gravity()])
    test_rocket.mass = mass
    test_rocket.thrust = thrust
    test_rocket.velocity = np.array([0, 0, 0])
    current_time, previous_time = 0
    assert test_rocket.rocket_acceleration(current_time, previous_time) == test_acceleration

    # Test for thrust and velocity with positive integers
    mass = {"total_mass": 110, "body_mass": 55, "comb_mass": 55}
    thrust = np.array([1, 1, 1])
    test_rocket.mass = mass
    test_rocket.thrust = thrust
    test_rocket.velocity = np.array([1, 1, 1])
    current_time, previous_time = 0
    assert test_rocket.rocket_acceleration(current_time, previous_time) == np.array([])

    # Test for thrust and velocity with with negative integers
    mass = {"total_mass": 110, "body_mass": 55, "comb_mass": 55}
    thrust = np.array([-1, -1, -1])
    test_rocket.mass = mass
    test_rocket.thrust = thrust
    test_rocket.velocity = np.array([-1, -1, -1])
    current_time, previous_time = 0
    assert test_rocket.rocket_acceleration(current_time, previous_time) == np.array([])

    # Test for thrust and velocity with positive floats
    mass = {"total_mass": 110, "body_mass": 55, "comb_mass": 55}
    test_acceleration, test_rocket.acceleration = np.array([0, 0, 0])
    thrust = np.array([0, 0, test_rocket.get_gravity()])
    test_rocket.mass = mass
    test_rocket.thrust = thrust
    test_rocket.velocity = np.array([0, 0, 0])
    current_time, previous_time = 0
    assert test_rocket.rocket_acceleration(current_time, previous_time) == test_acceleration


def test_rocket_velocity():
    test_rocket = rm.Rocket()
    assert test_rocket.rocket_velocity()


def test_rocket_position():
    test_rocket = rm.Rocket()
    assert test_rocket.rocket_position()
