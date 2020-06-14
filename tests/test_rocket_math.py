import pytest

import rocket_math as rm
import numpy as np


def test_ft_to_meters():
    assert rm.ft_to_meters(1) == np.array([0.3048])     # hand calc
    assert rm.ft_to_meters(-1) == np.array([-0.3048])   # hand calc
    assert rm.ft_to_meters(13) == np.array([3.9624])    # hand calc
    assert rm.ft_to_meters(0) == np.array([0])
    assert rm.ft_to_meters(2.6) == np.array([0.79248])  # hand calc


def test_meters_to_ft():
    assert rm.meters_to_ft(1) == np.array([3.2808])     # hand calc
    assert rm.meters_to_ft(-1) == np.array([-3.2808])   # hand calc
    assert rm.meters_to_ft(13) == np.array([42.6509])   # hand calc
    assert rm.meters_to_ft(0) == np.array([0])
    assert rm.meters_to_ft(2.6) == np.array([8.5302])   # hand calc


def test_get_gravity():
    assert rm.get_gravity()
    # skipping for now because requires hand calcs


def test_get_mass():
    mass = {"total_mass": 110, "body_mass": 55, "comb_mass": 55}
    test_rocket = rm.Rocket(mass)
    assert rm.get_mass(test_rocket) == 110 - rm.MASS_LOSS
    assert rm.get_mass(test_rocket) == 110 - (2 * rm.MASS_LOSS)
    assert test_rocket.mass["comb_mass"] == 55 - (2 * rm.MASS_LOSS)
    assert test_rocket.mass["body_mass"] == 55
    assert test_rocket.mass["total_mass"] == 110 - (2 * rm.MASS_LOSS)

    mass = {"total_mass": 100 + rm.MASS_LOSS - 0.01, "body_mass": 100, "comb_mass": rm.MASS_LOSS - 0.01}
    test_rocket = rm.Rocket(mass)
    assert rm.get_mass(test_rocket) == 100 - rm.MASS_LOSS - 0.01
    assert rm.get_mass(test_rocket) == 100 - rm.MASS_LOSS - 0.01  # only 1 mass loss because comb_mass = 0
    assert test_rocket.mass["comb_mass"] == 0  # 0 because the mass loss should only be subtracted twice
    assert test_rocket.mass["body_mass"] == 100
    assert test_rocket.mass["total_mass"] == 100 - rm.MASS_LOSS - 0.01  # only 1 mass loss because comb_mass = 0


def test_get_vel_magnitude():
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([0, 0, 0])
    assert rm.get_vel_magnitude(test_rocket) == 0

    test_rocket.velocity = np.array([1, 1, 1])
    assert rm.get_vel_magnitude(test_rocket) == np.sqrt(3)

    test_rocket.velocity = np.array([2.5, 3.5, 4.5])
    assert rm.get_vel_magnitude(test_rocket) == np.sqrt(38.75)

    test_rocket.velocity = np.array([-1, -1, -1])
    assert rm.get_vel_magnitude(test_rocket) == np.sqrt(3)


def test_get_vel_unit_vector():
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([0, 0, 0])
    assert rm.get_vel_unit_vector(test_rocket) == np.array([0, 0, 0])

    test_rocket.velocity = np.array([1, 1, 1])
    assert rm.get_vel_unit_vector(test_rocket) == np.array([(1/np.sqrt(3)), (1/np.sqrt(3)), (1/np.sqrt(3))])

    test_rocket.velocity = np.array([2.5, 3.5, 4.5])
    assert rm.get_vel_unit_vector(test_rocket) == np.array([(2.5/np.sqrt(38.75)), (3.5/np.sqrt(38.75)), (4.5/np.sqrt(38.75))])

    test_rocket.velocity = np.array([-1, -1, -1])
    assert rm.get_vel_unit_vector(test_rocket) == np.array([(-1/np.sqrt(3)), (-1/np.sqrt(3)), (-1/np.sqrt(3))])


def test_get_air_density():
    test_rocket = rm.Rocket()
    test_rocket.altitude = 0
    assert rm.get_air_density(test_rocket) == 23.77
    # skipping for now because requires hand calcs


def test_get_cross_sec_area():
    assert rm.get_cross_sec_area()
    # skipping due to fact function is becoming a constant


def test_get_drag_force():
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([0, 0, 0])
    test_rocket.altitude = 0  # parameter for air density
    assert rm.get_drag_force(test_rocket) == np.array([0, 0, 0])

    test_rocket.velocity = np.array([1, 1, 1])
    test_rocket.altitude = 5000
    assert rm.get_drag_force(test_rocket) == np.array([0.3482, 0.3482, 0.3482])     # hand calc

    test_rocket.velocity = np.array([-1, -1, -1])
    test_rocket.altitude = 5000
    assert rm.get_drag_force(test_rocket) == np.array([-0.3482, -0.3482, -0.3482])  # hand calc

    test_rocket.velocity = np.array([32.5, 42.5, 52.5])
    test_rocket.altitude = 10000
    assert rm.get_drag_force(test_rocket) == np.array([0.2242, 0.2932, 0.3622])     # hand calc


def test_get_thrust():
    assert rm.get_thrust()


def test_rocket_acceleration():
    assert rm.rocket_acceleration()


def test_rocket_position():
    assert rm.rocket_position()


def test_rocket_velocity():
    assert rm.rocket_velocity()
