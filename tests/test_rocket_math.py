# Test suite for rocket_math.py

import numpy as np
from pyquaternion import Quaternion

import rocket_math as rm

'''
Notes:
- For all calculations, there will be an tolerance of 0.001.
- No rocket_air_density unit test since the current method for
  determining/testing air density needs to be changed.
'''


# Testing functions for gravity().
def test_gravity_at_ground():
    test_rocket = rm.Rocket()
    assert abs(test_rocket.gravity() - 32.1389) <= rm.TOLERANCE


def test_gravity_at_100_ft():
    test_rocket = rm.Rocket()
    test_rocket.altitude = 100
    assert abs(test_rocket.gravity() - 32.1386) <= rm.TOLERANCE


def test_gravity_at_1000_ft():
    test_rocket = rm.Rocket()
    test_rocket.altitude = 1000
    assert abs(test_rocket.gravity() - 32.1358) <= rm.TOLERANCE


def test_gravity_at_10000_ft():
    test_rocket = rm.Rocket()
    test_rocket.altitude = 10000
    assert abs(test_rocket.gravity() - 32.1082) <= rm.TOLERANCE


def test_gravity_at_100000_ft():
    test_rocket = rm.Rocket()
    test_rocket.altitude = 100000
    assert abs(test_rocket.gravity() - 31.8336) <= rm.TOLERANCE


def test_gravity_at_float_num_ft():
    test_rocket = rm.Rocket()
    test_rocket.altitude = 5432.10
    assert abs(test_rocket.gravity() - 32.1222) <= rm.TOLERANCE


# Testing functions for update_mass().
def test_update_mass_above_mass_loss_threshold():
    """
    Test update_mass() for mass above the mass loss threshold from one
    timestep.
    """
    timestep = 1
    test_rocket = rm.Rocket()
    test_rocket.mass = {"total_mass": 110, "body_mass": 55, "prop_mass": 55}
    test_rocket.mass = test_rocket.update_mass(timestep)
    expected_mass = {"total_mass": 110 - rm.MASS_LOSS * timestep,
                     "body_mass": 55,
                     "prop_mass": 55 - rm.MASS_LOSS * timestep}
    assert all((test_rocket.mass[key] - expected_mass[key]) <= rm.TOLERANCE
               for key in test_rocket.mass)
    # assert test_rocket.update_mass(timestep) == {
    #    "total_mass": 110 - rm.MASS_LOSS * timestep,
    #    "body_mass": 55,
    #   "prop_mass": 55 - rm.MASS_LOSS * timestep}


def test_secondary_update_mass_above_mass_loss_threshold():
    """
    Test update_mass() again for mass above the mass loss threshold from one
    timestep to make sure mass decreases properly.
    """
    timestep = 1
    test_rocket = rm.Rocket()
    test_rocket.mass = {"total_mass": 110 - rm.MASS_LOSS * timestep,
                        "body_mass": 55,
                        "prop_mass": 55 - rm.MASS_LOSS * timestep}
    test_rocket.mass = test_rocket.update_mass(timestep)
    expected_mass = {"total_mass": 110 - (2 * rm.MASS_LOSS * timestep),
                     "body_mass": 55,
                     "prop_mass": 55 - (2 * rm.MASS_LOSS * timestep)}
    assert all((test_rocket.mass[key] - expected_mass[key]) <= rm.TOLERANCE
               for key in test_rocket.mass)


def test_update_mass_below_mass_loss_threshold():
    """
    Test update_mass() for mass below the mass threshold loss from one
    timestep.
    """
    timestep = 1
    test_rocket = rm.Rocket()
    test_rocket.mass = {"total_mass": 100 + rm.MASS_LOSS - 0.01,
                        "body_mass": 100, "prop_mass": rm.MASS_LOSS - 0.01}
    test_rocket.mass = test_rocket.update_mass(timestep)
    expected_mass = {"total_mass": 100,
                     "body_mass": 100,
                     "prop_mass": 0}
    assert all((test_rocket.mass[key] - expected_mass[key]) <= rm.TOLERANCE
               for key in test_rocket.mass)


def test_secondary_update_mass_below_mass_loss_threshold():
    """
    Test update_mass() again for mass below the mass loss threshold from one
    timestep to make sure mass does not decrease.
    """
    timestep = 1
    test_rocket = rm.Rocket()
    test_rocket.mass = {"total_mass": 100, "body_mass": 100, "prop_mass": 0}
    test_rocket.mass = test_rocket.update_mass(timestep)
    expected_mass = {"total_mass": 100,
                     "body_mass": 100,
                     "prop_mass": 0}
    assert all((test_rocket.mass[key] - expected_mass[key]) <= rm.TOLERANCE
               for key in test_rocket.mass)


# Testing functions for speed().
def test_speed_with_no_velocity():
    """
    Test speed() for no velocity (velocity vector is zero
    vector).
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([0, 0, 0])
    assert test_rocket.speed() == 0


def test_speed_with_positive_ints():
    """
    Test speed() for velocity vector with positive integers.
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([1, 1, 1])
    assert test_rocket.speed() == np.sqrt(3)


def test_speed_with_negative_ints():
    """
    Test speed() for velocity vector with negative integers.
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([-1, -1, -1])
    assert test_rocket.speed() == np.sqrt(3)


def test_speed_with_floats():
    """
    Test speed() for velocity vector with floats.
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([2.5, 3.5, 4.5])
    assert test_rocket.speed() == np.sqrt(38.75)


# Testing functions for velocity_uv().
def test_velocity_uv_with_no_velocity():
    """
    Test velocity_uv() for no velocity (velocity vector is zero vector).
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([0, 0, 0])
    assert np.all(test_rocket.velocity_uv() == np.array([0, 0, 0]))


def test_velocity_uv_with_positive_ints():
    """
    Test velocity_uv() for velocity vector with positive integers.
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([1, 1, 1])
    assert np.all(test_rocket.velocity_uv() == np.array(
        [(1 / np.sqrt(3)), (1 / np.sqrt(3)), (1 / np.sqrt(3))]))


def test_velocity_uv_with_negative_ints():
    """
    Test velocity_uv() for velocity vector with negative integers.
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([-1, -1, -1])
    assert np.all(test_rocket.velocity_uv() == np.array(
        [(-1 / np.sqrt(3)), (-1 / np.sqrt(3)), (-1 / np.sqrt(3))]))


def test_velocity_uv_with_floats():
    """
    Test velocity_uv() for velocity vector with floats.
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([2.5, 3.5, 4.5])
    assert np.all(test_rocket.velocity_uv() == np.array(
        [(2.5 / np.sqrt(38.75)), (3.5 / np.sqrt(38.75)),
         (4.5 / np.sqrt(38.75))]))


# Testing functions for air_density().
def test_air_density_at_ground():
    test_rocket = rm.Rocket()
    test_rocket.temperature = 15.04
    test_rocket.baro_pressure = 101.4009
    assert abs(test_rocket.air_density() - 1.2266) <= rm.TOLERANCE


def test_air_density_at_100_m():
    test_rocket = rm.Rocket()
    test_rocket.temperature = 14.391
    test_rocket.baro_pressure = 100.2062
    assert abs(test_rocket.air_density() - 1.2149) <= rm.TOLERANCE


def test_air_density_at_1000_m():
    test_rocket = rm.Rocket()
    test_rocket.temperature = 8.55
    test_rocket.baro_pressure = 89.9581
    assert abs(test_rocket.air_density() - 1.1133) <= rm.TOLERANCE


def test_air_density_at_10000_m():
    test_rocket = rm.Rocket()
    test_rocket.temperature = -49.86
    test_rocket.baro_pressure = 26.5162
    assert abs(test_rocket.air_density() - 0.4140) <= rm.TOLERANCE


def test_air_density_at_20000_m():
    test_rocket = rm.Rocket()
    test_rocket.temperature = -56.46
    test_rocket.baro_pressure = 5.5298
    assert abs(test_rocket.air_density() - 0.08897) <= rm.TOLERANCE


# Testing functions for drag_force().
def test_drag_with_no_velocity(mocker):
    """
    Test drag_force() for no velocity (velocity vector is zero vector).
    """
    test_rocket = rm.Rocket()
    mocker.patch('rocket_math.Rocket.speed', return_value=0)
    mocker.patch('rocket_math.Rocket.velocity_uv',
                 return_value=np.array([0, 0, 0]))
    mocker.patch('rocket_math.Rocket.air_density', return_value=1.22)
    assert np.all(test_rocket.drag_force() == np.array([0, 0, 0]))


def test_drag_with_positive_int_velocity(mocker):
    """
    Test drag_force() for velocity vector with positive integers.
        velocity = [1, 1, 1]
    """
    test_rocket = rm.Rocket()
    mocker.patch('rocket_math.Rocket.speed',
                 return_value=np.sqrt(3))
    mocker.patch('rocket_math.Rocket.velocity_uv',
                 return_value=np.array(
                     [(1 / np.sqrt(3)), (1 / np.sqrt(3)), (1 / np.sqrt(3))]))
    mocker.patch('rocket_math.Rocket.air_density', return_value=0.7204)
    assert np.all((test_rocket.drag_force() - np.array(
        [0.0367, 0.0367, 0.0367])) <= rm.TOLERANCE)


def test_drag_with_negative_int_velocity(mocker):
    """
    Test drag_force() for velocity vector with negative integers.
        velocity = [-1, -1, -1]
    """
    test_rocket = rm.Rocket()
    mocker.patch('rocket_math.Rocket.speed',
                 return_value=np.sqrt(3))
    mocker.patch('rocket_math.Rocket.velocity_uv',
                 return_value=np.array([(-1 / np.sqrt(3)), (-1 / np.sqrt(3)),
                                        (-1 / np.sqrt(3))]))
    mocker.patch('rocket_math.Rocket.air_density', return_value=0.7204)
    assert np.all((test_rocket.drag_force() - np.array(
        [-0.0367, -0.0367, -0.0367])) <= rm.TOLERANCE)


def test_drag_with_float_velocity(mocker):
    """
    Test drag_force() for velocity vector with floats.
        velocity = [32.5, 42.5, 52.5]
    """
    test_rocket = rm.Rocket()
    mocker.patch('rocket_math.Rocket.speed',
                 return_value=np.sqrt(5618.75))
    mocker.patch('rocket_math.Rocket.velocity_uv',
                 return_value=np.array(
                     [(32.5 / np.sqrt(5618.75)), (42.5 / np.sqrt(5618.75)),
                      (52.5 / np.sqrt(5618.75))]))
    mocker.patch('rocket_math.Rocket.air_density', return_value=0.4254)
    assert np.all((test_rocket.drag_force() - np.array(
        [30.5226, 39.9142, 49.3058])) <= rm.TOLERANCE)


# Testing functions for update_thrust().
def test_update_thrust_with_no_thrust_or_velocity(mocker):
    """
    Test update_thrust() for no thrust or velocity (thrust and velocity vectors
    are zero vectors).
    """
    test_rocket = rm.Rocket()
    current_time = 0
    mocker.patch('rocket_math.Rocket.speed', return_value=0)
    mocker.patch('rocket_math.Rocket.velocity_uv',
                 return_value=np.array([0, 0, 0]))

    assert np.all(
        test_rocket.update_thrust(current_time) == np.array([0, 0, 0]))


def test_update_thrust_with_pos_thrust_and_no_velocity_ints(mocker):
    """
    Test update_thrust() for thrust vector with positive integers and no
    velocity.
        velocity = [0, 0, 0]
    """
    test_rocket = rm.Rocket()
    test_rocket.burn_time = 10
    test_rocket.thrust = np.array([1, 1, 1])
    current_time = 0
    mocker.patch('rocket_math.Rocket.speed', return_value=0)
    mocker.patch('rocket_math.Rocket.velocity_uv',
                 return_value=np.array([0, 0, 0]))
    assert np.all(
        test_rocket.update_thrust(current_time) == np.array([1, 1, 1]))


def test_update_thrust_with_pos_thrust_and_velocity_ints(mocker):
    """
    Test update_thrust() for thrust and velocity vectors with positive
    integers.
        velocity = [1, 1, 1]
    """
    test_rocket = rm.Rocket()
    test_rocket.burn_time = 10
    test_rocket.thrust = np.array([1, 1, 1])
    current_time = 0
    mocker.patch('rocket_math.Rocket.speed',
                 return_value=np.sqrt(3))
    mocker.patch('rocket_math.Rocket.velocity_uv',
                 return_value=np.array(
                     [(1 / np.sqrt(3)), (1 / np.sqrt(3)), (1 / np.sqrt(3))]))
    assert np.all(
        test_rocket.update_thrust(current_time) == np.array([1, 1, 1]))


def test_update_thrust_with_pos_thrust_and_neg_velocity_ints(mocker):
    """
    Test update_thrust() for thrust vector with positive integers and velocity
    vector with negative integers.
        velocity = [-1, -1, -1]
    """
    test_rocket = rm.Rocket()
    test_rocket.burn_time = 10
    test_rocket.thrust = np.array([-1, -1, -1])
    current_time = 0
    mocker.patch('rocket_math.Rocket.speed',
                 return_value=np.sqrt(3))
    mocker.patch('rocket_math.Rocket.velocity_uv',
                 return_value=np.array(
                     [(1 / np.sqrt(3)), (1 / np.sqrt(3)), (1 / np.sqrt(3))]))
    assert np.all(
        test_rocket.update_thrust(current_time) == np.array([1, 1, 1]))


def test_update_thrust_with_neg_thrust_and_pos_velocity_ints(mocker):
    """
    Test update_thrust() for thrust vector with negative integers and velocity
    vector with positive integers.
        velocity = [1, 1, 1]
    """
    test_rocket = rm.Rocket()
    test_rocket.burn_time = 10
    test_rocket.thrust = np.array([1, 1, 1])
    current_time = 0
    mocker.patch('rocket_math.Rocket.speed',
                 return_value=np.sqrt(3))
    mocker.patch('rocket_math.Rocket.velocity_uv',
                 return_value=np.array(
                     [(-1 / np.sqrt(3)), (-1 / np.sqrt(3)),
                      (-1 / np.sqrt(3))]))
    assert np.all(
        test_rocket.update_thrust(current_time) == np.array([-1, -1, -1]))


def test_update_thrust_with_thrust_and_velocity_floats(mocker):
    """
    Test update_thrust() for thrust and velocity vectors with floats.
        velocity = [10.5, 11.5, 12.5]
    """
    test_rocket = rm.Rocket()
    test_rocket.burn_time = 10
    test_rocket.thrust = np.array([10.5, 11.5, 12.5])
    current_time = 5
    mocker.patch('rocket_math.Rocket.speed',
                 return_value=np.sqrt(398.75))
    mocker.patch('rocket_math.Rocket.velocity_uv',
                 return_value=np.array(
                     [(10.5 / np.sqrt(398.75)), (11.5 / np.sqrt(398.75)),
                      (12.5 / np.sqrt(398.75))]))
    assert np.all(
        test_rocket.update_thrust(current_time) == np.array(
            [10.5, 11.5, 12.5]))


def test_update_thrust_with_greater_current_time(mocker):
    """
    Test update_thrust() for current time greater than burn time.
        thrust = [1, 1, 1]
        velocity = [1, 1, 1]
    """
    test_rocket = rm.Rocket()
    test_rocket.burn_time = 10
    test_rocket.thrust = np.array([1, 1, 1])
    current_time = 20
    mocker.patch('rocket_math.Rocket.speed',
                 return_value=np.sqrt(3))
    mocker.patch('rocket_math.Rocket.velocity_uv',
                 return_value=np.array(
                     [(1 / np.sqrt(3)), (1 / np.sqrt(3)), (1 / np.sqrt(3))]))
    assert np.all(
        test_rocket.update_thrust(current_time) == np.array([0, 0, 0]))


# Testing functions for update_acceleration().
def test_update_acceleration_with_no_thrust_or_drag(mocker):
    """
    Test update_acceleration() for no thrust or drag force (thrust and drag
    force vectors are zero vectors).
    """
    test_rocket = rm.Rocket()
    test_rocket.thrust = np.array([0, 0, 0])
    test_rocket.mass = {"total_mass": 1, "body_mass": 0.5, "prop_mass": 0.5}
    mocker.patch('rocket_math.Rocket.drag_force',
                 return_value=np.array([0, 0, 0]))
    mocker.patch('rocket_math.Rocket.gravity',
                 return_value=32.1389)
    assert np.all(test_rocket.update_acceleration() == np.array(
        [0, 0, -32.1389]))


def test_update_acceleration_with_constant_update_acceleration(mocker):
    """
    Test update_acceleration() for constant acceleration/no resultant force
    (z component of thrust vector equals mass * gravity with no drag force).
    """
    test_rocket = rm.Rocket()
    test_rocket.mass = {"total_mass": 1, "body_mass": 0.5, "prop_mass": 0.5}
    test_rocket.thrust = np.array(
        [0, 0, test_rocket.mass["total_mass"] * 32.1389])
    test_rocket.world_acceleration = np.array([0, 0, 0])
    mocker.patch('rocket_math.Rocket.drag_force',
                 return_value=np.array([0, 0, 0]))
    mocker.patch('rocket_math.Rocket.gravity',
                 return_value=32.1389)
    assert np.all(test_rocket.update_acceleration() == np.array([0, 0, 0]))


def test_update_acceleration_with_pos_thrust_and_drag_ints(mocker):
    """
    Test update_acceleration() for thrust and drag force vectors with positive
    integers.
    """
    test_rocket = rm.Rocket()
    test_rocket.mass = {"total_mass": 1, "body_mass": 0.5, "prop_mass": 0.5}
    test_rocket.thrust = np.array([1, 1, 1])
    mocker.patch('rocket_math.Rocket.drag_force',
                 return_value=np.array([0.0622, 0.0622, 0.0622]))
    mocker.patch('rocket_math.Rocket.gravity',
                 return_value=32.1389)
    assert np.all(test_rocket.update_acceleration() == np.array(
        [0.9378, 0.9378, -31.2011]))


def test_update_acceleration_with_neg_thrust_and_drag_ints(mocker):
    """
    Test update_acceleration() for thrust and drag force vectors with negative
    integers.
    """
    test_rocket = rm.Rocket()
    test_rocket.mass = {"total_mass": 1, "body_mass": 0.5, "prop_mass": 0.5}
    test_rocket.thrust = np.array([-1, -1, -1])
    mocker.patch('rocket_math.Rocket.drag_force',
                 return_value=np.array([-0.0622, -0.0622, -0.0622]))
    mocker.patch('rocket_math.Rocket.gravity',
                 return_value=32.1389)
    assert np.all(test_rocket.update_acceleration() == np.array(
        [-0.9378, -0.9378, -33.0767]))


def test_update_acceleration_with_thrust_and_drag_floats(mocker):
    """
    Test update_acceleration() for thrust and drag force vectors with floats.
    """
    test_rocket = rm.Rocket()
    test_rocket.world_acceleration = np.array([0, 0, 0])
    test_rocket.mass = {"total_mass": 1, "body_mass": 0.5, "prop_mass": 0.5}
    test_rocket.thrust = np.array([10.5, 11.5, 12.5])
    mocker.patch('rocket_math.Rocket.drag_force',
                 return_value=np.array([7.5339, 8.2514, 8.9689]))
    mocker.patch('rocket_math.Rocket.gravity',
                 return_value=32.1389)
    assert np.all(test_rocket.update_acceleration() -
                  np.array([2.9661, 3.2486, -28.6078])) <= rm.TOLERANCE


# Testing functions for update_velocity().
def test_update_velocity_with_no_vel_or_accel():
    """
    Test update_velocity() for no velocity or acceleration (velocity and
    acceleration vectors are zero vectors).
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([0, 0, 0])
    test_rocket.world_acceleration = np.array([0, 0, 0])
    delta_time = 1
    assert np.all(test_rocket.update_velocity(delta_time) ==
                  np.array([0, 0, 0]))


def test_update_velocity_with_pos_int_vel_and_no_time_change():
    """
    Test update_velocity() for velocity vector with positive integers and no
    time change.
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([1, 1, 1])
    test_rocket.world_acceleration = np.array([1, 1, 1])
    delta_time = 0
    assert np.all(test_rocket.update_velocity(delta_time) ==
                  np.array([1, 1, 1]))


def test_update_velocity_with_pos_vel_and_no_accel_ints():
    """
    Test update_velocity() for velocity vector with positive integers and time
    change but no acceleration.
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([1, 1, 1])
    test_rocket.world_acceleration = np.array([0, 0, 0])
    delta_time = 1
    assert np.all(test_rocket.update_velocity(delta_time) ==
                  np.array([1, 1, 1]))


def test_update_velocity_with_pos_vel_and_accel_ints():
    """
    Test update_velocity() for velocity and acceleration vectors with positive
    integers and time change.
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([1, 1, 1])
    test_rocket.world_acceleration = np.array([1, 1, 1])
    delta_time = 1
    assert np.all(test_rocket.update_velocity(delta_time) ==
                  np.array([2, 2, 2]))


def test_update_velocity_with_neg_vel_and_accel_ints():
    """
    Test update_velocity() for velocity and acceleration vectors with negative
    integers and time change.
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([-1, -1, -1])
    test_rocket.world_acceleration = np.array([-1, -1, -1])
    delta_time = 1
    assert np.all(test_rocket.update_velocity(delta_time) ==
                  np.array([-2, -2, -2]))


def test_update_velocity_with_pos_vel_and_big_neg_accel_ints():
    """
    Test update_velocity() for velocity vector with positive integers and
    acceleration vector with negative integers (acceleration magnitude is
    bigger).
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([1, 1, 1])
    test_rocket.world_acceleration = np.array([-3, -3, -3])
    delta_time = 1
    assert np.all(test_rocket.update_velocity(delta_time) ==
                  np.array([-2, -2, -2]))


def test_update_velocity_with_neg_vel_and_big_pos_accel_ints():
    """
    Test update_velocity() for velocity vector with negative integers and
    acceleration vector with positive integers (acceleration magnitude is
    bigger).
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([-1, -1, -1])
    test_rocket.world_acceleration = np.array([3, 3, 3])
    delta_time = 1
    assert np.all(test_rocket.update_velocity(delta_time) ==
                  np.array([2, 2, 2]))


def test_update_velocity_with_big_pos_vel_and_neg_accel_ints():
    """
    Test update_velocity() for velocity vector with positive integers and
    acceleration vector with negative integers (velocity magnitude is bigger).
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([3, 3, 3])
    test_rocket.world_acceleration = np.array([-1, -1, -1])
    delta_time = 1
    assert np.all(test_rocket.update_velocity(delta_time) ==
                  np.array([2, 2, 2]))


def test_update_velocity_with_big_neg_vel_and_pos_accel_ints():
    """
    Test update_velocity() for velocity vector with negative integers and
    acceleration vector with positive integers (velocity magnitude is bigger).
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([-3, -3, -3])
    test_rocket.world_acceleration = np.array([1, 1, 1])
    delta_time = 1
    assert np.all(test_rocket.update_velocity(delta_time) ==
                  np.array([-2, -2, -2]))


def test_update_velocity_with_vel_and_accel_floats():
    """
    Test update_velocity() for velocity and acceleration vectors with floats.
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([1.5, 2.5, 3.5])
    test_rocket.world_acceleration = np.array([1.5, 2.5, 3.5])
    delta_time = 1
    assert np.all(test_rocket.update_velocity(delta_time) ==
                  np.array([3, 5, 7]))


# Testing functions for update_position().
def test_update_position_no_position_or_vel():
    """
    Test update_position() for no position or velocity (position and
    velocity vectors are zero vectors).
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([0, 0, 0])
    test_rocket.velocity = np.array([0, 0, 0])
    delta_time = 1
    assert np.all(test_rocket.update_position(delta_time) ==
                  np.array([0, 0, 0]))


def test_update_position_with_pos_int_position_and_no_time_change():
    """
    Test update_position() for position and velocity vectors with positive
    integers and no time change.
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([1, 1, 1])
    test_rocket.velocity = np.array([1, 1, 1])
    delta_time = 0
    assert np.all(test_rocket.update_position(delta_time) ==
                  np.array([1, 1, 1]))


def test_update_position_with_pos_int_position_and_no_vel():
    """
    Test update_position() for position vector with positive integers and
    time change but no velocity vector.
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([1, 1, 1])
    test_rocket.velocity = np.array([0, 0, 0])
    delta_time = 1
    assert np.all(test_rocket.update_position(delta_time) ==
                  np.array([1, 1, 1]))


def test_update_position_with_pos_position_and_vel_ints():
    """
    Test update_position() for position and velocity vectors with positive
    integers and time change.
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([1, 1, 1])
    test_rocket.velocity = np.array([1, 1, 1])
    delta_time = 1
    assert np.all(test_rocket.update_position(delta_time) ==
                  np.array([2, 2, 2]))


def test_update_position_with_neg_position_and_vel_ints():
    """
    Test update_position() for position and velocity vectors with negative
    integers and time change.
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([-1, -1, -1])
    test_rocket.velocity = np.array([-1, -1, -1])
    delta_time = 1
    assert np.all(test_rocket.update_position(delta_time) ==
                  np.array([-2, -2, 0]))


def test_update_position_with_pos_position_and_big_neg_vel_ints():
    """
    Test update_position() for position vector with positive integers and
    velocity vector with negative integers (velocity magnitude is bigger).
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([1, 1, 1])
    test_rocket.velocity = np.array([-3, -3, -3])
    delta_time = 1
    assert np.all(test_rocket.update_position(delta_time) ==
                  np.array([-2, -2, 0]))


def test_update_position_with_neg_position_and_big_pos_vel_ints():
    """
    Test update_position() for position vector with negative integers and
    velocity vector with positive integers (velocity magnitude is bigger).
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([-1, -1, -1])
    test_rocket.velocity = np.array([3, 3, 3])
    delta_time = 1
    assert np.all(test_rocket.update_position(delta_time) ==
                  np.array([2, 2, 2]))


def test_update_position_with_big_pos_position_and_neg_vel_ints():
    """
    Test update_position() for position vector with positive integers and
    velocity vector with negative integers (position magnitude is bigger).
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([3, 3, 3])
    test_rocket.velocity = np.array([-1, -1, -1])
    delta_time = 1
    assert np.all(test_rocket.update_position(delta_time) ==
                  np.array([2, 2, 2]))


def test_update_position_with_big_neg_position_and_pos_vel_ints():
    """
    Test update_position() for position vector with negative integers and
    velocity vector with positive integers (position magnitude is bigger).
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([-3, -3, -3])
    test_rocket.velocity = np.array([1, 1, 1])
    delta_time = 1
    assert np.all(test_rocket.update_position(delta_time) ==
                  np.array([-2, -2, 0]))


def test_update_position_with_position_and_vel_floats():
    """
    Test update_position() for position and velocity vectors with positive
    floats.
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([1.5, 2.5, 3.5])
    test_rocket.velocity = np.array([1.5, 2.5, 3.5])
    delta_time = 1
    assert np.all(test_rocket.update_position(delta_time) ==
                  np.array([3, 5, 7]))


def test_update_position_with_neg_z_and_no_vel():
    """
    Test update_position() for position vector with negative z component
    and no velocity vector.
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([1.5, 2.5, -3.5])
    test_rocket.velocity = np.array([0, 0, 0])
    delta_time = 1
    assert np.all(test_rocket.update_position(delta_time) ==
                  np.array([1.5, 2.5, 0]))


def test_update_position_with_only_neg_velocity():
    """
    Test update_position() for only velocity vector with negative floats
    and no position vector.
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([0, 0, 0])
    test_rocket.velocity = np.array([-1.5, -2.5, -3.5])
    delta_time = 1
    assert np.all(test_rocket.update_position(delta_time) ==
                  np.array([-1.5, -2.5, 0]))


def test_update_orientation_eighth_rev_x_axis():
    """
    Test update_orientation() rotating 1/8 of a revolution about the x-axis.
    """
    test_rocket = rm.Rocket()
    delta_time = 0.125  # 1/8 second
    angular_rates = np.array([2 * np.pi, 0, 0])  # 1 rev/s in x
    orientation_after_update = Quaternion(axis=[1, 0, 0],
                                          angle=(np.pi / 4)).elements
    new_orientation = test_rocket.update_orientation(angular_rates, delta_time)
    assert np.all(new_orientation == orientation_after_update)


def test_update_orientation_neg_eighth_rev_x_axis():
    """
    Test update_orientation() rotating -1/8 of a revolution about the x-axis.
    """
    test_rocket = rm.Rocket()
    delta_time = 0.125  # 1/8 second
    angular_rates = np.array([-2 * np.pi, 0, 0])  # 1 rev/s in x
    orientation_after_update = Quaternion(axis=[1, 0, 0],
                                          angle=(-np.pi / 4)).elements
    new_orientation = test_rocket.update_orientation(angular_rates, delta_time)
    assert np.all(new_orientation == orientation_after_update)


def test_update_orientation_quarter_rev_x_axis():
    """
    Test update_orientation() rotating 1/4 of a revolution about the x-axis.
    """
    test_rocket = rm.Rocket()
    delta_time = 0.25  # 1/4 second
    angular_rates = np.array([2 * np.pi, 0, 0])  # 1 rev/s in x
    orientation_after_update = Quaternion(axis=[1, 0, 0],
                                          angle=(np.pi / 2)).elements
    new_orientation = test_rocket.update_orientation(angular_rates, delta_time)
    assert np.all(new_orientation == orientation_after_update)


def test_update_orientation_half_rev_x_axis():
    """
    Test update_orientation() rotating 1/2 of a revolution about the x-axis.
    """
    test_rocket = rm.Rocket()
    delta_time = 0.5  # 1/2 second
    angular_rates = np.array([2 * np.pi, 0, 0])  # 1 rev/s in x
    orientation_after_update = Quaternion(axis=[1, 0, 0],
                                          angle=np.pi).elements
    new_orientation = test_rocket.update_orientation(angular_rates, delta_time)
    assert np.all(new_orientation == orientation_after_update)


def test_update_orientation_three_quarter_rev_x_axis():
    """
    Test update_orientation() rotating 3/4 of a revolution about the x-axis.
    """
    test_rocket = rm.Rocket()
    delta_time = 0.75  # 3/4 second
    angular_rates = np.array([2 * np.pi, 0, 0])  # 1 rev/s in x
    orientation_after_update = Quaternion(axis=[1, 0, 0],
                                          angle=(3 * np.pi / 2)).elements
    new_orientation = test_rocket.update_orientation(angular_rates, delta_time)
    assert np.all(new_orientation == orientation_after_update)


def test_update_orientation_eighth_rev_y_axis():
    """
    Test update_orientation() rotating 1/8 of a revolution about the y-axis.
    """
    test_rocket = rm.Rocket()
    delta_time = 0.125  # 1/8 second
    angular_rates = np.array([0, 2 * np.pi, 0])  # 1 rev/s in y
    orientation_after_update = Quaternion(axis=[0, 1, 0],
                                          angle=(np.pi / 4)).elements
    new_orientation = test_rocket.update_orientation(angular_rates, delta_time)
    assert np.all(new_orientation == orientation_after_update)


def test_update_orientation_neg_eighth_rev_y_axis():
    """
    Test update_orientation() rotating -1/8 of a revolution about the y-axis.
    """
    test_rocket = rm.Rocket()
    delta_time = 0.125  # 1/8 second
    angular_rates = np.array([0, -2 * np.pi, 0])  # 1 rev/s in y
    orientation_after_update = Quaternion(axis=[0, 1, 0],
                                          angle=(-np.pi / 4)).elements
    new_orientation = test_rocket.update_orientation(angular_rates, delta_time)
    assert np.all(new_orientation == orientation_after_update)


def test_update_orientation_quarter_rev_y_axis():
    """
    Test update_orientation() rotating 1/4 of a revolution about the y-axis.
    """
    test_rocket = rm.Rocket()
    delta_time = 0.25  # 1/4 second
    angular_rates = np.array([0, 2 * np.pi, 0])  # 1 rev/s in y
    orientation_after_update = Quaternion(axis=[0, 1, 0],
                                          angle=(np.pi / 2)).elements
    new_orientation = test_rocket.update_orientation(angular_rates, delta_time)
    assert np.all(new_orientation == orientation_after_update)


def test_update_orientation_half_rev_y_axis():
    """
    Test update_orientation() rotating 1/2 of a revolution about the y-axis.
    """
    test_rocket = rm.Rocket()
    delta_time = 0.5  # 1/2 second
    angular_rates = np.array([0, 2 * np.pi, 0])  # 1 rev/s in y
    orientation_after_update = Quaternion(axis=[0, 1, 0],
                                          angle=np.pi).elements
    new_orientation = test_rocket.update_orientation(angular_rates, delta_time)
    assert np.all(new_orientation == orientation_after_update)


def test_update_orientation_three_quarter_rev_y_axis():
    """
    Test update_orientation() rotating 3/4 of a revolution about the y-axis.
    """
    test_rocket = rm.Rocket()
    delta_time = 0.75  # 3/4 second
    angular_rates = np.array([0, 2 * np.pi, 0])  # 1 rev/s in y
    orientation_after_update = Quaternion(axis=[0, 1, 0],
                                          angle=(3 * np.pi / 2)).elements
    new_orientation = test_rocket.update_orientation(angular_rates, delta_time)
    assert np.all(new_orientation == orientation_after_update)


def test_update_orientation_eighth_rev_z_axis():
    """
    Test update_orientation() rotating 1/8 of a revolution about the z-axis.
    """
    test_rocket = rm.Rocket()
    delta_time = 0.125  # 1/8 second
    angular_rates = np.array([0, 0, 2 * np.pi])  # 1 rev/s in z
    orientation_after_update = Quaternion(axis=[0, 0, 1],
                                          angle=(np.pi / 4)).elements
    new_orientation = test_rocket.update_orientation(angular_rates, delta_time)
    assert np.all(new_orientation == orientation_after_update)


def test_update_orientation_neg_eighth_rev_z_axis():
    """
    Test update_orientation() rotating -1/8 of a revolution about the z-axis.
    """
    test_rocket = rm.Rocket()
    delta_time = 0.125  # 1/8 second
    angular_rates = np.array([0, 0, -2 * np.pi])  # 1 rev/s in z
    orientation_after_update = Quaternion(axis=[0, 0, 1],
                                          angle=(-np.pi / 4)).elements
    new_orientation = test_rocket.update_orientation(angular_rates, delta_time)
    assert np.all(new_orientation == orientation_after_update)


def test_update_orientation_quarter_rev_z_axis():
    """
    Test update_orientation() rotating 1/4 of a revolution about the z-axis.
    """
    test_rocket = rm.Rocket()
    delta_time = 0.25  # 1/4 second
    angular_rates = np.array([0, 0, 2 * np.pi])  # 1 rev/s in z
    orientation_after_update = Quaternion(axis=[0, 0, 1],
                                          angle=(np.pi / 2)).elements
    new_orientation = test_rocket.update_orientation(angular_rates, delta_time)
    assert np.all(new_orientation == orientation_after_update)


def test_update_orientation_half_rev_z_axis():
    """
    Test update_orientation() rotating 1/2 of a revolution about the z-axis.
    """
    test_rocket = rm.Rocket()
    delta_time = 0.5  # 1/2 second
    angular_rates = np.array([0, 0, 2 * np.pi])  # 1 rev/s in z
    orientation_after_update = Quaternion(axis=[0, 0, 1],
                                          angle=np.pi).elements
    new_orientation = test_rocket.update_orientation(angular_rates, delta_time)
    assert np.all(new_orientation == orientation_after_update)


def test_update_orientation_three_quarter_rev_z_axis():
    """
    Test update_orientation() rotating 3/4 of a revolution about the z-axis.
    """
    test_rocket = rm.Rocket()
    delta_time = 0.75  # 3/4 second
    angular_rates = np.array([0, 0, 2 * np.pi])  # 1 rev/s in z
    orientation_after_update = Quaternion(axis=[0, 0, 1],
                                          angle=(3 * np.pi / 2)).elements
    new_orientation = test_rocket.update_orientation(angular_rates, delta_time)
    assert np.all(new_orientation == orientation_after_update)


def test_update_orientation_half_rev_all_axes():
    """
    Test update_orientation() rotating 1/2 of a revolution about all of the
    axes.
    """
    test_rocket = rm.Rocket()
    delta_time = 0.5  # 1/2 second
    angular_rates = np.array([np.pi / np.sqrt(3), np.pi / np.sqrt(3),
                              np.pi / np.sqrt(3)])  # 1/6 rev/s for all
    orientation_after_update = Quaternion(axis=[1, 1, 1],
                                          angle=(np.pi / 2)).elements
    new_orientation = test_rocket.update_orientation(angular_rates, delta_time)
    assert np.all(new_orientation - orientation_after_update <= rm.TOLERANCE)


'''
Note for the next 3 unit tests:
    Angular rates were determined through an online calculator and by using the
    predetermined desired quaternion with the formula:
    
        angular_rates = (quaternion_axis * quaternion_angle) / timestep
        
    Since the initial quaternion is always [1, 0, 0, 0] in this context.
    
    Online calculator used:
    -https://www.andre-gaschler.com/rotationconverter/
'''


def test_update_orientation_quarter_rev_x_y_axes():
    """
    Test update_orientation() rotating 1/4 of a revolution about the x and y
    axes.
    """
    test_rocket = rm.Rocket()
    delta_time = 0.25  # 1/4 second
    angular_rates = np.array([2.80992617, 5.61985144, 0])
    orientation_after_update = Quaternion(axis=[1, 2, 0],
                                          angle=(np.pi / 2)).elements
    new_orientation = test_rocket.update_orientation(angular_rates, delta_time)
    assert np.all(new_orientation - orientation_after_update <= rm.TOLERANCE)


def test_update_orientation_quarter_rev_y_z_axes():
    """
    Test update_orientation() rotating 1/4 of a revolution about the y and z
    axes.
    """
    test_rocket = rm.Rocket()
    delta_time = 0.25  # 1/4 second
    angular_rates = np.array([0, 2.80992617, 5.61985144])
    orientation_after_update = Quaternion(axis=[0, 1, 2],
                                          angle=(np.pi / 2)).elements
    new_orientation = test_rocket.update_orientation(angular_rates, delta_time)
    assert np.all(new_orientation - orientation_after_update <= rm.TOLERANCE)


def test_update_orientation_quarter_rev_x_z_axes():
    """
    Test update_orientation() rotating 1/4 of a revolution about the x and z
    axes.
    """
    test_rocket = rm.Rocket()
    delta_time = 0.25  # 1/4 second
    angular_rates = np.array([5.61985144, 0, 2.80992617])
    orientation_after_update = Quaternion(axis=[2, 0, 1],
                                          angle=(np.pi / 2)).elements
    new_orientation = test_rocket.update_orientation(angular_rates, delta_time)
    assert np.all(new_orientation - orientation_after_update <= rm.TOLERANCE)


def test_update_orientation_from_non_identity_quat():
    """
    Test update_orientation() rotating from non-identity quaternion.

    Note: rotation is 1/4 revolution about x axis.
    """
    test_rocket = rm.Rocket()
    orientation = Quaternion(axis=[-1, 0, 0], angle=(np.pi / 2))
    test_rocket.orientation = orientation.elements
    delta_time = 0.25   # 1/4 second
    angular_rates = np.array([2 * np.pi, 0, 0])
    orientation_after_update = Quaternion().elements
    new_orientation = test_rocket.update_orientation(angular_rates, delta_time)
    assert np.all(new_orientation == orientation_after_update)


# Testing functions for finding temperature
def test_temperature_at_ground():
    test_rocket = rm.Rocket()
    test_rocket.altitude = 0
    assert(test_rocket.update_temperature() - 15.04 <= rm.TOLERANCE)


def test_temperature_at_100m():
    test_rocket = rm.Rocket()
    test_rocket.altitude = 100
    assert(test_rocket.update_temperature() - 14.391 <= rm.TOLERANCE)


def test_temperature_at_1000m():
    test_rocket = rm.Rocket()
    test_rocket.altitude = 1000
    assert(test_rocket.update_temperature() - 8.55 <= rm.TOLERANCE)


def test_temperature_at_10000m():
    test_rocket = rm.Rocket()
    test_rocket.altitude = 10000
    assert(test_rocket.update_temperature() - -49.86 <= rm.TOLERANCE)


# Testing functions for finding barometric pressure
def test_baro_pressure_at_ground():
    test_rocket = rm.Rocket()
    test_rocket.temperature = 15.04
    test_rocket.altitude = 0
    assert(abs(test_rocket.update_baro_pressure() - 101.4009) <= rm.TOLERANCE)


def test_baro_pressure_at_100m():
    test_rocket = rm.Rocket()
    test_rocket.temperature = 14.391
    test_rocket.altitude = 100
    assert(abs(test_rocket.update_baro_pressure() - 100.2062) <= rm.TOLERANCE)


def test_baro_pressure_at_1000m():
    test_rocket = rm.Rocket()
    test_rocket.temperature = 8.55
    test_rocket.altitude = 1000
    assert(abs(test_rocket.update_baro_pressure() - 89.9581) <= rm.TOLERANCE)


def test_baro_pressure_at_10000m():
    test_rocket = rm.Rocket()
    test_rocket.temperature = -49.86
    test_rocket.altitude = 10000
    assert(abs(test_rocket.update_baro_pressure() - 26.5162) <= rm.TOLERANCE)


def test_baro_pressure_at_20000m():
    test_rocket = rm.Rocket()
    test_rocket.temperature = -56.46
    test_rocket.altitude = 20000
    assert(abs(test_rocket.update_baro_pressure() - 5.5298) <= rm.TOLERANCE)


# Testing functions for finding body/proper acceleration
def test_body_acceleration_zeros():
    """
    Test update_body_acceleration() using a zero vector
    """
    test_rocket = rm.Rocket()
    test_rocket.orientation = Quaternion(axis=[1, 1, 1], angle=np.pi/2).elements
    test_rocket.world_acceleration = np.array([0, 0, 0])
    body_acceleration_after_rotate = np.array([0, 0, 0])
    new_body_acceleration = test_rocket.update_body_acceleration()
    assert np.all(body_acceleration_after_rotate == new_body_acceleration)


def test_body_acceleration_positive_integers():
    """
    Test update_body_acceleration() using a vector of positive integers
    """
    test_rocket = rm.Rocket()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1], angle=np.pi/2).elements
    test_rocket.world_acceleration = np.array([1, 1, 1])
    body_acceleration_after_rotate = np.array([0.2929, 0, 1.707])
    new_body_acceleration = test_rocket.update_body_acceleration()
    assert np.all(abs(body_acceleration_after_rotate - new_body_acceleration) <= rm.TOLERANCE)


def test_body_acceleration_negative_integers():
    """
    Test update_body_acceleration() using a vector of negative integers
    """
    test_rocket = rm.Rocket()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1], angle=np.pi/2).elements
    test_rocket.world_acceleration = np.array([-1,-1,-3])
    body_acceleration_after_rotate = np.array([-1.2929, 1.4142, -2.7071])
    new_body_acceleration = test_rocket.update_body_acceleration()
    assert np.all(abs(body_acceleration_after_rotate - new_body_acceleration) <= rm.TOLERANCE)


def test_body_acceleration_positive_floats():
    """
    Test update_body_acceleration() using a vector of positive floats
    """
    test_rocket = rm.Rocket()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1], angle=np.pi/2).elements
    test_rocket.world_acceleration = np.array([1.5, 2.5, 3.5])
    body_acceleration_after_rotate = np.array([0.7322, -1.4142, 4.2678])
    new_body_acceleration = test_rocket.update_body_acceleration()
    assert np.all(abs(body_acceleration_after_rotate - new_body_acceleration) <= rm.TOLERANCE)


def test_body_acceleration_negative_floats():
    """
    Test update_body_acceleration() using a vector of negative floats
    """
    test_rocket = rm.Rocket()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1], angle=np.pi/2).elements
    test_rocket.world_acceleration = np.array([-1.5, -2, -3])
    body_acceleration_after_rotate = np.array([-0.8358, 1.0606, -3.6642])
    new_body_acceleration = test_rocket.update_body_acceleration()
    assert np.all(abs(body_acceleration_after_rotate - new_body_acceleration) <= rm.TOLERANCE)


def test_body_acceleration_45_degrees():
    """
    Test update_body_acceleration() at an angle of 45 degrees
    """
    test_rocket = rm.Rocket()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1], angle=np.pi/4).elements
    test_rocket.world_acceleration = np.array([1, 1, 1])
    body_acceleration_after_rotate = np.array([0.5, 0.7071, 1.5])
    new_body_acceleration = test_rocket.update_body_acceleration()
    assert np.all(abs(body_acceleration_after_rotate - new_body_acceleration) <= rm.TOLERANCE)


def test_body_acceleration_180_degrees():
    """
    Test update_body_acceleration() at an angle of 180 degrees
    """
    test_rocket = rm.Rocket()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1], angle=np.pi).elements
    test_rocket.world_acceleration = np.array([1, 1, 1])
    body_acceleration_after_rotate = np.array([1, -1, 1])
    new_body_acceleration = test_rocket.update_body_acceleration()
    assert np.all(abs(body_acceleration_after_rotate - new_body_acceleration) <= rm.TOLERANCE)


# Testing functions for finding the magnetic field around the rocket
def test_magnetic_field_zeros():
    """
    Test update_magnetic_field() using a zero vector
    """
    test_rocket = rm.Rocket()
    test_rocket.orientation = Quaternion(axis=[1, 1, 1], angle=np.pi/2).elements
    test_rocket.world_mag_field = np.array([0, 0, 0])
    mag_field_after_rotate = np.array([0, 0, 0])
    new_mag_field = test_rocket.update_magnetic_field()
    assert np.all(abs(mag_field_after_rotate - new_mag_field) <= rm.TOLERANCE)


def test_magnetic_field_positive_integers():
    """
    Test update_magnetic_field() using a vector of positive integers
    """
    test_rocket = rm.Rocket()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1], angle=np.pi/2).elements
    test_rocket.world_mag_field = np.array([1, 1, 1])
    mag_field_after_rotate = np.array([0.2929, 0, 1.707])
    new_mag_field = test_rocket.update_magnetic_field()
    assert np.all(abs(mag_field_after_rotate - new_mag_field) <= rm.TOLERANCE)


def test_magnetic_field_negative_integers():
    test_rocket = rm.Rocket()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1], angle=np.pi/2).elements
    test_rocket.world_mag_field = np.array([-1,-1,-3])
    mag_field_after_rotate = np.array([-1.2929, 1.4142, -2.7071])
    new_mag_field = test_rocket.update_magnetic_field()
    assert np.all(abs(mag_field_after_rotate - new_mag_field) <= rm.TOLERANCE)


def test_magnetic_field_positive_floats():
    """
    Test update_magnetic_field() using a vector of positive floats
    """
    test_rocket = rm.Rocket()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1], angle=np.pi/2).elements
    test_rocket.world_mag_field = np.array([1.5, 2.5, 3.5])
    mag_field_after_rotate = np.array([0.7322, -1.4142, 4.2678])
    new_mag_field = test_rocket.update_magnetic_field()
    assert np.all(abs(mag_field_after_rotate - new_mag_field) <= rm.TOLERANCE)


def test_magnetic_field_negative_floats():
    """
    Test update_magnetic_field() using a vector of negative floats
    """
    test_rocket = rm.Rocket()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1], angle=np.pi/2).elements
    test_rocket.world_mag_field = np.array([-1.5, -2, -3])
    mag_field_after_rotate = np.array([-0.8358, 1.0606, -3.6642])
    new_mag_field = test_rocket.update_magnetic_field()
    assert np.all(abs(mag_field_after_rotate - new_mag_field) <= rm.TOLERANCE)


def test_magnetic_field_45_degrees():
    """
    Test update_magnetic_field() at an angle of 45 degrees
    """
    test_rocket = rm.Rocket()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1], angle=np.pi/4).elements
    test_rocket.world_mag_field = np.array([1, 1, 1])
    mag_field_after_rotate = np.array([0.5, 0.7071, 1.5])
    new_mag_field = test_rocket.update_magnetic_field()
    assert np.all(abs(mag_field_after_rotate - new_mag_field) <= rm.TOLERANCE)


def test_magnetic_field_180_degrees():
    """
    Test update_magnetic_field() at an angle of 180 degrees
    """
    test_rocket = rm.Rocket()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1], angle=np.pi).elements
    test_rocket.world_mag_field = np.array([1, 1, 1])
    mag_field_after_rotate = np.array([1, -1, 1])
    new_mag_field = test_rocket.update_magnetic_field()
    assert np.all(abs(mag_field_after_rotate - new_mag_field) <= rm.TOLERANCE)
