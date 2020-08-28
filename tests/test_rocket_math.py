# Test suite for rocket_math.py

import numpy as np

import rocket_math as rm

'''
Notes:
- For all calculations, there will be an arbitrary 4 decimal places max.
- No rocket_air_density unit test since the current method for
determining/testing air density needs to be changed.
'''


# Testing functions for gravity().
def test_gravity_at_ground():
    test_rocket = rm.Rocket()
    assert np.around(test_rocket.gravity(), 4) == 32.1389


def test_gravity_at_100_ft():
    test_rocket = rm.Rocket()
    test_rocket.altitude = 100
    assert np.around(test_rocket.gravity(), 4) == 32.1386


def test_gravity_at_1000_ft():
    test_rocket = rm.Rocket()
    test_rocket.altitude = 1000
    assert np.around(test_rocket.gravity(), 4) == 32.1358


def test_gravity_at_10000_ft():
    test_rocket = rm.Rocket()
    test_rocket.altitude = 10000
    assert np.around(test_rocket.gravity(), 4) == 32.1082


def test_gravity_at_100000_ft():
    test_rocket = rm.Rocket()
    test_rocket.altitude = 100000
    assert np.around(test_rocket.gravity(), 4) == 31.8336


def test_gravity_at_float_num_ft():
    test_rocket = rm.Rocket()
    test_rocket.altitude = 5432.10
    assert np.around(test_rocket.gravity(), 4) == 32.1222


# Testing functions for update_mass().
def test_update_mass_above_mass_loss_threshold():
    """
    Test update_mass() for mass above the mass loss threshold from one
    timestep.
    """
    test_rocket = rm.Rocket()
    test_rocket.mass = {"total_mass": 110, "body_mass": 55, "comb_mass": 55}
    assert test_rocket.update_mass() == {"total_mass": 110 - rm.MASS_LOSS,
                                         "body_mass": 55,
                                         "comb_mass": 55 - rm.MASS_LOSS}
    # Test again just to be sure it decreases properly.
    assert test_rocket.update_mass() == {
        "total_mass": 110 - (2 * rm.MASS_LOSS),
        "body_mass": 55,
        "comb_mass": 55 - (2 * rm.MASS_LOSS)}


def test_update_mass_below_mass_loss_threshold():
    """
    Test update_mass() for mass below the mass threshold loss from one
    timestep.
    """
    test_rocket = rm.Rocket()
    test_rocket.mass = {"total_mass": 100 + rm.MASS_LOSS - 0.01,
                        "body_mass": 100, "comb_mass": rm.MASS_LOSS - 0.01}
    assert test_rocket.update_mass() == {"total_mass": 100, "body_mass": 100,
                                         "comb_mass": 0}
    # Test again to be sure it doesn't decrease further.
    assert test_rocket.update_mass() == {"total_mass": 100, "body_mass": 100,
                                         "comb_mass": 0}


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
    assert np.all(np.around(test_rocket.drag_force(), rm.DECIMALS) == np.array(
        [0.0367, 0.0367, 0.0367]))  # hand calc


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
    assert np.all(np.around(test_rocket.drag_force(), rm.DECIMALS) == np.array(
        [-0.0367, -0.0367, -0.0367]))  # hand calc


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
    assert np.all(np.around(test_rocket.drag_force(), rm.DECIMALS) == np.array(
        [30.5226, 39.9142, 49.3058]))  # hand calc


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
    # TODO: investigate why thrust = [1, 1, 1] when this isn't set
    test_rocket.thrust = np.array([0, 0, 0])
    test_rocket.mass = {"total_mass": 1, "body_mass": 0.5, "comb_mass": 0.5}
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
    test_rocket.mass = {"total_mass": 1, "body_mass": 0.5, "comb_mass": 0.5}
    test_rocket.thrust = np.array(
        [0, 0, test_rocket.mass["total_mass"] * 32.1389])
    test_rocket.acceleration = np.array([0, 0, 0])
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
    test_rocket.mass = {"total_mass": 1, "body_mass": 0.5, "comb_mass": 0.5}
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
    test_rocket.mass = {"total_mass": 1, "body_mass": 0.5, "comb_mass": 0.5}
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
    test_rocket.acceleration = np.array([0, 0, 0])
    test_rocket.mass = {"total_mass": 1, "body_mass": 0.5, "comb_mass": 0.5}
    test_rocket.thrust = np.array([10.5, 11.5, 12.5])
    mocker.patch('rocket_math.Rocket.drag_force',
                 return_value=np.array([7.5339, 8.2514, 8.9689]))
    mocker.patch('rocket_math.Rocket.gravity',
                 return_value=32.1389)
    assert np.all(test_rocket.update_acceleration() == np.array(
        [2.9661, 3.2486, -28.6078]))


# Testing functions for update_velocity().
def test_update_velocity_with_no_vel_or_accel():
    """
    Test update_velocity() for no velocity or acceleration (velocity and
    acceleration vectors are zero vectors).
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([0, 0, 0])
    test_rocket.acceleration = np.array([0, 0, 0])
    current_time, previous_time = 1, 0
    assert np.all(
        test_rocket.update_velocity(current_time, previous_time) == np.array(
            [0, 0, 0]))


def test_update_velocity_with_pos_int_vel_and_no_time_change():
    """
    Test update_velocity() for velocity vector with positive integers and no
    time change.
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([1, 1, 1])
    test_rocket.acceleration = np.array([1, 1, 1])
    current_time, previous_time = 0, 0
    assert np.all(
        test_rocket.update_velocity(current_time, previous_time) == np.array(
            [1, 1, 1]))


def test_update_velocity_with_pos_vel_and_no_accel_ints():
    """
    Test update_velocity() for velocity vector with positive integers and time
    change but no acceleration.
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([1, 1, 1])
    test_rocket.acceleration = np.array([0, 0, 0])
    current_time, previous_time = 1, 0
    assert np.all(
        test_rocket.update_velocity(current_time, previous_time) == np.array(
            [1, 1, 1]))


def test_update_velocity_with_pos_vel_and_accel_ints():
    """
    Test update_velocity() for velocity and acceleration vectors with positive
    integers and time change.
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([1, 1, 1])
    test_rocket.acceleration = np.array([1, 1, 1])
    current_time, previous_time = 1, 0
    assert np.all(
        test_rocket.update_velocity(current_time, previous_time) == np.array(
            [2, 2, 2]))


def test_update_velocity_with_neg_vel_and_accel_ints():
    """
    Test update_velocity() for velocity and acceleration vectors with negative
    integers and time change.
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([-1, -1, -1])
    test_rocket.acceleration = np.array([-1, -1, -1])
    current_time, previous_time = 1, 0
    assert np.all(
        test_rocket.update_velocity(current_time, previous_time) == np.array(
            [-2, -2, -2]))


def test_update_velocity_with_pos_vel_and_big_neg_accel_ints():
    """
    Test update_velocity() for velocity vector with positive integers and
    acceleration vector with negative integers (acceleration magnitude is
    bigger).
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([1, 1, 1])
    test_rocket.acceleration = np.array([-3, -3, -3])
    current_time, previous_time = 1, 0
    assert np.all(
        test_rocket.update_velocity(current_time, previous_time) == np.array(
            [-2, -2, -2]))


def test_update_velocity_with_neg_vel_and_big_pos_accel_ints():
    """
    Test update_velocity() for velocity vector with negative integers and
    acceleration vector with positive integers (acceleration magnitude is
    bigger).
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([-1, -1, -1])
    test_rocket.acceleration = np.array([3, 3, 3])
    current_time, previous_time = 1, 0
    assert np.all(
        test_rocket.update_velocity(current_time, previous_time) == np.array(
            [2, 2, 2]))


def test_update_velocity_with_big_pos_vel_and_neg_accel_ints():
    """
    Test update_velocity() for velocity vector with positive integers and
    acceleration vector with negative integers (velocity magnitude is bigger).
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([3, 3, 3])
    test_rocket.acceleration = np.array([-1, -1, -1])
    current_time, previous_time = 1, 0
    assert np.all(
        test_rocket.update_velocity(current_time, previous_time) == np.array(
            [2, 2, 2]))


def test_update_velocity_with_big_neg_vel_and_pos_accel_ints():
    """
    Test update_velocity() for velocity vector with negative integers and
    acceleration vector with positive integers (velocity magnitude is bigger).
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([-3, -3, -3])
    test_rocket.acceleration = np.array([1, 1, 1])
    current_time, previous_time = 1, 0
    assert np.all(
        test_rocket.update_velocity(current_time, previous_time) == np.array(
            [-2, -2, -2]))


def test_update_velocity_with_vel_and_accel_floats():
    """
    Test update_velocity() for velocity and acceleration vectors with floats.
    """
    test_rocket = rm.Rocket()
    test_rocket.velocity = np.array([1.5, 2.5, 3.5])
    test_rocket.acceleration = np.array([1.5, 2.5, 3.5])
    current_time, previous_time = 1, 0
    assert np.all(
        test_rocket.update_velocity(current_time, previous_time) == np.array(
            [3, 5, 7]))


# Testing functions for update_position().
def test_update_position_no_position_or_vel():
    """
    Test update_position() for no position or velocity (position and
    velocity vectors are zero vectors).
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([0, 0, 0])
    test_rocket.velocity = np.array([0, 0, 0])
    current_time, previous_time = 1, 0
    assert np.all(
        test_rocket.update_position(current_time, previous_time) == np.array(
            [0, 0, 0]))


def test_update_position_with_pos_int_position_and_no_time_change():
    """
    Test update_position() for position and velocity vectors with positive
    integers and no time change.
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([1, 1, 1])
    test_rocket.velocity = np.array([1, 1, 1])
    current_time, previous_time = 0, 0
    assert np.all(
        test_rocket.update_position(current_time, previous_time) == np.array(
            [1, 1, 1]))


def test_update_position_with_pos_int_position_and_no_vel():
    """
    Test update_position() for position vector with positive integers and
    time change but no velocity vector.
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([1, 1, 1])
    test_rocket.velocity = np.array([0, 0, 0])
    current_time, previous_time = 1, 0
    assert np.all(
        test_rocket.update_position(current_time, previous_time) == np.array(
            [1, 1, 1]))


def test_update_position_with_pos_position_and_vel_ints():
    """
    Test update_position() for position and velocity vectors with positive
    integers and time change.
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([1, 1, 1])
    test_rocket.velocity = np.array([1, 1, 1])
    current_time, previous_time = 1, 0
    assert np.all(
        test_rocket.update_position(current_time, previous_time) == np.array(
            [2, 2, 2]))


def test_update_position_with_neg_position_and_vel_ints():
    """
    Test update_position() for position and velocity vectors with negative
    integers and time change.
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([-1, -1, -1])
    test_rocket.velocity = np.array([-1, -1, -1])
    current_time, previous_time = 1, 0
    assert np.all(
        test_rocket.update_position(current_time, previous_time) == np.array(
            [-2, -2, 0]))


def test_update_position_with_pos_position_and_big_neg_vel_ints():
    """
    Test update_position() for position vector with positive integers and
    velocity vector with negative integers (velocity magnitude is bigger).
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([1, 1, 1])
    test_rocket.velocity = np.array([-3, -3, -3])
    current_time, previous_time = 1, 0
    assert np.all(
        test_rocket.update_position(current_time, previous_time) == np.array(
            [-2, -2, 0]))


def test_update_position_with_neg_position_and_big_pos_vel_ints():
    """
    Test update_position() for position vector with negative integers and
    velocity vector with positive integers (velocity magnitude is bigger).
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([-1, -1, -1])
    test_rocket.velocity = np.array([3, 3, 3])
    current_time, previous_time = 1, 0
    assert np.all(
        test_rocket.update_position(current_time, previous_time) == np.array(
            [2, 2, 2]))


def test_update_position_with_big_pos_position_and_neg_vel_ints():
    """
    Test update_position() for position vector with positive integers and
    velocity vector with negative integers (position magnitude is bigger).
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([3, 3, 3])
    test_rocket.velocity = np.array([-1, -1, -1])
    current_time, previous_time = 1, 0
    assert np.all(
        test_rocket.update_position(current_time, previous_time) == np.array(
            [2, 2, 2]))


def test_update_position_with_big_neg_position_and_pos_vel_ints():
    """
    Test update_position() for position vector with negative integers and
    velocity vector with positive integers (position magnitude is bigger).
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([-3, -3, -3])
    test_rocket.velocity = np.array([1, 1, 1])
    current_time, previous_time = 1, 0
    assert np.all(
        test_rocket.update_position(current_time, previous_time) == np.array(
            [-2, -2, 0]))


def test_update_position_with_position_and_vel_floats():
    """
    Test update_position() for position and velocity vectors with positive
    floats.
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([1.5, 2.5, 3.5])
    test_rocket.velocity = np.array([1.5, 2.5, 3.5])
    current_time, previous_time = 1, 0
    assert np.all(
        test_rocket.update_position(current_time, previous_time) == np.array(
            [3, 5, 7]))


def test_update_position_with_neg_z_and_no_vel():
    """
    Test update_position() for position vector with negative z component
    and no velocity vector.
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([1.5, 2.5, -3.5])
    test_rocket.velocity = np.array([0, 0, 0])
    current_time, previous_time = 1, 0
    assert np.all(
        test_rocket.update_position(current_time, previous_time) == np.array(
            [1.5, 2.5, 0]))


def test_update_position_with_only_neg_velocity():
    """
    Test update_position() for only velocity vector with negative floats
    and no position vector.
    """
    test_rocket = rm.Rocket()
    test_rocket.position = np.array([0, 0, 0])
    test_rocket.velocity = np.array([-1.5, -2.5, -3.5])
    current_time, previous_time = 1, 0
    assert np.all(
        test_rocket.update_position(current_time, previous_time) == np.array(
            [-1.5, -2.5, 0]))
