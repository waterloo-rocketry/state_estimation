import sensors
import rocket_math as rm

import numpy as np
from pyquaternion import Quaternion

def test_update_orientation_eighth_rev_x_axis():
    """
    Test update_orientation() rotating 1/8 of a revolution about the x-axis.
    """
    test_rocket = rm.Rocket()
    gyro = sensors.Gyro()
    delta_time = 0.125  # 1/8 second
    angular_rates = np.array([2 * np.pi, 0, 0])  # 1 rev/s in x
    orientation_after_update = Quaternion(axis=[1, 0, 0],
                                          angle=(np.pi / 4)).elements
    new_orientation = gyro.update(test_rocket, angular_rates, delta_time)
    np.testing.assert_allclose(new_orientation, orientation_after_update, atol=rm.TOLERANCE)


def test_update_orientation_neg_eighth_rev_x_axis():
    """
    Test update_orientation() rotating -1/8 of a revolution about the x-axis.
    """
    test_rocket = rm.Rocket()
    gyro = sensors.Gyro()
    delta_time = 0.125  # 1/8 second
    angular_rates = np.array([-2 * np.pi, 0, 0])  # 1 rev/s in x
    orientation_after_update = Quaternion(axis=[1, 0, 0],
                                          angle=(-np.pi / 4)).elements
    new_orientation = gyro.update(test_rocket, angular_rates, delta_time)
    np.testing.assert_allclose(new_orientation, orientation_after_update, atol=rm.TOLERANCE)


def test_update_orientation_quarter_rev_x_axis():
    """
    Test update_orientation() rotating 1/4 of a revolution about the x-axis.
    """
    test_rocket = rm.Rocket()
    gyro = sensors.Gyro()
    delta_time = 0.25  # 1/4 second
    angular_rates = np.array([2 * np.pi, 0, 0])  # 1 rev/s in x
    orientation_after_update = Quaternion(axis=[1, 0, 0],
                                          angle=(np.pi / 2)).elements
    new_orientation = gyro.update(test_rocket, angular_rates, delta_time)
    np.testing.assert_allclose(new_orientation, orientation_after_update, atol=rm.TOLERANCE)


def test_update_orientation_half_rev_x_axis():
    """
    Test update_orientation() rotating 1/2 of a revolution about the x-axis.
    """
    test_rocket = rm.Rocket()
    gyro = sensors.Gyro()
    delta_time = 0.5  # 1/2 second
    angular_rates = np.array([2 * np.pi, 0, 0])  # 1 rev/s in x
    orientation_after_update = Quaternion(axis=[1, 0, 0],
                                          angle=np.pi).elements
    new_orientation = gyro.update(test_rocket, angular_rates, delta_time)
    np.testing.assert_allclose(new_orientation, orientation_after_update, atol=rm.TOLERANCE)


def test_update_orientation_three_quarter_rev_x_axis():
    """
    Test update_orientation() rotating 3/4 of a revolution about the x-axis.
    """
    test_rocket = rm.Rocket()
    gyro = sensors.Gyro()
    delta_time = 0.75  # 3/4 second
    angular_rates = np.array([2 * np.pi, 0, 0])  # 1 rev/s in x
    orientation_after_update = Quaternion(axis=[1, 0, 0],
                                          angle=(3 * np.pi / 2)).elements
    new_orientation = gyro.update(test_rocket, angular_rates, delta_time)
    np.testing.assert_allclose(new_orientation, orientation_after_update, atol=rm.TOLERANCE)


def test_update_orientation_eighth_rev_y_axis():
    """
    Test update_orientation() rotating 1/8 of a revolution about the y-axis.
    """
    test_rocket = rm.Rocket()
    gyro = sensors.Gyro()
    delta_time = 0.125  # 1/8 second
    angular_rates = np.array([0, 2 * np.pi, 0])  # 1 rev/s in y
    orientation_after_update = Quaternion(axis=[0, 1, 0],
                                          angle=(np.pi / 4)).elements
    new_orientation = gyro.update(test_rocket, angular_rates, delta_time)
    np.testing.assert_allclose(new_orientation, orientation_after_update, atol=rm.TOLERANCE)


def test_update_orientation_neg_eighth_rev_y_axis():
    """
    Test update_orientation() rotating -1/8 of a revolution about the y-axis.
    """
    test_rocket = rm.Rocket()
    gyro = sensors.Gyro()
    delta_time = 0.125  # 1/8 second
    angular_rates = np.array([0, -2 * np.pi, 0])  # 1 rev/s in y
    orientation_after_update = Quaternion(axis=[0, 1, 0],
                                          angle=(-np.pi / 4)).elements
    new_orientation = gyro.update(test_rocket, angular_rates, delta_time)
    np.testing.assert_allclose(new_orientation, orientation_after_update, atol=rm.TOLERANCE)


def test_update_orientation_quarter_rev_y_axis():
    """
    Test update_orientation() rotating 1/4 of a revolution about the y-axis.
    """
    test_rocket = rm.Rocket()
    gyro = sensors.Gyro()
    delta_time = 0.25  # 1/4 second
    angular_rates = np.array([0, 2 * np.pi, 0])  # 1 rev/s in y
    orientation_after_update = Quaternion(axis=[0, 1, 0],
                                          angle=(np.pi / 2)).elements
    new_orientation = gyro.update(test_rocket, angular_rates, delta_time)
    np.testing.assert_allclose(new_orientation, orientation_after_update, atol=rm.TOLERANCE)


def test_update_orientation_half_rev_y_axis():
    """
    Test update_orientation() rotating 1/2 of a revolution about the y-axis.
    """
    test_rocket = rm.Rocket()
    gyro = sensors.Gyro()
    delta_time = 0.5  # 1/2 second
    angular_rates = np.array([0, 2 * np.pi, 0])  # 1 rev/s in y
    orientation_after_update = Quaternion(axis=[0, 1, 0],
                                          angle=np.pi).elements
    new_orientation = gyro.update(test_rocket, angular_rates, delta_time)
    np.testing.assert_allclose(new_orientation, orientation_after_update, atol=rm.TOLERANCE)


def test_update_orientation_three_quarter_rev_y_axis():
    """
    Test update_orientation() rotating 3/4 of a revolution about the y-axis.
    """
    test_rocket = rm.Rocket()
    gyro = sensors.Gyro()
    delta_time = 0.75  # 3/4 second
    angular_rates = np.array([0, 2 * np.pi, 0])  # 1 rev/s in y
    orientation_after_update = Quaternion(axis=[0, 1, 0],
                                          angle=(3 * np.pi / 2)).elements
    new_orientation = gyro.update(test_rocket, angular_rates, delta_time)
    np.testing.assert_allclose(new_orientation, orientation_after_update, atol=rm.TOLERANCE)


def test_update_orientation_eighth_rev_z_axis():
    """
    Test update_orientation() rotating 1/8 of a revolution about the z-axis.
    """
    test_rocket = rm.Rocket()
    gyro = sensors.Gyro()
    delta_time = 0.125  # 1/8 second
    angular_rates = np.array([0, 0, 2 * np.pi])  # 1 rev/s in z
    orientation_after_update = Quaternion(axis=[0, 0, 1],
                                          angle=(np.pi / 4)).elements
    new_orientation = gyro.update(test_rocket, angular_rates, delta_time)
    np.testing.assert_allclose(new_orientation, orientation_after_update, atol=rm.TOLERANCE)


def test_update_orientation_neg_eighth_rev_z_axis():
    """
    Test update_orientation() rotating -1/8 of a revolution about the z-axis.
    """
    test_rocket = rm.Rocket()
    gyro = sensors.Gyro()
    delta_time = 0.125  # 1/8 second
    angular_rates = np.array([0, 0, -2 * np.pi])  # 1 rev/s in z
    orientation_after_update = Quaternion(axis=[0, 0, 1],
                                          angle=(-np.pi / 4)).elements
    new_orientation = gyro.update(test_rocket, angular_rates, delta_time)
    np.testing.assert_allclose(new_orientation, orientation_after_update, atol=rm.TOLERANCE)


def test_update_orientation_quarter_rev_z_axis():
    """
    Test update_orientation() rotating 1/4 of a revolution about the z-axis.
    """
    test_rocket = rm.Rocket()
    gyro = sensors.Gyro()
    delta_time = 0.25  # 1/4 second
    angular_rates = np.array([0, 0, 2 * np.pi])  # 1 rev/s in z
    orientation_after_update = Quaternion(axis=[0, 0, 1],
                                          angle=(np.pi / 2)).elements
    new_orientation = gyro.update(test_rocket, angular_rates, delta_time)
    np.testing.assert_allclose(new_orientation, orientation_after_update, atol=rm.TOLERANCE)


def test_update_orientation_half_rev_z_axis():
    """
    Test update_orientation() rotating 1/2 of a revolution about the z-axis.
    """
    test_rocket = rm.Rocket()
    gyro = sensors.Gyro()
    delta_time = 0.5  # 1/2 second
    angular_rates = np.array([0, 0, 2 * np.pi])  # 1 rev/s in z
    orientation_after_update = Quaternion(axis=[0, 0, 1],
                                          angle=np.pi).elements
    new_orientation = gyro.update(test_rocket, angular_rates, delta_time)
    np.testing.assert_allclose(new_orientation, orientation_after_update, atol=rm.TOLERANCE)


def test_update_orientation_three_quarter_rev_z_axis():
    """
    Test update_orientation() rotating 3/4 of a revolution about the z-axis.
    """
    test_rocket = rm.Rocket()
    gyro = sensors.Gyro()
    delta_time = 0.75  # 3/4 second
    angular_rates = np.array([0, 0, 2 * np.pi])  # 1 rev/s in z
    orientation_after_update = Quaternion(axis=[0, 0, 1],
                                          angle=(3 * np.pi / 2)).elements
    new_orientation = gyro.update(test_rocket, angular_rates, delta_time)
    np.testing.assert_allclose(new_orientation, orientation_after_update, atol=rm.TOLERANCE)


def test_update_orientation_half_rev_all_axes():
    """
    Test update_orientation() rotating 1/2 of a revolution about all of the
    axes.
    """
    test_rocket = rm.Rocket()
    gyro = sensors.Gyro()
    delta_time = 0.5  # 1/2 second
    angular_rates = np.array([np.pi / np.sqrt(3), np.pi / np.sqrt(3),
                              np.pi / np.sqrt(3)])  # 1/6 rev/s for all
    orientation_after_update = Quaternion(axis=[1, 1, 1],
                                          angle=(np.pi / 2)).elements
    new_orientation = gyro.update(test_rocket, angular_rates, delta_time)
    np.testing.assert_allclose(new_orientation, orientation_after_update, atol=rm.TOLERANCE)


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
    gyro = sensors.Gyro()
    delta_time = 0.25  # 1/4 second
    angular_rates = np.array([2.80992617, 5.61985144, 0])
    orientation_after_update = Quaternion(axis=[1, 2, 0],
                                          angle=(np.pi / 2)).elements
    new_orientation = gyro.update(test_rocket, angular_rates, delta_time)
    np.testing.assert_allclose(new_orientation, orientation_after_update, atol=rm.TOLERANCE)


def test_update_orientation_quarter_rev_y_z_axes():
    """
    Test update_orientation() rotating 1/4 of a revolution about the y and z
    axes.
    """
    test_rocket = rm.Rocket()
    gyro = sensors.Gyro()
    delta_time = 0.25  # 1/4 second
    angular_rates = np.array([0, 2.80992617, 5.61985144])
    orientation_after_update = Quaternion(axis=[0, 1, 2],
                                          angle=(np.pi / 2)).elements
    new_orientation = gyro.update(test_rocket, angular_rates, delta_time)
    np.testing.assert_allclose(new_orientation, orientation_after_update, atol=rm.TOLERANCE)


def test_update_orientation_quarter_rev_x_z_axes():
    """
    Test update_orientation() rotating 1/4 of a revolution about the x and z
    axes.
    """
    test_rocket = rm.Rocket()
    gyro = sensors.Gyro()
    delta_time = 0.25  # 1/4 second
    angular_rates = np.array([5.61985144, 0, 2.80992617])
    orientation_after_update = Quaternion(axis=[2, 0, 1],
                                          angle=(np.pi / 2)).elements
    new_orientation = gyro.update(test_rocket, angular_rates, delta_time)
    np.testing.assert_allclose(new_orientation, orientation_after_update, atol=rm.TOLERANCE)


def test_update_orientation_from_non_identity_quat():
    """
    Test update_orientation() rotating from non-identity quaternion.

    Note: rotation is 1/4 revolution about x axis.
    """
    test_rocket = rm.Rocket()
    gyro = sensors.Gyro()
    orientation = Quaternion(axis=[-1, 0, 0], angle=(np.pi / 2))
    test_rocket.orientation = orientation.elements
    delta_time = 0.25  # 1/4 second
    angular_rates = np.array([2 * np.pi, 0, 0])
    orientation_after_update = Quaternion().elements
    new_orientation = gyro.update(test_rocket, angular_rates, delta_time)
    np.testing.assert_allclose(new_orientation, orientation_after_update, atol=rm.TOLERANCE)


# Testing functions for finding temperature
def test_temperature_at_ground():
    test_rocket = rm.Rocket()
    thermistor = sensors.Thermistor()
    test_rocket.altitude = 0
    assert (abs(thermistor.update(test_rocket) - 15.04) <= rm.TOLERANCE)


def test_temperature_at_100m():
    test_rocket = rm.Rocket()
    thermistor = sensors.Thermistor()
    test_rocket.altitude = 100
    assert (abs(thermistor.update(test_rocket) - 14.391) <= rm.TOLERANCE)


def test_temperature_at_1000m():
    test_rocket = rm.Rocket()
    thermistor = sensors.Thermistor()
    test_rocket.altitude = 1000
    assert (abs(thermistor.update(test_rocket) - 8.55) <= rm.TOLERANCE)


def test_temperature_at_10000m():
    test_rocket = rm.Rocket()
    thermistor = sensors.Thermistor()
    test_rocket.altitude = 10000
    assert (abs(thermistor.update(test_rocket) - -49.86) <= rm.TOLERANCE)


# Testing functions for finding barometric pressure
def test_baro_pressure_at_ground():
    test_rocket = rm.Rocket()
    baro = sensors.Baro_Pressure_Sensor()
    test_rocket.temperature = 15.04
    test_rocket.altitude = 0
    assert (abs(baro.update(test_rocket) - 101.4009) <= rm.TOLERANCE)


def test_baro_pressure_at_100m():
    test_rocket = rm.Rocket()
    baro = sensors.Baro_Pressure_Sensor()
    test_rocket.temperature = 14.391
    test_rocket.altitude = 100
    assert (abs(baro.update(test_rocket) - 100.2062) <= rm.TOLERANCE)


def test_baro_pressure_at_1000m():
    test_rocket = rm.Rocket()
    baro = sensors.Baro_Pressure_Sensor()
    test_rocket.temperature = 8.55
    test_rocket.altitude = 1000
    assert (abs(baro.update(test_rocket) - 89.9581) <= rm.TOLERANCE)


def test_baro_pressure_at_10000m():
    test_rocket = rm.Rocket()
    baro = sensors.Baro_Pressure_Sensor()
    test_rocket.temperature = -49.86
    test_rocket.altitude = 10000
    assert (abs(baro.update(test_rocket) - 26.5162) <= rm.TOLERANCE)


def test_baro_pressure_at_20000m():
    test_rocket = rm.Rocket()
    baro = sensors.Baro_Pressure_Sensor()
    test_rocket.temperature = -56.46
    test_rocket.altitude = 20000
    assert (abs(baro.update(test_rocket) - 5.5298) <= rm.TOLERANCE)


# Testing functions for finding body/proper acceleration
def test_body_acceleration_zeros():
    """
    Test update_body_acceleration() using a zero vector
    """
    test_rocket = rm.Rocket()
    accelerometer = sensors.Accelerometer()
    test_rocket.orientation = Quaternion(axis=[1, 1, 1],
                                         angle=np.pi / 2).elements
    test_rocket.world_acceleration = np.array([0, 0, 0])
    body_acceleration_after_rotate = np.array([0, 0, 0])
    new_body_acceleration = accelerometer.update(test_rocket)
    assert np.all(body_acceleration_after_rotate == new_body_acceleration)


def test_body_acceleration_positive_integers():
    """
    Test update_body_acceleration() using a vector of positive integers
    """
    test_rocket = rm.Rocket()
    accelerometer = sensors.Accelerometer()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1],
                                         angle=np.pi / 2).elements
    test_rocket.world_acceleration = np.array([1, 1, 1])
    body_acceleration_after_rotate = np.array([0.2929, 0, 1.707])
    new_body_acceleration = accelerometer.update(test_rocket)
    np.testing.assert_allclose(new_body_acceleration, body_acceleration_after_rotate,
                               atol=rm.TOLERANCE)


def test_body_acceleration_negative_integers():
    """
    Test update_body_acceleration() using a vector of negative integers
    """
    test_rocket = rm.Rocket()
    accelerometer = sensors.Accelerometer()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1],
                                         angle=np.pi / 2).elements
    test_rocket.world_acceleration = np.array([-1, -1, -3])
    body_acceleration_after_rotate = np.array([-1.2929, 1.4142, -2.7071])
    new_body_acceleration = accelerometer.update(test_rocket)
    np.testing.assert_allclose(new_body_acceleration, body_acceleration_after_rotate,
                               atol=rm.TOLERANCE)


def test_body_acceleration_positive_floats():
    """
    Test update_body_acceleration() using a vector of positive floats
    """
    test_rocket = rm.Rocket()
    accelerometer = sensors.Accelerometer()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1],
                                         angle=np.pi / 2).elements
    test_rocket.world_acceleration = np.array([1.5, 2.5, 3.5])
    body_acceleration_after_rotate = np.array([0.7322, -1.4142, 4.2678])
    new_body_acceleration = accelerometer.update(test_rocket)
    np.testing.assert_allclose(new_body_acceleration, body_acceleration_after_rotate,
                               atol=rm.TOLERANCE)


def test_body_acceleration_negative_floats():
    """
    Test update_body_acceleration() using a vector of negative floats
    """
    test_rocket = rm.Rocket()
    accelerometer = sensors.Accelerometer()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1],
                                         angle=np.pi / 2).elements
    test_rocket.world_acceleration = np.array([-1.5, -2, -3])
    body_acceleration_after_rotate = np.array([-0.8358, 1.0606, -3.6642])
    new_body_acceleration = accelerometer.update(test_rocket)
    np.testing.assert_allclose(new_body_acceleration, body_acceleration_after_rotate,
                               atol=rm.TOLERANCE)


def test_body_acceleration_45_degrees():
    """
    Test update_body_acceleration() at an angle of 45 degrees
    """
    test_rocket = rm.Rocket()
    accelerometer = sensors.Accelerometer()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1],
                                         angle=np.pi / 4).elements
    test_rocket.world_acceleration = np.array([1, 1, 1])
    body_acceleration_after_rotate = np.array([0.5, 0.7071, 1.5])
    new_body_acceleration = accelerometer.update(test_rocket)
    np.testing.assert_allclose(new_body_acceleration, body_acceleration_after_rotate,
                               atol=rm.TOLERANCE)


def test_body_acceleration_180_degrees():
    """
    Test update_body_acceleration() at an angle of 180 degrees
    """
    test_rocket = rm.Rocket()
    accelerometer = sensors.Accelerometer()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1], angle=np.pi).elements
    test_rocket.world_acceleration = np.array([1, 1, 1])
    body_acceleration_after_rotate = np.array([1, -1, 1])
    new_body_acceleration = accelerometer.update(test_rocket)
    np.testing.assert_allclose(new_body_acceleration, body_acceleration_after_rotate,
                               atol=rm.TOLERANCE)


# Testing functions for finding the magnetic field around the rocket
def test_magnetic_field_zeros():
    """
    Test update_magnetic_field() using a zero vector
    """
    test_rocket = rm.Rocket()
    magnetometer = sensors.Magnetometer()
    test_rocket.orientation = Quaternion(axis=[1, 1, 1],
                                         angle=np.pi / 2).elements
    test_rocket.world_mag_field = np.array([0, 0, 0])
    mag_field_after_rotate = np.array([0, 0, 0])
    new_mag_field = magnetometer.update(test_rocket)
    np.testing.assert_allclose(new_mag_field, mag_field_after_rotate, atol=rm.TOLERANCE)


def test_magnetic_field_positive_integers():
    """
    Test update_magnetic_field() using a vector of positive integers
    """
    test_rocket = rm.Rocket()
    magnetometer = sensors.Magnetometer()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1],
                                         angle=np.pi / 2).elements
    test_rocket.world_mag_field = np.array([1, 1, 1])
    mag_field_after_rotate = np.array([0.2929, 0, 1.707])
    new_mag_field = magnetometer.update(test_rocket)
    np.testing.assert_allclose(new_mag_field, mag_field_after_rotate, atol=rm.TOLERANCE)


def test_magnetic_field_negative_integers():
    test_rocket = rm.Rocket()
    magnetometer = sensors.Magnetometer()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1],
                                         angle=np.pi / 2).elements
    test_rocket.world_mag_field = np.array([-1, -1, -3])
    mag_field_after_rotate = np.array([-1.2929, 1.4142, -2.7071])
    new_mag_field = magnetometer.update(test_rocket)
    np.testing.assert_allclose(new_mag_field, mag_field_after_rotate, atol=rm.TOLERANCE)


def test_magnetic_field_positive_floats():
    """
    Test update_magnetic_field() using a vector of positive floats
    """
    test_rocket = rm.Rocket()
    magnetometer = sensors.Magnetometer()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1],
                                         angle=np.pi / 2).elements
    test_rocket.world_mag_field = np.array([1.5, 2.5, 3.5])
    mag_field_after_rotate = np.array([0.7322, -1.4142, 4.2678])
    new_mag_field = magnetometer.update(test_rocket)
    np.testing.assert_allclose(new_mag_field, mag_field_after_rotate, atol=rm.TOLERANCE)


def test_magnetic_field_negative_floats():
    """
    Test update_magnetic_field() using a vector of negative floats
    """
    test_rocket = rm.Rocket()
    magnetometer = sensors.Magnetometer()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1],
                                         angle=np.pi / 2).elements
    test_rocket.world_mag_field = np.array([-1.5, -2, -3])
    mag_field_after_rotate = np.array([-0.8358, 1.0606, -3.6642])
    new_mag_field = magnetometer.update(test_rocket)
    np.testing.assert_allclose(new_mag_field, mag_field_after_rotate, atol=rm.TOLERANCE)


def test_magnetic_field_45_degrees():
    """
    Test update_magnetic_field() at an angle of 45 degrees
    """
    test_rocket = rm.Rocket()
    magnetometer = sensors.Magnetometer()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1],
                                         angle=np.pi / 4).elements
    test_rocket.world_mag_field = np.array([1, 1, 1])
    mag_field_after_rotate = np.array([0.5, 0.7071, 1.5])
    new_mag_field = magnetometer.update(test_rocket)
    np.testing.assert_allclose(new_mag_field, mag_field_after_rotate, atol=rm.TOLERANCE)


def test_magnetic_field_180_degrees():
    """
    Test update_magnetic_field() at an angle of 180 degrees
    """
    test_rocket = rm.Rocket()
    magnetometer = sensors.Magnetometer()
    test_rocket.orientation = Quaternion(axis=[1, 0, 1], angle=np.pi).elements
    test_rocket.world_mag_field = np.array([1, 1, 1])
    mag_field_after_rotate = np.array([1, -1, 1])
    new_mag_field = magnetometer.update(test_rocket)
    np.testing.assert_allclose(new_mag_field, mag_field_after_rotate, atol=rm.TOLERANCE)
