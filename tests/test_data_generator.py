# Test suite for data_generator.py

import numpy as np

import rocket_math as rm
import data_generator as data_gen
import sensors

'''
Notes:
- For all calculations, there will be an tolerance of 0.001.
'''


# Helper Function for testing sensors in update functions
def get_sensor_list():
    return list(
        [sensors.Gyro(), sensors.Thermistor(), sensors.Baro_Pressure_Sensor(),
         sensors.Accelerometer(), sensors.Magnetometer()])


# Testing functions for is_negative_values().
def test_no_negative_values():
    assert not data_gen.is_any_negative(np.array([0, 1, 2]))


def test_one_negative_int():
    assert data_gen.is_any_negative(np.array([0, 1, -2]))


def test_all_negative_ints():
    assert data_gen.is_any_negative(np.array([-1, -2, -3]))


def test_one_negative_float():
    assert data_gen.is_any_negative(np.array([1.1, -2.2, 3.3]))


def test_all_negative_floats():
    assert data_gen.is_any_negative(np.array([-1.1, -2.2, -3.3]))


# Testing functions for init_rocket_state().
def test_no_negative_init_inputs(monkeypatch):
    """
    Test init_rocket_state() with no negative inputs.
    """
    rocket = rm.Rocket(
        {"total_mass": 110, "body_mass": rm.BODY_MASS,
         "prop_mass": 110 - rm.BODY_MASS},
        np.array([1, 2, 3]), 120,
        {"press_noise": 1, "temp_noise": 2, "accel_noise": 3,
         "gyro_noise": 4, "mag_noise": 5})
    rocket_params = '110 1,2,3 120 1 2 3 4 5\n'
    monkeypatch.setattr('builtins.input', lambda user_input: rocket_params)
    test_rocket = data_gen.init_rocket_state()
    assert test_rocket == rocket


def test_no_negative_init_inputs_again(monkeypatch):
    """
    Test init_rocket_state() a second time with no negative inputs.
    """
    rocket = rm.Rocket(
        {"total_mass": 120, "body_mass": rm.BODY_MASS,
         "prop_mass": 120 - rm.BODY_MASS},
        np.array([10, 20, 30]), 130,
        {"press_noise": 10, "temp_noise": 20, "accel_noise": 30,
         "gyro_noise": 40, "mag_noise": 50})
    rocket_params = '120 10,20,30 130 10 20 30 40 50\n'
    monkeypatch.setattr('builtins.input', lambda user_input: rocket_params)
    test_rocket = data_gen.init_rocket_state()
    assert test_rocket == rocket


def test_mass_negative_init_input(monkeypatch):
    """
    Test init_rocket_state() with total_mass as a negative input, which is
    then corrected to a positive input.
    """
    rocket = rm.Rocket(
        {"total_mass": 120, "body_mass": rm.BODY_MASS,
         "prop_mass": 120 - rm.BODY_MASS},
        np.array([10, 20, 30]), 130,
        {"press_noise": 10, "temp_noise": 20, "accel_noise": 30,
         "gyro_noise": 40, "mag_noise": 50})
    rocket_params = iter(["-120 10,20,30 130 10 20 30 40 50\n",
                          "120 10,20,30 130 10 20 30 40 50\n"])
    monkeypatch.setattr('builtins.input',
                        lambda user_input: next(rocket_params))
    test_rocket = data_gen.init_rocket_state()
    assert test_rocket == rocket


def test_thrust_negative_init_input(monkeypatch):
    """
    Test init_rocket_state() with thrust as a negative input, which is then
    corrected to a positive input.
    """
    rocket = rm.Rocket(
        {"total_mass": 120, "body_mass": rm.BODY_MASS,
         "prop_mass": 120 - rm.BODY_MASS},
        np.array([10, 20, 30]), 130,
        {"press_noise": 10, "temp_noise": 20, "accel_noise": 30,
         "gyro_noise": 40, "mag_noise": 50})
    rocket_params = iter(["120 -10,20,30 130 10 20 30 40 50\n",
                          "120 10,20,30 130 10 20 30 40 50\n"])
    monkeypatch.setattr('builtins.input',
                        lambda user_input: next(rocket_params))
    test_rocket = data_gen.init_rocket_state()
    assert test_rocket == rocket


def test_burn_time_negative_init_input(monkeypatch):
    """
    Test init_rocket_state() with burn_time as a negative input, which is then
    corrected to a positive input.
    """
    rocket = rm.Rocket(
        {"total_mass": 120, "body_mass": rm.BODY_MASS,
         "prop_mass": 120 - rm.BODY_MASS},
        np.array([10, 20, 30]), 130,
        {"press_noise": 10, "temp_noise": 20, "accel_noise": 30,
         "gyro_noise": 40, "mag_noise": 50})
    rocket_params = iter(["120 10,20,30 -130 10 20 30 40 50\n",
                          "120 10,20,30 130 10 20 30 40 50\n"])
    monkeypatch.setattr('builtins.input',
                        lambda user_input: next(rocket_params))
    test_rocket = data_gen.init_rocket_state()
    assert test_rocket == rocket


def test_all_negative_init_inputs(monkeypatch):
    """
    Test init_rocket_state() with mass, thrust, and burn_time as negative
    inputs, which are then corrected to positive inputs.
    """
    rocket = rm.Rocket(
        {"total_mass": 120, "body_mass": rm.BODY_MASS,
         "prop_mass": 120 - rm.BODY_MASS},
        np.array([10, 20, 30]), 130,
        {"press_noise": 10, "temp_noise": 20, "accel_noise": 30,
         "gyro_noise": 40, "mag_noise": 50})
    rocket_params = iter(["-120 -10,20,30 -130 10 20 30 40 50\n",
                          "120 10,20,30 130 10 20 30 40 50\n"])
    monkeypatch.setattr('builtins.input',
                        lambda user_input: next(rocket_params))
    test_rocket = data_gen.init_rocket_state()
    assert test_rocket == rocket


# Testing functions for time_update().
def test_initial_time_update(mocker):
    """
    Test time_update() from initial state (initial launch).
    """
    time_dict = {"current_time": 0, "previous_time": 0, "timestep": 1}
    timestep = 0.1
    test_rocket = rm.Rocket(
        {"total_mass": 110, "body_mass": 55, "prop_mass": 55},
        np.array([0, 0, 20000]), 100,
        {"press_noise": 1, "temp_noise": 1, "accel_noise": 1,
         "gyro_noise": 1, "mag_noise": 1})
    test_rocket_after_update = rm.Rocket(
        {"total_mass": 110 - rm.MASS_LOSS * time_dict["timestep"],
         "body_mass": 55,
         "prop_mass": 55 - rm.MASS_LOSS * time_dict["timestep"]},
        np.array([0, 0, 20000]), 100, {"press_noise": 1, "temp_noise": 1,
                                       "accel_noise": 1, "gyro_noise": 1,
                                       "mag_noise": 1})
    test_rocket_after_update.world_acceleration = np.array([0, 0, 149.6793])
    test_rocket_after_update.orientation = \
        np.array([0.6216, 0, 0.0044, 0.7833])
    test_rocket_after_update.temperature = 20
    test_rocket_after_update.baro_pressure = 10
    test_rocket_after_update.body_acceleration = np.array([1, 1, 1])
    test_rocket_after_update.body_mag_field = np.array([0.5, 0.5, 0.5])
    test_rocket_after_update.world_mag_field = np.array([0.5, 0.5, 0.5])
    mocker.patch('rocket_math.Rocket.update_thrust',
                 return_value=np.array([0, 0, 20000]))
    mocker.patch('rocket_math.Rocket.update_acceleration',
                 return_value=np.array([0, 0, 149.6793]))
    mocker.patch('rocket_math.Rocket.update_velocity',
                 return_value=np.array([0, 0, 0]))
    mocker.patch('rocket_math.Rocket.update_position',
                 return_value=np.array([0, 0, 0]))
    mocker.patch('rocket_math.Rocket.update_mass', return_value={
        "total_mass": 110 - rm.MASS_LOSS * time_dict["timestep"],
        "body_mass": 55,
        "prop_mass": 55 - rm.MASS_LOSS * time_dict["timestep"]})
    mocker.patch('rocket_math.Rocket.update_world_magnetic_field',
                 return_value=np.array([0.5, 0.5, 0.5]))
    mocker.patch('sensors.Gyro.update',
                 return_value=np.array([0.6216, 0, 0.0044, 0.7833]))
    mocker.patch('sensors.Thermistor.update',
                 return_value=20)
    mocker.patch('sensors.Baro_Pressure_Sensor.update',
                 return_value=10)
    mocker.patch('sensors.Accelerometer.update',
                 return_value=np.array([1, 1, 1]))
    mocker.patch('sensors.Magnetometer.update',
                 return_value=np.array([0.5, 0.5, 0.5]))
    data_gen.time_update(test_rocket, get_sensor_list(), time_dict["current_time"], timestep)
    assert test_rocket == test_rocket_after_update


def test_secondary_time_update(mocker):
    """
    Test time_update() from secondary state (after rocket has launched).
    """
    time_dict = {"current_time": 1, "previous_time": 0, "timestep": 1}
    timestep = 0.1
    test_rocket = rm.Rocket(
        {"total_mass": 110 - rm.MASS_LOSS * time_dict["timestep"],
         "body_mass": 55,
         "prop_mass": 55 - rm.MASS_LOSS * time_dict["timestep"]},
        np.array([0, 0, 20000]), 100, {"press_noise": 1, "temp_noise": 1,
                                       "accel_noise": 1, "gyro_noise": 1,
                                       "mag_noise": 1})
    test_rocket_after_update = rm.Rocket(
        {"total_mass": 110 - 2 * rm.MASS_LOSS * time_dict["timestep"],
         "body_mass": 55,
         "prop_mass": 55 - 2 * rm.MASS_LOSS * time_dict["timestep"]},
        np.array([0, 0, 20000]), 100, {"press_noise": 1, "temp_noise": 1,
                                       "accel_noise": 1, "gyro_noise": 1,
                                       "mag_noise": 1})
    test_rocket_after_update.world_acceleration = np.array([0, 0, 149.762])
    test_rocket_after_update.velocity = np.array([0, 0, 149.762])
    test_rocket_after_update.position = np.array([0, 0, 149.762])
    test_rocket_after_update.altitude = 149.762
    test_rocket_after_update.orientation = \
        np.array([-0.2272, 0, 0.0054, 0.9738])
    test_rocket_after_update.temperature = 25
    test_rocket_after_update.baro_pressure = 15
    test_rocket_after_update.body_acceleration = np.array([3, 3, 3])
    test_rocket_after_update.body_mag_field = np.array([1, 1, 1])
    test_rocket_after_update.world_mag_field = np.array([1, 1, 1])
    mocker.patch('rocket_math.Rocket.update_thrust',
                 return_value=np.array([0, 0, 20000]))
    mocker.patch('rocket_math.Rocket.update_acceleration',
                 return_value=np.array([0, 0, 149.762]))
    mocker.patch('rocket_math.Rocket.update_velocity',
                 return_value=np.array([0, 0, 149.762]))
    mocker.patch('rocket_math.Rocket.update_position',
                 return_value=np.array([0, 0, 149.762]))
    mocker.patch('rocket_math.Rocket.update_mass', return_value={
        "total_mass": 110 - 2 * rm.MASS_LOSS * time_dict["timestep"],
        "body_mass": 55,
        "prop_mass": 55 - 2 * rm.MASS_LOSS * time_dict["timestep"]})
    mocker.patch('rocket_math.Rocket.update_world_magnetic_field',
                 return_value=np.array([1, 1, 1]))
    mocker.patch('sensors.Gyro.update',
                 return_value=np.array([-0.2272, 0, 0.0054, 0.9738]))
    mocker.patch('sensors.Thermistor.update',
                 return_value=25)
    mocker.patch('sensors.Baro_Pressure_Sensor.update',
                 return_value=15)
    mocker.patch('sensors.Accelerometer.update',
                 return_value=np.array([3, 3, 3]))
    mocker.patch('sensors.Magnetometer.update',
                 return_value=np.array([1, 1, 1]))
    data_gen.time_update(test_rocket, get_sensor_list(), time_dict["current_time"], timestep)
    assert test_rocket == test_rocket_after_update
