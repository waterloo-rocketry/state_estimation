# Test suite for data_generator.py
from io import StringIO

import numpy as np

import rocket_math as rm
import data_generator as data_gen

'''
Notes:
- For all calculations, there will be an tolerance of 0.001.
'''


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
    mocker.patch('rocket_math.Rocket.update_orientation',
                 return_value=np.array([0.6216, 0, 0.0044, 0.7833]))
    mocker.patch('rocket_math.Rocket.update_temperature',
                 return_value=20)
    mocker.patch('rocket_math.Rocket.update_baro_pressure',
                 return_value=10)
    mocker.patch('rocket_math.Rocket.update_body_acceleration',
                 return_value=np.array([1, 1, 1]))
    mocker.patch('rocket_math.Rocket.update_magnetic_field',
                 return_value=np.array([0.5, 0.5, 0.5]))
    data_gen.time_update(test_rocket, time_dict)
    assert test_rocket == test_rocket_after_update


def test_secondary_time_update(mocker):
    """
    Test time_update() from secondary state (after rocket has launched).
    """
    time_dict = {"current_time": 1, "previous_time": 0, "timestep": 1}
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
    mocker.patch('rocket_math.Rocket.update_orientation',
                 return_value=np.array([-0.2272, 0, 0.0054, 0.9738]))
    mocker.patch('rocket_math.Rocket.update_temperature',
                 return_value=25)
    mocker.patch('rocket_math.Rocket.update_baro_pressure',
                 return_value=15)
    mocker.patch('rocket_math.Rocket.update_body_acceleration',
                 return_value=np.array([3, 3, 3]))
    mocker.patch('rocket_math.Rocket.update_magnetic_field',
                 return_value=np.array([1, 1, 1]))
    data_gen.time_update(test_rocket, time_dict)
    assert test_rocket == test_rocket_after_update


# Test functions for write_data_to_file().
def test_initial_write():
    """
    Test write_data_to_file() with initial write to file.
    """
    valid_file = StringIO("[0.0000 0.0000 0.0000]            "
                          "[0.0000 0.0000 0.0000]            "
                          "[0.0000 0.0000 0.0000]            "
                          "[1.0000 0.0000 0.0000 0.0000]    \n")
    valid_sensor_file = StringIO("20.0000                           "
                                 "40.0000                           "
                                 "[0.0000 0.0000 1.0000]            "
                                 "[0.0000 0.0000 1.0000]           \n")
    test_rocket = rm.Rocket(
        {"total_mass": 110, "body_mass": 55, "prop_mass": 55},
        np.array([0, 0, 0]), 100,
        {"press_noise": 1, "temp_noise": 1, "accel_noise": 1,
         "gyro_noise": 1, "mag_noise": 1})
    test_rocket.temperature = 40.0
    test_rocket.baro_pressure = 20.0
    test_rocket.body_acceleration = np.array([0.0, 0.0, 1.0])
    test_rocket.body_mag_field = np.array([0.0, 0.0, 1.0])

    gt_test_file = StringIO()
    sd_test_file = StringIO()
    data_gen.write_data_to_file(test_rocket, gt_test_file, sd_test_file)
    gt_file_value = gt_test_file.getvalue()
    sd_file_value = sd_test_file.getvalue()
    valid = valid_file.getvalue()
    sensor_valid = valid_sensor_file.getvalue()

    gt_test_file.close()
    sd_test_file.close()
    valid_file.close()
    valid_sensor_file.close()
    assert gt_file_value == valid and sd_file_value == sensor_valid


def test_two_writes():
    """
    Test write_data_to_file() with writing multiple consecutive lines to file.
    """
    valid_file = StringIO("[0.0000 0.0000 0.0000]            "
                          "[0.0000 0.0000 0.0000]            "
                          "[  0.0000   0.0000 149.6793]      "
                          "[0.6216 0.0000 0.0044 0.7833]    \n")

    valid_sensor_file = StringIO("20.0000                           "
                                 "40.0000                           "
                                 "[0.0000 0.0000 1.0000]            "
                                 "[0.0000 0.0000 1.0000]           \n")

    test_rocket = rm.Rocket(
        {"total_mass": 110, "body_mass": 55, "prop_mass": 55},
        np.array([0, 0, 20000]), 100,
        {"press_noise": 1, "temp_noise": 1, "accel_noise": 1,
         "gyro_noise": 1, "mag_noise": 1})
    test_rocket.world_acceleration = np.array([0, 0, 149.6793])
    test_rocket.orientation = np.array([0.6216, 0, 0.0044, 0.7833])
    test_rocket.temperature = 40.0
    test_rocket.baro_pressure = 20.0
    test_rocket.body_acceleration = np.array([0.0, 0.0, 1.0])
    test_rocket.body_mag_field = np.array([0.0, 0.0, 1.0])

    gt_test_file = StringIO()
    sd_test_file = StringIO()
    data_gen.write_data_to_file(test_rocket, gt_test_file, sd_test_file)
    gt_file_value1 = gt_test_file.getvalue()
    sd_file_value1 = sd_test_file.getvalue()
    valid1 = valid_file.getvalue()
    sensor_valid1 = valid_sensor_file.getvalue()

    valid_file = StringIO("[0.0000 0.0000 0.0000]            "
                          "[0.0000 0.0000 0.0000]            "
                          "[  0.0000   0.0000 149.6793]      "
                          "[0.6216 0.0000 0.0044 0.7833]    \n"
                          "[0.0000 0.0000 0.0150]            "
                          "[0.0000 0.0000 1.4976]            "
                          "[  0.0000   0.0000 149.7620]      "
                          "[-0.2272  0.0000  0.0054  0.9738]\n")
    valid_sensor_file = StringIO("20.0000                           "
                                 "40.0000                           "
                                 "[0.0000 0.0000 1.0000]            "
                                 "[0.0000 0.0000 1.0000]           \n"
                                 "30.0000                           "
                                 "50.0000                           "
                                 "[0.0000 0.0000 2.0000]            "
                                 "[0.0000 0.0000 2.0000]           \n")

    test_rocket.world_acceleration = np.array([0, 0, 149.7620])
    test_rocket.velocity = np.array([0, 0, 1.4976])
    test_rocket.position = np.array([0, 0, 0.0150])
    test_rocket.orientation = np.array([-0.2272, 0.0000, 0.0054, 0.9738])
    test_rocket.temperature = 50.0000
    test_rocket.baro_pressure = 30.0000
    test_rocket.body_acceleration = np.array([0.0000, 0.0000, 2.0000])
    test_rocket.body_mag_field = np.array([0.0000, 0.0000, 2.0000])
    data_gen.write_data_to_file(test_rocket, gt_test_file, sd_test_file)
    gt_file_value2 = gt_test_file.getvalue()
    sd_file_value2 = sd_test_file.getvalue()
    valid2 = valid_file.getvalue()
    sensor_valid2 = valid_sensor_file.getvalue()
    gt_test_file.close()
    sd_test_file.close()
    valid_file.close()
    valid_sensor_file.close()
    assert gt_file_value1 == valid1 and gt_file_value2 == valid2 and \
           sd_file_value1 == sensor_valid1 and sd_file_value2 == sensor_valid2


def test_full_length_write():
    """
    Test write_data_to_file() writing data that is at its full length.
    """
    valid_file = StringIO("[11111.1111 11111.1111 11111.1111] "
                          "[22222.2222 22222.2222 22222.2222] "
                          "[33333.3333 33333.3333 33333.3333] "
                          "[44444.4444 44444.4444 44444.4444 44444.4444]\n")
    sensor_valid_file = StringIO("11111.1111                        "
                                 "22222.2222                        "
                                 "[33333.3333 33333.3333 33333.3333] "
                                 "[44444.4444 44444.4444 44444.4444]\n")
    test_rocket = rm.Rocket(
        {"total_mass": 110, "body_mass": 55, "prop_mass": 55},
        np.array([0, 0, 0]), 100,
        {"press_noise": 1, "temp_noise": 1, "accel_noise": 1,
         "gyro_noise": 1, "mag_noise": 1})
    test_rocket.world_acceleration = np.array(
        [33333.3333, 33333.3333, 33333.3333])
    test_rocket.velocity = np.array([22222.2222, 22222.2222, 22222.2222])
    test_rocket.position = np.array([11111.1111, 11111.1111, 11111.1111])
    test_rocket.orientation = \
        np.array([44444.4444, 44444.4444, 44444.4444, 44444.4444])
    test_rocket.temperature = 22222.2222
    test_rocket.baro_pressure = 11111.1111
    test_rocket.body_acceleration = np.array(
        [33333.3333, 33333.3333, 33333.3333])
    test_rocket.body_mag_field = np.array([44444.4444, 44444.4444, 44444.4444])

    gt_test_file = StringIO()
    sd_test_file = StringIO()
    data_gen.write_data_to_file(test_rocket, gt_test_file, sd_test_file)
    gt_file_value = gt_test_file.getvalue()
    sd_file_value = sd_test_file.getvalue()
    valid = valid_file.getvalue()
    sensor_valid = sensor_valid_file.getvalue()
    gt_test_file.close()
    sd_test_file.close()
    assert gt_file_value == valid and sd_file_value == sensor_valid
