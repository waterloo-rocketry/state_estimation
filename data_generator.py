"""
This file controls the main functionality of the state estimation
data generation.
"""

import os

import numpy as np
from tabulate import tabulate

import rocket_math as rm
import sensors

# -----------------------CONSTANTS---------------------------
# File Paths:
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
GEN_FILES_PATH = os.path.join(ROOT_PATH, "generated_files")
# Check if the 'generated_files' directory already exists
if not os.path.isdir(GEN_FILES_PATH):
    os.mkdir(GEN_FILES_PATH)
GT_PATH = os.path.join(GEN_FILES_PATH, "ground_truth.txt")
SD_PATH = os.path.join(GEN_FILES_PATH, "sensor_data.txt")
# -----------------------------------------------------------

# --------------------NUMPY PRINT OPTIONS--------------------
# Variable Precision:
np.set_printoptions(precision=4)
# Disable Scientific Notation:
np.set_printoptions(suppress=True)
# -----------------------------------------------------------

# -------------------SENSORS LIST INDEXES--------------------
GYRO_SENSOR_IND = 0
THERMISTOR_SENSOR_IND = 1
BARO_PRESSURE_SENSOR_IND = 2
ACCELEROMETER_SENSOR_IND = 3
MAGNETOMETER_SENSOR_IND = 4

# -----------------------------------------------------------


def is_any_negative(array):
    """
    Determines if any element of an array is negative.

    Parameters
    ----------
    array: numpy.array

    Returns
    -------
    bool
        True if there is a  negative element, false otherwise.
    """
    return any(array < 0)


# TODO: add exception blocks
def init_rocket_state() -> rm.Rocket:
    """
    Initializes the Rocket object with user input.

    Returns
    -------
    current_rocket: Rocket
        Rocket object to be used to generate data sets based on user input.
    """
    # Get user input
    total_mass = thrust = burn_time = press_noise = temp_noise = \
        accel_noise = gyro_noise = mag_noise = 0
    valid_user_input = False
    while not valid_user_input:
        total_mass, thrust, burn_time, press_noise, temp_noise, accel_noise, \
        gyro_noise, mag_noise = input(
            "Enter a total mass, thrust, burn time, and noise params "
            "(pressure, temperature, acceleration, gyro, and magnetic "
            "noise). Please separate each value with a space: ").split()

        total_mass = float(total_mass)
        thrust = np.fromstring(thrust, dtype=float, sep=",")
        burn_time = float(burn_time)
        if total_mass < 0 or burn_time < 0 or is_any_negative(thrust):
            print(
                "User input is invalid. Reason: one or more inputs required "
                "to be positive, is negative. Please try again.")
        else:
            valid_user_input = True

    return rm.Rocket({"total_mass": total_mass, "body_mass": rm.BODY_MASS,
                      "prop_mass": total_mass - rm.BODY_MASS},
                     thrust, burn_time, {"press_noise": float(press_noise),
                                         "temp_noise": float(temp_noise),
                                         "accel_noise": float(accel_noise),
                                         "gyro_noise": float(gyro_noise),
                                         "mag_noise": float(mag_noise)})


def init_sensors() -> list:
    """
    Initializes the sensor objects.

    Returns
    -------
    sensors: numpy.array
        The sensors available to be used with the main Rocket Object.
    """
    sens_list = []
    sens_list.insert(GYRO_SENSOR_IND, sensors.Gyro(calibration=1))
    sens_list.insert(THERMISTOR_SENSOR_IND, sensors.Thermistor(calibration=1))
    sens_list.insert(BARO_PRESSURE_SENSOR_IND, sensors.Baro_Pressure_Sensor(calibration=1))
    sens_list.insert(ACCELEROMETER_SENSOR_IND, sensors.Accelerometer(calibration=1))
    sens_list.insert(MAGNETOMETER_SENSOR_IND, sensors.Magnetometer(calibration=1))

    return sens_list


def time_update(rocket, sensors_list, current_time, timestep):
    """
    Updates the state of the Rocket for every timestep.

    Parameters
    ----------

    rocket: Rocket
        Rocket object to update.
    sensors_list: list
        Array of sensors used with Rocket Object to update.
    current_time: float
        Current time during the data generation process
    timestep: float
        The length of time current_time is increased by for a step forward in "simulated" time.
    """
    # Calculate updated state from previous timestep's state
    updated_position = rocket.update_position(timestep)
    updated_velocity = rocket.update_velocity(timestep)
    updated_acceleration = rocket.update_acceleration()
    updated_thrust = rocket.update_thrust(current_time)
    updated_mass = rocket.update_mass(timestep)
    updated_world_mag_field = rocket.update_world_magnetic_field()

    updated_orientation = sensors_list[GYRO_SENSOR_IND].update(rocket,
                                                               rm.ANGULAR_RATES,
                                                               timestep)
    updated_temperature = sensors_list[THERMISTOR_SENSOR_IND].update(rocket)
    updated_baro_pressure = sensors_list[BARO_PRESSURE_SENSOR_IND].update(
        rocket)
    updated_body_acceleration = sensors_list[ACCELEROMETER_SENSOR_IND].update(
        rocket)
    updated_body_mag_field = sensors_list[MAGNETOMETER_SENSOR_IND].update(
        rocket)

    # Update the Rocket object
    rocket.position = updated_position
    rocket.velocity = updated_velocity
    rocket.world_acceleration = updated_acceleration
    rocket.thrust = updated_thrust
    rocket.mass = updated_mass
    rocket.world_mag_field = updated_world_mag_field
    rocket.altitude = rocket.position[2]

    rocket.orientation = updated_orientation
    rocket.temperature = updated_temperature
    rocket.baro_pressure = updated_baro_pressure
    rocket.body_acceleration = updated_body_acceleration
    rocket.body_mag_field = updated_body_mag_field


def main():
    """
    Main function for generating data based on the input Rocket.
    """
    # Simulated time setup
    current_time = 0
    end_time = 100
    timestep = 0.01

    # Data lists and headings initializations
    gt_gen_data = []
    sensor_gen_data = []
    headings_gt = ["Position [m]", "Velocity [m/s]", "Acceleration [m/s^2]",
                   "Orientation"]
    headings_sd = ["Baro_Pressure [KPa]", "Temperature [Celsius]",
                   "Acceleration [m/s^2]", "Magnetic_Field [T]"]

    with open(GT_PATH, "w") as ground_truth, open(SD_PATH, "w") as sensor_data:
        # Get the initial rocket state
        current_rocket = init_rocket_state()

        # Init sensors
        sensors_list = init_sensors()

        # Update state and write data to file
        while current_time < end_time:
            # Update rocket params with current timestep
            time_update(current_rocket, sensors_list, current_time, timestep)
            gt_gen_data.append(
                [current_rocket.position, current_rocket.velocity,
                 current_rocket.world_acceleration,
                 current_rocket.orientation])
            sensor_gen_data.append(
                [current_rocket.baro_pressure,
                 current_rocket.temperature,
                 current_rocket.body_acceleration,
                 current_rocket.body_mag_field])
            current_time += timestep
        # Write generated data to ground truth and sensor data files
        ground_truth.write(
            tabulate(gt_gen_data, headers=headings_gt, tablefmt="rst"))
        sensor_data.write(
            tabulate(sensor_gen_data, headers=headings_sd, tablefmt="rst",
                     numalign="left", floatfmt=".4f"))


if __name__ == "__main__":
    main()
