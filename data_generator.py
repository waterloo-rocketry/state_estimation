"""
This file controls the main functionality of the state estimation
data generation.
"""

import os

import numpy as np

import rocket_math as rm

# -----------------------CONSTANTS---------------------------
# File Paths:
ROOT_PATH = os.path.dirname(os.path.abspath("requirements.txt"))
GT_PATH = os.path.join(ROOT_PATH, "generated_files", "ground_truth.txt")
SD_PATH = os.path.join(ROOT_PATH, "generated_files", "sensor_data.txt")


# -----------------------------------------------------------

# Checks if an array has any negative elements
def is_any_negative(array):
    for element in array:
        if element < 0:
            return True
    return False


# TODO: add exception blocks
# Initializes rocket object from user input
def init_rocket_state() -> rm.Rocket:
    # Initializing the current rocket
    current_rocket = rm.Rocket(
        {"total_mass": 0, "body_mass": rm.BODY_MASS, "comb_mass": 0},
        np.array([0, 0, 0]), 0,
        {"press_noise": 0, "temp_noise": 0, "accel_noise": 0,
         "gyro_noise": 0,
         "mag_noise": 0})
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
    current_rocket.mass["total_mass"] = total_mass
    current_rocket.mass["comb_mass"] = total_mass - current_rocket.mass[
        "body_mass"]
    current_rocket.thrust = thrust
    current_rocket.burn_time = burn_time
    current_rocket.sensor_noise["press_noise"] = float(press_noise)
    current_rocket.sensor_noise["temp_noise"] = float(temp_noise)
    current_rocket.sensor_noise["accel_noise"] = float(accel_noise)
    current_rocket.sensor_noise["gyro_noise"] = float(gyro_noise)
    current_rocket.sensor_noise["mag_noise"] = float(mag_noise)
    return current_rocket


# Updates the state of the rocket
# TODO: add sensor updates
def time_update(rocket, time_dict):
    rocket.thrust = rocket.rocket_thrust(time_dict["current_time"])
    rocket.acceleration = rocket.rocket_acceleration()
    rocket.velocity = rocket.rocket_velocity(time_dict["current_time"],
                                             time_dict["previous_time"])
    rocket.position = rocket.rocket_position(time_dict["current_time"],
                                             time_dict["previous_time"])
    rocket.altitude = rocket.position[2]
    rocket.mass = rocket.rocket_mass()


# Writes the state of the rocket to the ground_truth and sensor_data files
# TODO: add writing to sensor_data files
def write_data_to_file(rocket, gt_file, sd_file):
    new_data_gt = np.array(
        [rocket.position, rocket.velocity,
         rocket.acceleration, rocket.thrust])
    data_gt = ["", "", "", ""]
    for i, data_elem_gt in enumerate(new_data_gt):
        data_gt[i] = np.array2string(data_elem_gt, precision=4,
                                     floatmode='fixed')
    data_to_write = ' '.join(["{0: <33}".format(data) for data in data_gt])
    gt_file.write(data_to_write + "\n")
    sd_file.write('')  # Add what is being written to sensor_data file here


# TODO: Check if the title output can be shortened
# Initializing files being written to and main loop for recording data
def main():
    # Timestep setup
    timestep = 0.01
    time_dict = {"current_time": 0, "previous_time": 0, "end_time": 100}

    with open(GT_PATH, "w") as ground_truth:
        with open(SD_PATH, "w") as sensor_data:
            headings_gt = ["Position\t", "Velocity\t", "Acceleration\t",
                           "Thrust\t", "Orientation\t"]
            headings_sd = ["Baro_Pressure\t", "Temperature\t",
                           "Acceleration\t", "Angular_Rates\t",
                           "Magnetic Field\t"]
            for (heading_gt, heading_sd) in zip(headings_gt, headings_sd):
                col_titles_gt = heading_gt.split()
                col_titles_sd = heading_sd.split()

                initialize_headings_gt = ' '.join(
                    ['{0: <34}'.format(title) for title in col_titles_gt])
                initialize_headings_sd = ' '.join(
                    ['{0: <16}'.format(title) for title in col_titles_sd])

                ground_truth.write(initialize_headings_gt)
                sensor_data.write(initialize_headings_sd)

            current_rocket = init_rocket_state()
            ground_truth.write("\n")
            sensor_data.write("\n")

            # time_update()
            while time_dict["current_time"] < time_dict["end_time"]:
                time_update(current_rocket, time_dict)
                write_data_to_file(current_rocket, ground_truth, sensor_data)
                time_dict["previous_time"] = time_dict["current_time"]
                time_dict["current_time"] += timestep

        sensor_data.close()
    ground_truth.close()


if __name__ == "__main__":
    main()
