# This file controls the main functionality of the state estimation data generation

import numpy as np
import rocket_math as rm
import os


# -----------------------CONSTANTS---------------------------
# File Paths:
ROOT_PATH = os.path.dirname(os.path.abspath("requirements.txt"))
GT_PATH = os.path.join(ROOT_PATH, "generated_files", "ground_truth.txt")
SD_PATH = os.path.join(ROOT_PATH, "generated_files", "sensor_data.txt")
# -----------------------------------------------------------


# TODO: add exception blocks
# Getting initial params for rocket call
def init_rocket_state() -> rm.Rocket:
    # Initializing the current rocket
    current_rocket = rm.Rocket({"total_mass": 0.0, "body_mass": 100.0, "comb_mass": 0.0}, np.array([0.0, 0.0, 0.0]), 0,
                               {"pressure_noise": 0.0, "temp_noise": 0.0, "accel_noise": 0.0, "mag_noise": 0.0})
    # try:
    total_mass, thrust, burn_time = input("Enter a total mass, thrust, and burn time: ").split()
    # except:

    # try:
    pressure_noise, temp_noise, accel_noise, mag_noise = input(
            "Enter noise params (pressure, temperature, "
            "acceleration, gyro, and magnetic noise):"
            " ").split()
    # except:

    current_rocket.mass["total_mass"] = total_mass
    current_rocket.mass["comb_mass"] = total_mass - current_rocket.mass["body_mass"]
    current_rocket.thrust[2] = thrust
    current_rocket.burn_time = float(burn_time)
    current_rocket.sensor_noise["pressure_noise"] = pressure_noise
    current_rocket.sensor_noise["temp_noise"] = temp_noise
    current_rocket.sensor_noise["accel_noise"] = accel_noise
    current_rocket.sensor_noise["mag_noise"] = mag_noise
    return current_rocket


# Updates the state of the rocket to the ground_truth and sensor_data files
# TODO: add writing to sensor_data files
def time_update(rocket: rm.Rocket, gt_file, sd_file, time_dict):
    # TODO: consider making previous_time and current_time 1 variable
    rocket.thrust = rm.get_thrust(rocket, time_dict["current_time"])
    rocket.position_loc_cart = rm.rocket_position_local_cartesian(rocket, time_dict["previous_time"], time_dict["current_time"])
    rocket.position_enu = rm.loc_cart_to_enu_2(rocket, time_dict["previous_time"], time_dict["current_time"])  # position should calculated first
    rocket.velocity = rm.rocket_velocity(rocket, time_dict["previous_time"], time_dict["current_time"])
    rocket.acceleration = rm.rocket_acceleration(rocket, time_dict["previous_time"], time_dict["current_time"])
    rocket.altitude = rocket.position_loc_cart[2]
    rocket.mass = rm.get_mass(rocket)

    new_data_gt = np.array([rocket.position_loc_cart, rocket.velocity, rocket.acceleration])  # position is in local cartesian for testing purposes
    data_gt = ["", "", ""]
    for i, data_elem_gt in enumerate(new_data_gt):
        data_gt[i] = np.array2string(data_elem_gt, precision=3)
    data_to_write = ' '.join(["{0: <33}".format(data) for data in data_gt])
    gt_file.write(data_to_write + "\n")
    sd_file.write('')  # Add what is being written to sensor_data file here


# TODO: Check if the title output can be shortened
# Initializing files being writen to and main loop for recording data
def main():
    # Timestep setup
    timestep = 0.01
    time_dict = {"current_time": 0, "previous_time": 0, "end_time": 0}

    with open(GT_PATH, "w") as ground_truth:
        with open(SD_PATH, "w") as sensor_data:
            headings_gt = ["Position_ENU\t", "Velocity\t", "Acceleration\t", "Orientation\t"]
            headings_sd = ["Baro_Pressure\t", "Temperature\t", "Acceleration\t", "Angular_Rates\t", "Magnetic Field\t"]
            for (heading_gt, heading_sd) in zip(headings_gt, headings_sd):
                col_titles_gt = heading_gt.split()
                col_titles_sd = heading_sd.split()

                initialize_headings_gt = ' '.join(['{0: <34}'.format(title) for title in col_titles_gt])
                initialize_headings_sd = ' '.join(['{0: <16}'.format(title) for title in col_titles_sd])

                ground_truth.write(initialize_headings_gt)
                sensor_data.write(initialize_headings_sd)

            current_rocket = init_rocket_state()
            ground_truth.write("\n")
            sensor_data.write("\n")

            # print(type(current_rocket.burn_time))
            print("This is burn_time: " + str(current_rocket.burn_time))
            print("This is thrust: " + str(current_rocket.thrust))

            # time_update()
            while time_dict["current_time"] < time_dict["end_time"]:
                time_update(current_rocket, ground_truth, sensor_data, time_dict)
                time_dict["previous_time"] = time_dict["current_time"]
                time_dict["current_time"] += timestep
                # print(current_rocket.__dict__) --> this is just for debugging purposes

        sensor_data.close()
    ground_truth.close()

