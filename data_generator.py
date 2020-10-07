"""
This file controls the main functionality of the state estimation
data generation.
"""

import os

import numpy as np

import rocket_math as rm

# -----------------------CONSTANTS---------------------------
# File Paths:
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
GT_PATH = os.path.join(ROOT_PATH, "generated_files", "ground_truth.txt")
SD_PATH = os.path.join(ROOT_PATH, "generated_files", "sensor_data.txt")


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
    for element in array:
        if element < 0:
            return True
    return False


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


def time_update(rocket, time_dict):
    """
    Updates the state of the Rocket for every timestep.

    Parameters
    ----------
    rocket: Rocket
        Rocket object to update.
    time_dict: dict of {str : float}
        Stores the time attributes in the data generation process.
    """
    # Calculate updated state from previous timestep's state
    updated_position = rocket.update_position(time_dict["timestep"])
    updated_velocity = rocket.update_velocity(time_dict["timestep"])
    updated_acceleration = rocket.update_acceleration()
    updated_thrust = rocket.update_thrust(time_dict["current_time"])
    updated_mass = rocket.update_mass(time_dict["timestep"])
    updated_orientation = rocket.update_orientation(rm.ANGULAR_RATES,
                                                    time_dict["timestep"])
    #                                                
    updated_temperature = rocket.update_temperature()
    updated_baro_pressure = rocket.update_baro_pressure()
    updated_body_acceleration = rocket.update_body_acceleration()
    updated_mag_field = rocket.update_magnetic_field()

    # Update the Rocket object
    rocket.position = updated_position
    rocket.velocity = updated_velocity
    rocket.acceleration = updated_acceleration
    rocket.thrust = updated_thrust
    rocket.mass = updated_mass
    rocket.orientation = updated_orientation
    rocket.altitude = rocket.position[2]
    rocket.temperature = updated_temperature
    rocket.baro_pressure = updated_baro_pressure
    rocket.body_mag_field = updated_mag_field
    rocket.body_acceleration = updated_body_acceleration

def write_data_to_file(rocket, gt_file, sd_file):
    """
    Writes the info of the Rocket to the ground_truth and sensor_data files.

    Parameters
    ----------
    rocket: Rocket
        Rocket object with info to write to file.
    gt_file: io.TestIOWrapper
        ground_truth file to write Rocket info to.
    sd_file: io.TestIOWrapper
        sensor_data file to write Rocket info to.
    """
    new_gt_data = np.array(
        [rocket.position, rocket.velocity,
         rocket.body_acceleration, rocket.orientation])
    sensor_data = np.array(
        [str(rocket.baro_pressure), str(rocket.temperature),
        np.array2string(rocket.body_acceleration), np.array2string(rocket.body_mag_field)])
    gt_data = ["", "", "", ""]

    for i, data_elem_gt in enumerate(new_gt_data):
        gt_data[i] = np.array2string(data_elem_gt, precision=4,
                                     floatmode='fixed')
        
    data_to_write = ' '.join(["{0: <33}".format(data) for data in gt_data])
    sensor_data_to_write = ' '.join(["{0: <33}".format(data) for data in sensor_data])
    gt_file.write(data_to_write + "\n")
    sd_file.write(sensor_data_to_write + "\n")


# TODO: Check if the title output can be shortened
def main():
    """
    Main function for generating data based on the input Rocket.
    """
    # Timestep setup
    time_dict = {"current_time": 0, "end_time": 100, "timestep": 0.01}

    with open(GT_PATH, "w") as ground_truth:
        with open(SD_PATH, "w") as sensor_data:
            headings_gt = ["Position\t", "Velocity\t", "Acceleration\t",
                           "Orientation\t"]
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

            # Update state and write data to file
            while time_dict["current_time"] < time_dict["end_time"]:
                time_update(current_rocket, time_dict)
                write_data_to_file(current_rocket, ground_truth, sensor_data)
                time_dict["current_time"] += time_dict["timestep"]

def test():
    rocket = rm.Rocket()
    new_gt_data = np.array(
        [rocket.position, rocket.velocity,
         rocket.body_acceleration, rocket.orientation])
    sensor_data = np.array(
        [str(rocket.baro_pressure), str(rocket.temperature),
        np.array2string(rocket.body_acceleration), np.array2string(rocket.body_mag_field)])
    gt_data = ["", "", "", ""]

    for i, data_elem_gt in enumerate(new_gt_data):
        gt_data[i] = np.array2string(data_elem_gt, precision=4,
                                     floatmode='fixed')
        
    data_to_write = ' '.join(["{0: <33}".format(data) for data in gt_data])
    sensor_data_to_write = ' '.join(["{0: <33}".format(data) for data in sensor_data])
    print(data_to_write)
    print(sensor_data_to_write)

if __name__ == "__main__":
    #main()
    test()
