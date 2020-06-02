# This file controls the main functionality of the state estimation data generation

import numpy as np
import rocket_math as rm

'''
Simulator inputs:
    Rate/timestep
    Rocket parameters: mass, average thrust, burn time. If we decide to implement a more sophisticated or flexible rocket model, we can add more parameters here, but I think this is the minimum we'd want and good enough to start with.
    Sensor models: noise, sensitivity, bias, hysteresis, etc. Any information from that datasheet that we need to account for when transforming our ground truth data into sensor measurements. Again, we can start off simple here (noise only) and add more parameters as we go.
    Other parameters: launch tower angle, launch site altitude (we need this for later pressure calculations), local magnetic field data. These are the first ones I thought of, but there may be more later.
Simulator outputs:
    A "ground truth" file with ground truth simulated data (no sensor models) for each timestep. To start, we definitely want the following simulated:
        Absolute position (ENU)
        Velocity
        Acceleration
        Orientation (another post to follow with regards to rotation representations)
    A "sensor data" file with the simulated sensor measurements for each timestep. Right now, we need:
        Barometric pressure
        Temperature
        Acceleration (x, y, z)
        Angular rates (x, y, z)
        Magnetic field (B field) components (x, y, z)
Implementation:
    We can likely make the following assumptions to make initial implementation easier:
        The rocket's yaw rate is constant and zero.
        The rocket's pitch rate is constant and relatively small (~1 degree/s).
        The angle of attack is zero.
    At each timestep, the simulator should do the following:
        Recall the previous state of the rocket: position, orientation, velocity, angular velocity.
        Determine the forces acting on the rocket: gravity, thrust (if the engine is still burning), drag.
        Determine the acceleration of the rocket (F = ma).
        Update the rocket state:
            Integrate velocity to find the new position of the rocket.
            Integrate acceleration to find the new velocity of the rocket.
            Update the orientation of the rocket (details will depend on how we choose to represent orientation).
        Output the rocket state to the ground truth file.
        Convert ground truth rocket state to ground truth sensor measurements:
            From rocket and launch site altitude, determine barometric pressure and temperature.
            From orientation and known pitch/roll rates, determine measured angular velocities.
            From orientation and known absolute acceleration, determine accelerometer measurements.
            From orientation and local magnetic field data, determine magnetometer measurements.
        Add sensor errors: noise, bias, etc.
        Output sensor data to the sensor data file.
'''

# The rocket's fixed angular velocities
yaw_rate = 0
pitch_rate = 1
roll_rate = 180  # in deg/s

# Launch site parameters
tower_angle = 90
launch_site_altitude = 0
local_magnetic_field = 0  # TODO: figure out what we need to actually store here

# Timestep setup
timestep = 0.01
end_time = 20
current_time = 0
previous_time = 0


# Initializing the current rocket
current_rocket = rm.Rocket(0, np.array([0.0, 0.0, 0.0]), 0, 0, 0, 0, 0, 0, np.array([0.0, 0.0, 0.0]),
                           np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]),
                           np.array([0.0, 0.0, 0.0]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


# TODO: add exception blocks
# Getting initial params for rocket call
def init_rocket_state(initial_rocket: rm.Rocket) -> rm.Rocket:
    # try:
    mass, thrust, burn_time = input("Enter a mass, thrust, and burn time: ").split()
    # except:

    # try:
    pressure_noise, temp_noise, accel_noise, gyro_noise, mag_noise = input(
            "Enter noise params (pressure, temperature, "
            "acceleration, gyro, and magnetic noise):"
            " ").split()
    # except:

    initial_rocket.mass = float(mass)
    initial_rocket.thrust[2] = thrust
    initial_rocket.burn_time = float(burn_time)
    initial_rocket.pressure_noise = pressure_noise
    initial_rocket.temp_noise = temp_noise
    initial_rocket.accel_noise = accel_noise
    initial_rocket.gyro_noise = gyro_noise
    initial_rocket.mag_noise = mag_noise
    return initial_rocket


# Updates the state of the rocket to the ground_truth and sensor_data files
# TODO: add writing to sensor_data files
def time_update():
    # TODO: consider making previous_time and current_time 1 variable
    current_rocket.vel_unit_vec = rm.get_vel_unit_vector(current_rocket, tower_angle)
    current_rocket.thrust = rm.get_thrust(current_rocket, current_time)
    current_rocket.position_loc_cart = rm.rocket_position_local_cartesian(current_rocket, previous_time, current_time)
    current_rocket.position_enu = rm.loc_cart_to_enu_2(current_rocket, previous_time, current_time)  # position should calculated first
    current_rocket.velocity = rm.rocket_velocity(current_rocket, previous_time, current_time)
    current_rocket.acceleration = rm.rocket_acceleration(current_rocket, previous_time, current_time)
    current_rocket.altitude = current_rocket.position_loc_cart[2]
    current_rocket.mass = rm.get_mass(current_rocket)

    new_data_gt = np.array([current_rocket.position_loc_cart, current_rocket.velocity, current_rocket.acceleration])  # position is in local cartesian for testing purposes
    data_gt = ["", "", ""]
    for i, data_elem_gt in enumerate(new_data_gt):
        data_gt[i] = np.array2string(data_elem_gt, precision=3)
    data_to_write = ' '.join(["{0: <33}".format(data) for data in data_gt])
    ground_truth.write(data_to_write + "\n")
    sensor_data.write('')  # Add what is being written to sensor_data file here


# TODO: Check if the title output can be shortened
# Initializing files being writen to and main loop for recording data
with open("ground_truth.txt", "w") as ground_truth:
    with open("sensor_data.txt", "w") as sensor_data:
        headings_gt = ["Position_ENU\t", "Velocity\t", "Acceleration\t", "Orientation\t"]
        headings_sd = ["Baro_Pressure\t", "Temperature\t", "Acceleration\t", "Angular_Rates\t", "Magnetic Field\t"]
        for (heading_gt, heading_sd) in zip(headings_gt, headings_sd):
            col_titles_gt = heading_gt.split()
            col_titles_sd = heading_sd.split()

            initialize_headings_gt = ' '.join(['{0: <34}'.format(title) for title in col_titles_gt])
            initialize_headings_sd = ' '.join(['{0: <16}'.format(title) for title in col_titles_sd])

            ground_truth.write(initialize_headings_gt)
            sensor_data.write(initialize_headings_sd)

        current_rocket = init_rocket_state(current_rocket)
        ground_truth.write("\n")
        sensor_data.write("\n")

        # print(type(current_rocket.burn_time))
        print("This is burn_time: " + str(current_rocket.burn_time))
        print("This is thrust: " + str(current_rocket.thrust))

        # time_update()
        while current_time < end_time:
            time_update()
            previous_time = current_time
            current_time += timestep
            # print(current_rocket.__dict__) --> this is just for debugging purposes

    sensor_data.close()
ground_truth.close()

