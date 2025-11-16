import mujoco
import mujoco.viewer

import time
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import matplotlib.pyplot as plt

with open("cartpole_wheels.xml", "r") as f:
    xml = f.read()

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

positions = []
forces = []

def controller(model, data):
    time_elapsed = data.time
    amplitude = 15.0  # Sinusoidal force amplitude
    frequency = 1.0  # Sinusoidal force frequency
    force = amplitude * np.sin(2 * np.pi * frequency * time_elapsed)
    data.ctrl[0] = force
    forces.append(force)

mujoco.set_mjcb_control(controller)

with mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False) as viewer:
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -30.0
    viewer.cam.distance = 3.0
    viewer.cam.lookat = np.array([0., 0., 0.5])

    N = 5_000
    sim_time = 0
    i = 0
    start = time.time()

    while viewer.is_running() and i < N:
        step_start = time.time()
        mujoco.mj_forward(model, data)
        sim_time += model.opt.timestep
        i += 1
        mujoco.mj_step(model, data)
        positions.append(data.qpos[0])
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

time_axis = np.arange(len(positions)) * model.opt.timestep

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time_axis, positions, label="Cart Position")
plt.title("Cart Position Over Time")
plt.xlabel("Time (s)")
plt.ylabel("X Position (m)")
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_axis, forces[:len(time_axis)], label="Applied Force", color='orange')
plt.title("Applied Force Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
