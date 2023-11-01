import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def add():

    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((1, 1), 2, 6, fill=None))
    ax.plot(3, 4, 'ro')
    ax.arrow(x=1, y=2, dx=0.5, dy=0.5, head_width=0.05, color="r")
    plt.savefig("_zz.pdf")


def add2():

    room_length = 6
    room_width = 5
    room_height = 4
    source_positions = np.array(([[3,2,1], [4,3,3]]))
    mic_positions = np.array(([[1, 4, 3], [1, 4.2, 3.2]]))
    agent_positions = np.array(([[2,3,2], [2,3,2]]))
    agent_look_directions = np.array(([[0.3, 0.3, 0.3], [0.3, -0.3, -0.3]]))
    agent_ray_types = ["positive", "negative"]

    plot_top_view(room_length, room_width, room_height, source_positions, mic_positions, agent_positions, agent_look_directions, agent_ray_types)


def plot_top_view(room_length, room_width, room_height, source_positions, mic_positions, agent_positions, agent_look_directions, agent_ray_types):

    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")

    ax.add_patch(Rectangle((0, 0), room_length, room_width, fill=None))
    
    for position in source_positions:
        line_source, = ax.plot(position[0], position[1], c="grey", marker="o", label="Sources")

    for position in mic_positions:
        line_mic, = ax.plot(position[0], position[1], c="blue", marker="+", label="Microphones")

    for position, look_direction, ray_type in zip(agent_positions, agent_look_directions, agent_ray_types):
        line_agent, = ax.plot(position[0], position[1], c="red", marker="o", label="Agents")

        if ray_type == "positive":
            color = "red"
            ax.arrow(x=position[0], y=position[1], dx=look_direction[0], dy=look_direction[1], head_width=0.05, color=color)

        elif ray_type == "negative":
            color = "black"
            ax.arrow(x=position[0], y=position[1], dx=look_direction[0], dy=look_direction[1], head_width=0.05, color=color)
        

    plt.legend(handles=[line_source, line_mic, line_agent], fontsize='small', loc=1)

    # ax.plot(3, 4, 'ro')
    # ax.arrow(x=1, y=2, dx=0.5, dy=0.5, head_width=0.05, color="r")
    plt.savefig("_zz.pdf")
    from IPython import embed; embed(using=False); os._exit(0)



if __name__ == "__main__":
    add2()