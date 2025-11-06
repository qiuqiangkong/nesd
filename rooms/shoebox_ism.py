import argparse
import random
import time
from pathlib import Path

import h5py
import numpy as np
import pyroomacoustics as pra

from nesd.utils.numpy import (normalize, sample_rotation_matrix,
                              transform_coordinate)
from nesd.utils.utils import (get_eigenmike32, log_uniform,
                              save_list_of_list_to_hdf5)

# Empirical number based on Pyroomacoustics
MAX_IMGS_DICT = {
    0: 1,
    1: 10,
    2: 50,
    3: 100,
    4: 200,
    5: 300
}


def simulate_room_with_image_sources(args) -> None:
    r"""Simulate room with image sources and save to HDF5.
    """

    # Arguments
    mic_csv = args.mic_csv
    data_num = args.data_num
    out_dir = args.out_dir

    env_config = {
        "min_x": 2.,
        "max_x": 10.,
        "min_y": 2.,
        "max_y": 10.,
        "min_z": 2.,
        "max_z": 4.,
    }

    # Build room
    room = ShoeboxISM(env_config, mic_csv)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for n in range(data_num):
        
        t0 = time.time()

        # Sample an environment and compute image sources
        data = room.sample()

        # Save out to HDF5
        out_path = Path(out_dir, f"{n:06d}.h5")
        room.save_to_hdf5(data, out_path)
        print("{}/{} Write out to {} Time: {:.4f} s".format(
            n, data_num, out_path, time.time() - t0)
        )


class ShoeboxISM:
    def __init__(
        self, 
        env_config: dict,
        mic_csv: str, 
        min_srcs=1, 
        max_srcs=5, 
        max_ism_order=3, 
        min_ism_order=3
    ):
        self.env_config = env_config
        self.mic_local_pos = get_eigenmike32(mic_csv)
        self.mic_local_dir = normalize(self.mic_local_pos)  # Direction
        self.mics_num = len(self.mic_local_pos)

        self.min_srcs = min_srcs
        self.max_srcs = max_srcs
        self.max_ism_order = max_ism_order
        self.min_ism_order = min_ism_order
        self.max_imgs = MAX_IMGS_DICT[max_ism_order]

        self.dtype = np.float32
        
    def sample(self) -> dict:
        r"""Sample a room, microphone positions and directions, source 
        positions. Compute image sources.

        1. Sample environment
        2. Sample microphone positions and directions
        3. Sample source positions and directions
        4. Sample listener positions
        5. Compute image sources

        m: mics_num
        s: srcs_num
        l: lis_num
        i: image_sources_num

        Args:
            None

        Returns:
            data (dict)
        """
        
        # --- 1. Sample environment ---
        env = Shoebox(**self.env_config)

        # --- 2. Sample microphone positions and directions ---
        # World coordinate system
        world_origin = np.zeros(3)  # World origin
        world_R = np.eye(3)  # World rotation matrix

        # Local coordinate system of the microphone array
        local_origin = env.sample_position()  # Random position
        local_R = sample_rotation_matrix()  # Random roatation matrix
        
        # Convert microphone positions from local to world coordinates
        mic_world_pos = transform_coordinate(
            x=self.mic_local_pos, 
            origin_from=local_origin,
            origin_to=world_origin,
            R_from=local_R,
            R_to=world_R
        )  # (m, 3)
        mic_pos = mic_world_pos

        # Convert mic local direction to world direction
        mic_dir = normalize(transform_coordinate(
            x=self.mic_local_dir,
            origin_from=np.zeros(3),
            origin_to=np.zeros(3),
            R_from=local_R,
            R_to=world_R
        ))  # (m, 3)

        # --- 3. Sample source positions and directions ---
        srcs_num = random.randint(self.min_srcs, self.max_srcs)
        if srcs_num == 0:
            src_pos = np.zeros((0, 3))
        else:
            src_pos = np.stack([sample_noncolliding_position(env, mic_pos) for _ in range(srcs_num)])  # (s, 3)
        
        # --- 4. Sample listeners ---
        # Place the listener at the center of microphones
        lis_pos = np.mean(mic_pos, axis=0)[None, :]  # (l, 3) 
        lis_num = len(lis_pos)
        
        # --- 5. Compute image sources ---
        ism_order = random.randint(self.min_ism_order, self.max_ism_order)

        # Build room
        room = pra.Room.from_corners(
            corners=env.corners.T,
            max_order=ism_order,
        )
        room.extrude(height=env.height)

        # Add microphone to the room.
        receiver_pos = np.concatenate((mic_pos, lis_pos), axis=0)  # (m+l, 3)
        room.add_microphone(receiver_pos.T)

        # Add sources to the room.
        safe_add_sources(room, src_pos)

        # Render image sources.
        room.image_source_model()

        mics_num = self.mics_num
        mic_imgs = [[None] * mics_num for _ in range(srcs_num)]  # (s, m, i, 3)
        mic_orders = [[None] * mics_num for _ in range(srcs_num)]  # (s, m, i)
        lis_imgs = [[None] * lis_num for _ in range(srcs_num)]  # (s, l, i, 3)
        lis_orders = [[None] * lis_num for _ in range(srcs_num)]  # (s, l, i)

        for s in range(srcs_num):
            for m in range(mics_num):
                idx = room.visibility[s][m]  # (i,)
                mic_imgs[s][m] = room.sources[s].images.T[idx]  # (i, 3)
                mic_orders[s][m] = room.sources[s].orders[idx]  # (i,)

            for l in range(lis_num):
                idx = room.visibility[s][l]  # (i,)
                lis_imgs[s][l] = room.sources[s].images.T[idx]  # (i, 3)
                lis_orders[s][l] = room.sources[s].orders[idx]  # (i,)

        data = {
            "src_pos": src_pos.astype(self.dtype),  # (s, 3)
            "srcs_num": srcs_num,  # scalar
            "mic_pos": mic_pos.astype(self.dtype),  # (m, 3)
            "mic_dir": mic_dir.astype(self.dtype),  # (m, 3)
            "mic_img": mic_imgs,  # list of list, (s, m, i, 3)
            "mic_order": mic_orders,  # list of list, (s, m, i)
            "mic_rot_mat": local_R.astype(self.dtype),  # (3, 3)
            "mics_num": mics_num,  # scalar
            "lis_pos": lis_pos.astype(self.dtype),  # (l, 3)
            "lis_img": lis_imgs,  # list of list, (s, l, i, 3)
            "lis_order": lis_orders,  # list of list, (s, l, i)
            "lis_rot_mat": local_R.astype(self.dtype),  # (3, 3)
            "lis_num": lis_num,  # scalar
            "max_srcs": self.max_srcs,  # scalar
            "max_imgs": self.max_imgs  # scalar
        }

        return data

    def save_to_hdf5(self, data: dict, out_path: str) -> None:
        r"""Save data to HDF5."""
        with h5py.File(out_path, 'w') as hf:

            hf.create_dataset("src_pos", data=data["src_pos"], dtype=np.float32)
            hf.create_dataset("mic_pos", data=data["mic_pos"], dtype=np.float32)
            hf.create_dataset("mic_dir", data=data["mic_dir"], dtype=np.float32)
            hf.create_dataset("mic_rot_mat", data=data["mic_rot_mat"], dtype=np.float32)
            hf.create_dataset("lis_pos", data=data["lis_pos"], dtype=np.float32)
            hf.create_dataset("lis_rot_mat", data=data["lis_rot_mat"], dtype=np.float32)

            hf.attrs.create("srcs_num", data=data["srcs_num"], dtype=np.int32)
            hf.attrs.create("mics_num", data=data["mics_num"], dtype=np.int32)
            hf.attrs.create("lis_num", data=data["lis_num"], dtype=np.int32)
            hf.attrs.create("max_srcs", data=data["max_srcs"], dtype=np.int32)
            hf.attrs.create("max_imgs", data=data["max_imgs"], dtype=np.int32)

            save_list_of_list_to_hdf5(hf, name="mic_img", data=data["mic_img"], dtype=np.float32)
            save_list_of_list_to_hdf5(hf, name="mic_order", data=data["mic_order"], dtype=np.int32)
            save_list_of_list_to_hdf5(hf, name="lis_img", data=data["lis_img"], dtype=np.float32)
            save_list_of_list_to_hdf5(hf, name="lis_order", data=data["lis_order"], dtype=np.int32)


class Shoebox:
    def __init__(
        self, 
        min_x=2.,  # Minimum room length
        max_x=10.,  # Maximum room length
        min_y=2.,  # Minimum room width
        max_y=10.,  # Maximum room width
        min_z=2.,  # Minimum room height
        max_z=4.  # Maximum room height
    ):
        r"""Shoebox room."""

        # x: length, y: width, z: height. Sample room length, width, and height.
        self.x_bnd = log_uniform(min_x, max_x)
        self.y_bnd = log_uniform(min_y, max_y)
        self.z_bnd = log_uniform(min_z, min(self.x_bnd, self.y_bnd, max_z))

        self.corners = np.array([
            [0, 0], 
            [0, self.y_bnd], 
            [self.x_bnd, self.y_bnd], 
            [self.x_bnd, 0]
        ])

        self.height = self.z_bnd

    def sample_position(self, margin=0.2) -> np.ndarray:
        r"""Randomly sample a position inside the room."""
        x = random.uniform(margin, self.x_bnd - margin)
        y = random.uniform(margin, self.y_bnd - margin)
        z = random.uniform(margin, self.z_bnd - margin)
        pos = np.array([x, y, z])

        return pos


def sample_noncolliding_position(env: object, poses: np.ndarray) -> np.ndarray:
    r"""Sample a position until not collide with existing positions"""
    while True:
        pos = env.sample_position()
        if no_colliding(pos, poses):
            return pos


def no_colliding(pos: np.ndarray, poses: np.ndarray, radius=1.0) -> bool:
    r"""Return True if pos does not collide with existing positions."""
    for p in poses:
        if np.linalg.norm(pos - p) < radius:
            return False
    return True


def safe_add_sources(room, poses: np.ndarray):
    r"""Retry adding sources in a loop to handle occasional failures."""
    for pos in poses:
        while True:
            try:
                room.add_source(pos)
                break
            except:
                continue


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mic_csv', type=str, required=True)
    parser.add_argument('--data_num', type=int, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    
    args = parser.parse_args()
    simulate_room_with_image_sources(args)
