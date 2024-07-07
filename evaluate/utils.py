import numpy as np
from sklearn.cluster import KMeans
import math
from nesd.utils import sph2cart, get_included_angle
import matplotlib.pyplot as plt


def grouping(part_locss, start_frame):

    # tmp = {}
    all_events = {}
    buffer = {}
    event_id = 0

    for i in range(len(part_locss)):

        curr_locs = part_locss[i]

        for curr_loc in curr_locs:

            curr_loc_pos = sph2cart(azimuth=curr_loc[0], elevation=curr_loc[1], r=1.)

            new_event = True

            for key, data in buffer.items():

                if data["frame_index"][-1] == i - 1:

                    _loc = data["loc"][-1]
                    _loc_pos = sph2cart(azimuth=_loc[0], elevation=_loc[1], r=1.)

                    included_angle = np.rad2deg(get_included_angle(curr_loc_pos, _loc_pos))
                    if included_angle < 10.:
                        buffer[key]["frame_index"].append(start_frame + i)
                        buffer[key]["loc"].append(curr_loc)
                        new_event = False
                        break

            if new_event:
                buffer[event_id] = {"frame_index": [start_frame + i], "loc": [curr_loc]}
                event_id += 1

    return buffer


def get_locss(pred_tensor):

    grid_deg = 2

    frames_num, azi_grids, ele_grids = pred_tensor.shape

    params = []
    for frame_index in range(frames_num):
        param = (frame_index, pred_tensor[frame_index], grid_deg)
        params.append(param)

    # No parallel is faster
    results = []
    for param in params:
        result = _calculate_centers(param)
        results.append(result)

    # with ProcessPoolExecutor(max_workers=None) as pool: # Maximum workers on the machine.
    #     results = pool.map(_calculate_centers, params)

    locss = list(results)

    return locss


def _calculate_centers(param):

    frame_index, x, grid_deg = param

    print(frame_index)

    tmp = np.stack(np.where(x > 0.8), axis=1)

    if len(tmp) == 0:
        return []
    
    for n_clusters in range(1, 10):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(tmp)
        distances = np.linalg.norm(tmp - kmeans.cluster_centers_[kmeans.labels_], axis=-1)
        if np.mean(distances) < 5:
            break
    
    locs = np.deg2rad(kmeans.cluster_centers_ * grid_deg) - np.array([math.pi, math.pi / 2])

    return locs


def _multiple_process_gt_mat(param):

    frame_index, class_index, source_azi, source_ele, azi_grids, ele_grids, grid_deg, half_angle = param
    print(frame_index)

    gt_mat = np.zeros((azi_grids, ele_grids))

    source_direction = np.array(sph2cart(source_azi, source_ele, 1.))

    tmp = []
    azi_grids, ele_grids = gt_mat.shape

    for i in range(azi_grids):
        for j in range(ele_grids):

            _azi = np.deg2rad(i * grid_deg - 180)
            _ele = np.deg2rad(j * grid_deg - 90)

            plot_direction = np.array(sph2cart(_azi, _ele, 1))

            ray_angle = get_included_angle(source_direction, plot_direction)

            if ray_angle < half_angle:
                gt_mat[i, j] = 1
                tmp.append((i, j))

    return frame_index, class_index, gt_mat


def _multiple_process_plot(param):

    frame_index, gt_text, gt_mat, pred_mat, locs, grid_deg, png_path = param
    print("Plot: {}".format(frame_index))

    azi_grids, ele_grids = gt_mat.shape

    if len(locs) > 0:
        centers = locs + np.array([math.pi, math.pi / 2])
        centers = np.rad2deg(centers) / grid_deg
        pred_mat = plot_center_to_mat(centers=locs, x=pred_mat, grid_deg=grid_deg)

    plt.figure(figsize=(20, 10))
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].matshow(gt_mat.T, origin='lower', aspect='equal', cmap='jet', vmin=0, vmax=1)
    axs[1].matshow(pred_mat.T, origin='lower', aspect='equal', cmap='jet', vmin=0, vmax=1)
    for i in range(2):
        axs[i].grid(color='w', linestyle='--', linewidth=0.1)
        axs[i].xaxis.set_ticks(np.arange(0, azi_grids + 1, 10))
        axs[i].yaxis.set_ticks(np.arange(0, ele_grids + 1, 10))
        axs[i].xaxis.set_ticklabels(np.arange(0, 361, 10 * grid_deg), rotation=90)
        axs[i].yaxis.set_ticklabels(np.arange(0, 181, 10 * grid_deg))
    axs[0].set_title(gt_text)

    # Path("_tmp").mkdir(parents=True, exist_ok=True)
    # plt.savefig('_tmp/{:04d}.png'.format(frame_index))
    plt.savefig(png_path)


def plot_center_to_mat(centers, x, grid_deg):
    
    for center in centers:
        center_azi = round(np.rad2deg(center[0] + math.pi) / grid_deg)
        center_ele = round(np.rad2deg(center[1] + math.pi / 2) / grid_deg)
        x[max(center_azi - 5, 0) : min(center_azi + 6, x.shape[0]), center_ele] = np.nan
        x[center_azi, max(center_ele - 5, 0) : min(center_ele + 6, x.shape[1])] = np.nan
    
    return x


def _calculate_centers(param):

    frame_index, x, grid_deg = param

    print(frame_index)

    tmp = np.stack(np.where(x > 0.8), axis=1)

    if len(tmp) == 0:
        return []
    
    for n_clusters in range(1, 10):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(tmp)
        distances = np.linalg.norm(tmp - kmeans.cluster_centers_[kmeans.labels_], axis=-1)
        if np.mean(distances) < 5:
            break
    
    locs = np.deg2rad(kmeans.cluster_centers_ * grid_deg) - np.array([math.pi, math.pi / 2])

    # plt.matshow(x.T, origin='lower', aspect='auto', cmap='jet')
    # plt.savefig("_zz.pdf")
    # from IPython import embed; embed(using=False); os._exit(0)

    return locs