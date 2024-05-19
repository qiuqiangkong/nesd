from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import evaluate.SELD_evaluation_metrics_2019 as SELDMetrics2019
from evaluate.SELD_data_utilities import (load_dcase_format,
                                          to_metrics2020_format)
from evaluate.SELD_evaluation_metrics_2020 import \
    SELDMetrics as SELDMetrics2020
from evaluate.SELD_evaluation_metrics_2020 import early_stopping_metric
# from evaluate.dataset_creation.pack_audios_to_hdf5s.dcase2021_task3 import LABELS


class AA:
    def __init__(self):
        self.clip_length = None
        self.label_resolution = None


def evaluate():

    """ Evaluate scores
    """

    '''Directories'''
    # print('Inference ID is {}\n'.format(cfg['inference']['infer_id']))

    cfg = {
        'dataset_dir': "/home/tiger/datasets/dcase2021/task3",
        'inference': {'testset_type': 'dev'},
    }

    dataset = AA()
    # from IPython import embed; embed(using=False); os._exit(0)
    # dataset.clip_length = 60.0
    # dataset.label_resolution = 0.1
    # dataset.label_set = [
    #     "alarm",
    #     "crying baby",
    #     "crash",
    #     "barking dog",
    #     "female scream",
    #     "female speech",
    #     "footsteps",
    #     "knocking on door",
    #     "male scream",
    #     "male speech",
    #     "ringing phone",
    #     "piano",
    # ]

    dataset.clip_length = 60.0
    dataset.label_resolution = 0.02
    dataset.label_set = [
        "clearthroat", 
        "cough", 
        "doorslam", 
        "drawer", 
        "keyboard", 
        "keysDrop", 
        "knock", 
        "laughter", 
        "pageturn", 
        "phone", 
        "speech"
    ]

    split = "test"

    if split == "train":
        submissions_dir = Path("./submissions/01_train")
        meta_dir = Path("/home/tiger/datasets/dcase2021/task3/metadata_dev/dev-train")

    elif split == "test":
        # submissions_dir = "/home/qiuqiangkong/workspaces/nesd/results/dcase2019_task3/pred_csvs"
        # meta_dir = "/datasets/dcase2019/task3/metadata_eval"

        submissions_dir = "/home/qiuqiangkong/workspaces/nesd/results/dcase2019_task3/pred_csvs_1"
        meta_dir = "/datasets/dcase2019/task3/metadata_eval"
        asdf

    # submissions_dir = Path("/home/tiger/datasets/dcase2021/task3/metadata_dev/dev-test")
    
    
    # meta_dir = Path("/home/tiger/datasets/dcase2021/task3/metadata_dev/dev-test")

    # out_infer_dir = Path(cfg['workspace_dir']).joinpath('out_infer').joinpath(cfg['method']) \
    #     .joinpath(cfg['inference']['infer_id'])
    # submissions_dir = out_infer_dir.joinpath('submissions')

    # main_dir = Path(cfg['dataset_dir'])
    # main_dir = "/home/tiger/datasets/dcase2021/task3"

    # dev_meta_dir = main_dir.joinpath('metadata_dev')
    # eval_meta_dir = main_dir.joinpath('metadata_eval')   
    # if cfg['inference']['testset_type'] == 'dev':
    #     meta_dir = dev_meta_dir
    # elif cfg['inference']['testset_type'] == 'eval':
    #     meta_dir = eval_meta_dir

    
    
    pred_frame_begin_index = 0
    gt_frame_begin_index = 0
    frame_length = int(dataset.clip_length / dataset.label_resolution) + 1
    pred_output_dict, pred_sed_metrics2019, pred_doa_metrics2019 = {}, [], []
    gt_output_dict, gt_sed_metrics2019, gt_doa_metrics2019= {}, [], []
    # from IPython import embed; embed(using=False); os._exit(0)
    
    for pred_path in sorted(Path(submissions_dir).glob('*.csv')):
        fn = pred_path.name
        gt_path = Path(meta_dir).joinpath(fn)
        
        '''
        # pred
        output_dict, sed_metrics2019, doa_metrics2019 = load_dcase_format(
            pred_path, frame_begin_index=pred_frame_begin_index, 
            frame_length=frame_length, num_classes=len(dataset.label_set), set_type='pred')
        pred_output_dict.update(output_dict)
        pred_sed_metrics2019.append(sed_metrics2019)
        pred_doa_metrics2019.append(doa_metrics2019)
        pred_frame_begin_index += frame_length
        '''

        # gt
        output_dict, sed_metrics2019, doa_metrics2019 = load_dcase_format(
            gt_path, frame_begin_index=gt_frame_begin_index, 
            frame_length=frame_length, num_classes=len(dataset.label_set), set_type='gt')
        gt_output_dict.update(output_dict)
        gt_sed_metrics2019.append(sed_metrics2019)
        gt_doa_metrics2019.append(doa_metrics2019)
        gt_frame_begin_index += frame_length

    pred_sed_metrics2019 = np.concatenate(pred_sed_metrics2019, axis=0)
    pred_doa_metrics2019 = np.concatenate(pred_doa_metrics2019, axis=0)
    gt_sed_metrics2019 = np.concatenate(gt_sed_metrics2019, axis=0)
    gt_doa_metrics2019 = np.concatenate(gt_doa_metrics2019, axis=0)
    pred_metrics2020_dict = to_metrics2020_format(pred_output_dict, 
        pred_sed_metrics2019.shape[0], label_resolution=dataset.label_resolution)
    gt_metrics2020_dict = to_metrics2020_format(gt_output_dict, 
        gt_sed_metrics2019.shape[0], label_resolution=dataset.label_resolution)    
    
    if True:
        pred_sed_metrics2019
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].matshow(gt_sed_metrics2019.T, origin='lower', aspect='auto', cmap='jet')
        axs[1].matshow(pred_sed_metrics2019.T, origin='lower', aspect='auto', cmap='jet')
        # axs[0].xaxis.set_ticks(np.arange(0, 101, 10))
        # axs[0].xaxis.set_ticklabels(np.arange(11))
        classes_num = len(LABELS)
        for i in range(2):
            axs[i].yaxis.set_ticks(np.arange(classes_num))
            axs[i].yaxis.set_ticklabels([lb[0:10] for lb in LABELS], fontsize=8)
        axs[0].xaxis.grid(color='k', linestyle='solid', linewidth=1)
        axs[1].xaxis.grid(color='w', linestyle='solid', linewidth=1)
        axs[1].yaxis.grid(color='w', linestyle='solid', linewidth=0.2)
        plt.savefig('_zz.pdf')
        # from IPython import embed; embed(using=False); os._exit(0)

    # 2019 metrics
    num_frames_1s = int(1 / dataset.label_resolution)
    ER_19, F_19 = SELDMetrics2019.compute_sed_scores(pred_sed_metrics2019, gt_sed_metrics2019, num_frames_1s)
    LE_19, LR_19, _, _, _, _ = SELDMetrics2019.compute_doa_scores_regr(
        pred_doa_metrics2019, gt_doa_metrics2019, pred_sed_metrics2019, gt_sed_metrics2019)
    seld_score_19 = SELDMetrics2019.early_stopping_metric([ER_19, F_19], [LE_19, LR_19])

    # 2020 metrics
    dcase2020_metric = SELDMetrics2020(nb_classes=len(dataset.label_set), doa_threshold=20)
    dcase2020_metric.update_seld_scores(pred_metrics2020_dict, gt_metrics2020_dict)
    ER_20, F_20, LE_20, LR_20 = dcase2020_metric.compute_seld_scores()
    seld_score_20 = early_stopping_metric([ER_20, F_20], [LE_20, LR_20])

    metrics_scores ={
        'ER20': ER_20,
        'F20': F_20,
        'LE20': LE_20,
        'LR20': LR_20,
        'seld20': seld_score_20,
        'ER19': ER_19,
        'F19': F_19,
        'LE19': LE_19,
        'LR19': LR_19,
        'seld19': seld_score_19,
    }

    out_str = 'test: '
    for key, value in metrics_scores.items():
        out_str += '{}: {:.3f},  '.format(key, value)
    print('---------------------------------------------------------------------------------------------------'
        +'-------------------------------------------------')
    print(out_str)
    print('---------------------------------------------------------------------------------------------------'
        +'-------------------------------------------------')

    from IPython import embed; embed(using=False); os._exit(0)

    # out_eval_dir = Path(cfg['workspace_dir']).joinpath('out_eval').joinpath(cfg['method']) \
    #     .joinpath(cfg['inference']['infer_id'])
    # out_eval_dir.mkdir(parents=True, exist_ok=True)
    # result_path = out_eval_dir.joinpath('results.csv')
    # df = pd.DataFrame(metrics_scores, index=[0])
    # df.to_csv(result_path, sep=',', mode='a')

evaluate()
