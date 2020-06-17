
import os
import torch
import dateutil.tz
from datetime import datetime
import time
import logging

import numpy as np


def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    genotypes_path = os.path.join(prefix, 'Genotypes')
    os.makedirs(genotypes_path)
    path_dict['genotypes_path'] = genotypes_path

    graph_vis_path = os.path.join(prefix, 'Graph_vis')
    os.makedirs(graph_vis_path)
    path_dict['graph_vis_path'] = graph_vis_path

    return path_dict


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def record(filepath=None, Arch=None,
           best_IS=None, best_IS_epoch=None,
           best_fid=None, best_fid_epoch=None):
    with open(os.path.join(filepath, 'search_line.txt'), 'a') as f:
        f.write('Arch_' + Arch+'_epoch' + ';'+
                'best_IS '+ str(best_IS) + ',' + 'at_' + best_IS_epoch + '_epoch' + ';' +
                'best_fid ' + str(best_fid) + ',' + 'at_' + best_fid_epoch + '_epoch' + '\n')

# draw the search line efficiently
def early_stop(epoch, best_IS, best_fid):
    if epoch == 20:
        if (best_IS < 1.5) and (best_fid > 250):
            return True
    if epoch == 40:
        if (best_IS < 4.7) or (best_fid > 80):
            return True
    if epoch == 60:
        if (best_IS < 7) or (best_fid > 40):
            return True
    if epoch == 80:
        if (best_IS < 7.5) or (best_fid > 30):
            return True
    if epoch == 160:
        if (best_IS < 8) or (best_fid > 20):
            return True

    return False



