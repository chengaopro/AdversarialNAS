# @Date    : 2019-10-22
# @Author  : Chen Gao

from __future__ import absolute_import, division, print_function


import cfg
import archs
import datasets
from network import train, validate, LinearLrDecay, load_params, copy_params
from utils.utils import set_log_dir, save_checkpoint, create_logger, count_parameters_in_MB
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from utils.genotype import alpha2genotype, beta2genotype, draw_graph_G, draw_graph_D

import torch
import os
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
from architect import Architect_gen, Architect_dis
from utils.flop_benchmark import print_FLOPs

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)

    # set visible GPU ids
    if len(args.gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # set TensorFlow environment for evaluation (calculate IS and FID)
    _init_inception()
    inception_path = check_or_download_inception('./tmp/imagenet/')
    create_inception_graph(inception_path)

    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for _id in range(len(str_ids)):
        if _id >= 0:
            args.gpu_ids.append(_id)
    if len(args.gpu_ids) > 1:
        args.gpu_ids = args.gpu_ids[1:]
    else:
        args.gpu_ids = args.gpu_ids

    # import network
    basemodel_gen = eval('archs.' + args.arch + '.Generator')(args=args)
    gen_net = torch.nn.DataParallel(basemodel_gen, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])

    basemodel_dis = eval('archs.' + args.arch + '.Discriminator')(args=args)
    dis_net = torch.nn.DataParallel(basemodel_dis, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])

    architect_gen = Architect_gen(gen_net, args)
    architect_dis = Architect_dis(dis_net, args)

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    gen_net.apply(weights_init)
    dis_net.apply(weights_init)

    # set optimizer
    arch_params_gen = gen_net.module.arch_parameters()
    arch_params_gen_ids = list(map(id, arch_params_gen))
    weight_params_gen = filter(lambda p: id(p) not in arch_params_gen_ids, gen_net.parameters())
    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, weight_params_gen),
                                     args.g_lr, (args.beta1, args.beta2))

    arch_params_dis = dis_net.module.arch_parameters()
    arch_params_dis_ids = list(map(id, arch_params_dis))
    weight_params_dis = filter(lambda p: id(p) not in arch_params_dis_ids, dis_net.parameters())
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, weight_params_dis),
                                     args.d_lr, (args.beta1, args.beta2))

    # set up data_loader
    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train

    # epoch number for dis_net
    args.max_epoch_D = args.max_epoch_G * args.n_critic
    if args.max_iter_G:
        args.max_epoch_D = np.ceil(args.max_iter_G * args.n_critic / len(train_loader))
    args.max_iter_D = args.max_epoch_D * len(train_loader)

    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter_D)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter_D)

    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))
    gen_avg_param = copy_params(gen_net)
    start_epoch = 0
    # best_fid = 1e4

    # set writer
    if args.checkpoint:
        # resuming
        print(f'=> resuming from {args.checkpoint}')
        assert os.path.exists(os.path.join('exps', args.checkpoint))
        checkpoint_file = os.path.join('exps', args.checkpoint, 'Model', 'checkpoint_best.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        avg_gen_net = deepcopy(gen_net)
        avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(avg_gen_net)
        del avg_gen_net

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
    else:
        # create new log dir
        assert args.exp_name
        args.path_helper = set_log_dir('exps', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    logger.info("param size of G = %fMB", count_parameters_in_MB(gen_net))
    logger.info("param size of D = %fMB", count_parameters_in_MB(dis_net))

    # search loop
    for epoch in tqdm(range(int(start_epoch), int(args.max_epoch_D)), desc='total progress'):
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        tau_decay = np.log(args.tau_max / args.tau_min) / args.max_epoch_D if args.gumbel_softmax else None
        tau = max(0.1, args.tau_max * np.exp(-tau_decay * epoch)) if args.gumbel_softmax else None
        if tau:
            gen_net.module.set_tau(tau)
            dis_net.module.set_tau(tau)

        # search arch and train weights
        if epoch > 0:
            train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict,
                  lr_schedulers, architect_gen=architect_gen, architect_dis=architect_dis)

        # save and visualise current searched arch
        if epoch == 0 or epoch % args.derive_freq == 0 or epoch == int(args.max_epoch_D) - 1:
            genotype_G = alpha2genotype(gen_net.module.alphas_normal, gen_net.module.alphas_up, save=True,
                                        file_path=os.path.join(args.path_helper['genotypes_path'], str(epoch) + '_G.npy'))
            genotype_D = beta2genotype(dis_net.module.alphas_normal, dis_net.module.alphas_down, save=True,
                                       file_path=os.path.join(args.path_helper['genotypes_path'], str(epoch) + '_D.npy'))
            if args.draw_arch:
                draw_graph_G(genotype_G, save=True, file_path=os.path.join(args.path_helper['graph_vis_path'], str(epoch) + '_G'))
                draw_graph_D(genotype_D, save=True, file_path=os.path.join(args.path_helper['graph_vis_path'], str(epoch) + '_D'))

        # validate current searched arch
        if epoch == 0 or epoch % args.val_freq == 0 or epoch == int(args.max_epoch_D) - 1:
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param)

            inception_score, std, fid_score = validate(args, fixed_z, fid_stat, gen_net, writer_dict)
            logger.info(f'Inception score mean: {inception_score}, Inception score std: {std}, '
                        f'FID score: {fid_score} || @ epoch {epoch}.')

        avg_gen_net = deepcopy(gen_net)
        load_params(avg_gen_net, gen_avg_param)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.arch,
            'gen_state_dict': gen_net.state_dict(),
            'dis_state_dict': dis_net.state_dict(),
            'avg_gen_state_dict': avg_gen_net.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict(),
            'path_helper': args.path_helper
        }, False, args.path_helper['ckpt_path'])
        del avg_gen_net


if __name__ == '__main__':
    main()
