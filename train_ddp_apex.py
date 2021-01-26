import os
import socket
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import torch.nn.functional as F
import apex
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier
from torch.cuda.amp import autocast
import torch
import models.losses as losses
import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
from utils.fid_scores import fid_pytorch
import config
import random

def train(rank, world_size, opt):
    setup(rank, world_size, opt)

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    ############################################################################
    # register the local rank - a copy of this train function is sent to each gpu
    ############################################################################
    opt.rank = rank

    ############################################################################
    # Stuff for logging
    ############################################################################
    if opt.rank==0:
        #--- create utils ---#
        timer = utils.timer(opt)
        visualizer_losses = utils.losses_saver(opt)
    dist.barrier()

    ############################################################################
    # Get dataloader, model, optimizer and
    # some helper functions
    ############################################################################
    losses_computer = losses.losses_computer(opt)
    dataloader, dataloader_val = dataloaders.get_dataloaders(opt, distributed_data_parallel = True)


    if opt.rank==0:
        im_saver = utils.image_saver(opt)
        fid_computer = fid_pytorch(opt, dataloader_val)
    dist.barrier()

    len_dataloader = len(dataloader)
    model = models.OASIS_model_ddp(opt)#, ema = opt.rank==0)

    model.netG = model.netG.to(opt.rank)
    model.netD = model.netD.to(opt.rank)
    model.netEMA = model.netEMA.to(opt.rank)

    optimizerG = torch.optim.Adam(model.netG.parameters(),
                                    lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
    optimizerD = torch.optim.Adam(model.netD.parameters(),
                                    lr=opt.lr_d, betas=(opt.beta1, opt.beta2))

    if opt.amp_implemention == "apex":
        [model.netG, model.netD ], [optimizerG, optimizerD] = amp.initialize(
                    [model.netG, model.netD ], [optimizerG, optimizerD],
                    opt_level= 'O1', num_losses = 2)#, keep_batchnorm_fp32 = True)
                                          #opt_level=args.opt_level,
                                          #keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                          #loss_scale=args.loss_scale
                                          #)
        #model.netD, optimizer = amp.initialize(model.netD, optimizerD, opt_level="01")
        ############################################################################
        # Prepare model and dataloader for multi-gpu training.
        # This handles also the single gpu case automatically.
        ############################################################################

        model.netG = apex.parallel.DistributedDataParallel( model.netG, delay_allreduce=True)
        model.netD = apex.parallel.DistributedDataParallel( model.netD, delay_allreduce=True)
        model.netEMA = apex.parallel.DistributedDataParallel( model.netEMA, delay_allreduce=True)

    elif opt.amp_implemention == "pytorch_native":
        scaler = torch.cuda.amp.GradScaler()

    #dist.barrier()
    model.netG.train()
    model.netD.train()
    model.netEMA.train()



    start_epoch, start_iter = utils.get_start_iters(opt.loaded_latest_iter, len_dataloader)
    for epoch in range(start_epoch, opt.num_epochs):
        for i, data_i in enumerate(dataloader):

            cur_iter = epoch*len_dataloader + i
            image, label = models.preprocess_input(opt, data_i)
            #image = image.half()
            #label = label.half()


            if opt.amp_implemention == "apex":
                #--- generator update ---#
                model.netG.zero_grad()
                loss_G, losses_G_list = model.forward(image, label, "losses_G", losses_computer)
                loss_G, losses_G_list = loss_G.mean(), [loss.mean() if loss is not None else None for loss in losses_G_list]

                with amp.scale_loss(loss_G, optimizerG, loss_id=0) as scaled_loss_G:
                    scaled_loss_G.backward()

                optimizerG.step()

                #--- discriminator update ---#
                model.netD.zero_grad()
                loss_D, losses_D_list = model.forward(image, label, "losses_D", losses_computer)
                loss_D, losses_D_list = loss_D.mean(), [loss.mean() if loss is not None else None for loss in losses_D_list]

                with amp.scale_loss(loss_D, optimizerD, loss_id=1) as scaled_loss_D:
                    scaled_loss_D.backward()

                optimizerD.step()

            elif opt.amp_implemention == "pytorch_native":
                with autocast():
                    loss_G, losses_G_list = model.forward(image, label, "losses_G", losses_computer)
                    loss_G, losses_G_list = loss_G.mean(), [loss.mean() if loss is not None else None for loss in losses_G_list]
                    #print("loss_G", loss_G)
                scaler.scale(loss_G).backward()
                scaler.step(optimizerG)

                with autocast():
                    loss_D, losses_D_list = model.forward(image, label, "losses_D", losses_computer)
                    loss_D, losses_D_list = loss_D.mean(), [loss.mean() if loss is not None else None for loss in losses_D_list]
                    #print("loss_D", loss_D)
                scaler.scale(loss_D).backward()
                scaler.step(optimizerD)

                scaler.update()

            #--- stats update ---#
            if not opt.no_EMA:
                utils.update_EMA(model, cur_iter, dataloader, opt)

            if opt.rank==0:

                if cur_iter % opt.freq_print == 0:
                    im_saver.visualize_batch(model, image, label, cur_iter)
                    timer(epoch, cur_iter)
                if cur_iter % opt.freq_save_ckpt == 0:
                    utils.save_networks(opt, cur_iter, model)
                if cur_iter % opt.freq_save_latest == 0:
                    utils.save_networks(opt, cur_iter, model, latest=True)
                if cur_iter % opt.freq_fid == 0 and cur_iter > 0:
                    is_best = fid_computer.update(model, cur_iter)
                    if is_best:
                        utils.save_networks(opt, cur_iter, model, best=True)

                        # TODO: for amp pytorch_native, save scaler,
                        # as in https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#saving-resuming
                visualizer_losses(cur_iter, losses_G_list+losses_D_list)
            #dist.barrier()

    #--- after training ---#
    if opt.rank==0:
        utils.update_EMA(model, cur_iter, dataloader, opt, force_run_stats=True)
        utils.save_networks(opt, cur_iter, model)
        utils.save_networks(opt, cur_iter, model, latest=True)
        is_best = fid_computer.update(model, cur_iter)
        if is_best:
            utils.save_networks(opt, cur_iter, model, best=True)

        print("The training has successfully finished")

    dist.destroy_process_group()


def port_is_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return not s.connect_ex(('localhost', port)) == 0

'''
def init_process(rank, size, fct , opy):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = opy.selected_port
    torch.cuda.set_device(opy.rank)
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(opy.rank)
    dist.init_process_group(backend='nccl', init_method='env://',
                            rank=rank, world_size=size)
    fct(opy)
    dist.destroy_process_group()
'''

def setup(rank, world_size, opt):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = opt.selected_port

    # initialize the process group
    torch.cuda.set_device(rank)
    #dist.init_process_group("gloo", rank=rank, world_size=world_size)
    dist.init_process_group(backend='nccl', init_method='env://',
                            rank=rank, world_size=world_size)

if __name__ == '__main__':

    ############################################################################
    # load config
    ############################################################################
    opt = config.read_arguments(train=True)

    ############################################################################
    # Boilerplate for multiprocessing
    ############################################################################

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    num_gpus = len(opt.gpu_ids.replace(",",""))
    opt.num_gpus = num_gpus
    opt.distributed = num_gpus > 1
    torch.multiprocessing.set_start_method('spawn')
    opt.ddp_apex = True

    if opt.amp_implemention == "apex":
        opt.lr_g = opt.lr_g/10
        opt.lr_d = opt.lr_d/10
    elif opt.amp_implemention == "pytorch_native":
        opt.lr_g = opt.lr_g#/2
        opt.lr_d = opt.lr_d#/2

    ############################################################################
    # Test some random ports and see if they are free. Necessary when running
    # on a cluster with shared nodes. If you have a better way of doing this
    # create a github issue.
    ############################################################################
    for i in range(20):
        selected_port = random.randint(0,65535)
        if port_is_free(selected_port):
            print("using port ",selected_port)
            selected_port = str(selected_port)
            break
    opt.selected_port = selected_port

    mp.spawn(train, args=(num_gpus,opt), nprocs=num_gpus, join=True)
