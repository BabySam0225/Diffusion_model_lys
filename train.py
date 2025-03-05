import argparse
import logging
import math
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

# 分布式训练，可在多个GPU上进行训练
def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training初始化分布式训练"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # 返回可用gpu的数量
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # 初始化分布式进程组


def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, default="options/train/setting.yml")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"  # none means disabled distributed training
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### 设置随机种子
    seed = opt["train"]["manual_seed"]

    #### 分布式训练设置
    if args.launcher == "none":  # disabled distributed training 禁用分布式训练
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True  # 初始化分布式环境
        opt["dist"] = True  #
        init_dist()
        world_size = (torch.distributed.get_world_size())  # 返回当前进程组中的进程数
        rank = torch.distributed.get_rank()  # 返回当前进程组的等级/排名
        # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True # 当设置为True时，cudnn会尝试寻找最优的卷积算法来加速卷积操作。训练速度会提升，但是可能会增加内存的使用量，并且在某些情况下可能会影响模型的稳定性
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers 创建目录和日志记录
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
            # os.system("rm ./log")
            # os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        # config loggers. 配置日志记录器
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")


    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### create model
    model = create_model(opt) 
    device = model.device

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)

    scale = opt['degradation']['scale']

    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    best_psnr = 0.0
    best_iter = 0
    error = mp.Value('b', False)

    # -------------------------------------------------------------------------
    # -------------------------------正式开始训练--------------------------------
    # -------------------------------------------------------------------------
    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader): # 遍历数据加载器
            current_step += 1

            if current_step > total_iters:
                break

            LQ, GT = train_data["LQ"], train_data["GT"]  # b 3 32 32; b 3 128 128
            # bicubic, which can be repleced by deep networks使用双三次插值方法对低质量图像LQ进行上采样，以匹配真实高质量图像GT的尺寸
            LQ = util.upscale(LQ, scale)

            # random timestep and state (noisy map) via SDE 使用SDE模型生成随机时间步和状态，这些状态将用于模拟图像中的噪声
            timesteps, states = sde.generate_random_states(x0=GT, mu=LQ)  # t=batchsize，states [b 3 128 128]

            model.feed_data(states, LQ, GT)  # xt, mu, x0, 将加了噪声的LR图xt，LR以及GT输入改进的UNet进行去噪！！！！！！！！！

            model.optimize_parameters(current_step, timesteps, sde)

            model.update_learning_rate(current_step, warmup_iter=opt["train"]["warmup_iter"])  # 更新学习率

            # 记录训练日志
            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation, to produce ker_map_list(fake) 验证模型性能
            if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
                avg_psnr = 0.0
                idx = 0
                for _, val_data in enumerate(val_loader):

                    LQ, GT = val_data["LQ"], val_data["GT"]
                    LQ = util.upscale(LQ, scale)
                    noisy_state = sde.noise_state(LQ)  # 在LR上加噪声，得到噪声LR图，噪声是随机生成的

                    # valid Predictor
                    model.feed_data(noisy_state, LQ, GT)
                    model.test(sde)
                    visuals = model.get_current_visuals()

                    output = util.tensor2img(visuals["Output"].squeeze())  # uint8
                    gt_img = util.tensor2img(visuals["GT"].squeeze())  # uint8

                    # save the validation results
                    save_path = str(opt["path"]["experiments_root"]) + '/val_images/' + str(current_step)
                    util.mkdirs(save_path)
                    save_name = save_path + '/'+'{0:03d}'.format(idx) + '.png'
                    util.save_img(output, save_name)

                    # calculate PSNR
                    avg_psnr += util.calculate_psnr(output, gt_img)
                    idx += 1

                avg_psnr = avg_psnr / idx

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_iter = current_step

                # log
                logger.info("# Validation # PSNR: {:.6f}, Best PSNR: {:.6f}| Iter: {}".format(avg_psnr, best_psnr, best_iter))
                logger_val = logging.getLogger("val")  # validation logger
                logger_val.info(
                    "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                        epoch, current_step, avg_psnr
                    )
                )
                print("<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                        epoch, current_step, avg_psnr
                    ))
                # tensorboard logger
                if opt["use_tb_logger"] and "debug" not in opt["name"]:
                    tb_logger.add_scalar("psnr", avg_psnr, current_step)

            if error.value:
                sys.exit(0)
            #### save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
    tb_logger.close()


if __name__ == "__main__":
    main()
