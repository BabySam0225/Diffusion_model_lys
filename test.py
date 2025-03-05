import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import numpy as np

import options as option
from models import create_model

# sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, default="options/test/nwpu.yml", help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

print("current working directory:", os.getcwd())
result_dir = "result"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
log_dir = os.path.join(result_dir,"log")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

util.setup_logger(
    "base",
    log_dir,
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default 加载预训练模型
model = create_model(opt)
device = model.device

sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)

scale = opt['degradation']['scale']  # 设置噪声比例

for test_loader in test_loaders:  # 遍历之前创建的所有测试数据加载器
    test_set_name = test_loader.dataset.opt["name"]  # path opt[''] 获取当前测试数据集的名称，并使用logger记录开始测试的信息
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_sudir = os.path.join(result_dir, test_set_name)  # 创建一个目录用于保存当前数据集的测试结果
    util.mkdir(dataset_sudir)

    test_results = OrderedDict()  # 创建一个有序字典来存储不同指标的测试结果
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["psnr_y"] = []
    test_results["ssim_y"] = []
    test_results["lpips"] = []
    test_times = []  # 创建一个列表来记录每张图像的处理时间

    for i, test_data in enumerate(test_loader):  # 遍历数据加载器中的每张图像
        single_img_psnr = []
        single_img_ssim = []
        single_img_psnr_y = []
        single_img_ssim_y = []
        need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True  # 确定是否需要真实图像（GT）
        img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        #### input dataset_LQ
        LQ, GT = test_data["LQ"], test_data["GT"]
        LQ = util.upscale(LQ, scale)  # 使用util.upscale函数对LQ图像进行上采样
        noisy_state = sde.noise_state(LQ)  # 使用SDE对象的noise_state方法为LQ图像添加噪声

        ### 模型测试
        model.feed_data(noisy_state, LQ, GT)  # 输入
        tic = time.time()
        model.test(sde, save_states=True)  # 输出
        toc = time.time()
        test_times.append(toc - tic)

        visuals = model.get_current_visuals()
        SR_img = visuals["Output"]
        output = util.tensor2img(SR_img.squeeze())  # uint8
        LQ_ = util.tensor2img(visuals["Input"].squeeze())  # uint8
        GT_ = util.tensor2img(visuals["GT"].squeeze())  # uint8
        
        suffix = opt["suffix"]  # 为输出图像添加后缀
        if suffix:
            save_img_path = os.path.join(dataset_sudir, img_name + suffix + ".png")
        else:
            save_img_path = os.path.join(dataset_sudir, img_name + ".png")
        util.save_img(output, save_img_path)  # 保存

        # remove it if you only want to save output images
        
    print(f"average test time: {np.mean(test_times):.4f}")  # 计算平均测试时间
