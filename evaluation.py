import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import argparse
from dataset.sod_dataset import getSODDataloader
from model.cssam import CSSAM
import torch
from tqdm import tqdm
import os
import shutil
from collections import OrderedDict
import numpy as np
import cv2
import os
import py_sod_metrics
import time
import matplotlib.pyplot as plt
dataset = ["CSOD10K"]
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of an evaluation script.")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
    )
    parser.add_argument(
        "--data_path", type=str, required=True
    )
    parser.add_argument(
        "--result_path", type=str, default="./output"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0
    )
    parser.add_argument(
        "--img_size", type=int, default=512,
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0
    )

    parser.add_argument(
        "--model_size", type = str, default = "large"
    )

    args = parser.parse_args()

    return args


def eval(net, dataloader, output_path, dataset):
    net.eval()
    print("Start eval dataset: {}".format(dataset))

    sigmoid = torch.nn.Sigmoid()

    MAE = py_sod_metrics.MAE()
    WFM = py_sod_metrics.WeightedFmeasure()
    SM = py_sod_metrics.Smeasure()
    EM = py_sod_metrics.Emeasure()
    FM = py_sod_metrics.Fmeasure()

    total_frames = 0
    total_time = 0.0

    with torch.no_grad():
        for data in tqdm(dataloader, ncols=100):
            img = data["img"].to(device).to(torch.float32)
            ori_label = data['ori_mask']
            name = data['mask_name']

            start_time = time.time()
            pos_out, out, neg_out, class_out = net(img)

            end_time = time.time()
            total_time += (end_time - start_time)
            total_frames += 1

            out = sigmoid(out)

            out = torch.nn.functional.interpolate(out, [ori_label.shape[1], ori_label.shape[2]], mode='bilinear',
                                                  align_corners=False)

            pred = (out * 255).squeeze().cpu().data.numpy().astype(np.uint8)
            ori_label = (ori_label * 255).squeeze(0).data.numpy().astype(np.uint8)

            FM.step(pred=pred, gt=ori_label)
            WFM.step(pred=pred, gt=ori_label)
            SM.step(pred=pred, gt=ori_label)
            EM.step(pred=pred, gt=ori_label)
            MAE.step(pred=pred, gt=ori_label)

            pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(output_path + "/" + name[0] + ".png", pred)


    # 计算指标
    fm = FM.get_results()["fm"]
    pr = FM.get_results()["pr"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = MAE.get_results()["mae"]

    maxFm = FM.get_results()['mf']
    meanFm = fm['curve'].mean()
    fm = fm['adp']
    em = em['curve'].mean()

    print("{} results:".format(dataset))
    print("mae:{:.5f}, maxFm:{:.4f},  sm:{:.3f}, em:{:.3f}".format(mae, maxFm, sm, em))

    fps = total_frames / total_time if total_time > 0 else 0
    print("Processed {} frames in {:.2f} seconds (inference only), FPS: {:.2f}".format(total_frames, total_time, fps))

    return mae, maxFm, sm, em

if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}")
    net = CSSAM(args.model_size).to(device)
    # load pretrained weights
    ckpt_dic = torch.load(args.checkpoint)
    if 'model' in ckpt_dic.keys():
        ckpt_dic = ckpt_dic['model']
    dic = OrderedDict()
    for k, v in ckpt_dic.items():
        if 'module.' in k:
            dic[k[7:]] = v
        else:
            dic[k] = v
    msg = net.load_state_dict(dic, strict=False)
    print(msg)
    datasets = ["CSOD10K"]
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    for dataset in datasets:
        testLoader = getSODDataloader(
            data_path=os.path.join(args.data_path, dataset),
            batch_size=1,
            num_workers=args.num_workers,
            mode='test',
            local_rank=0,
            label_file=r'data/CSOD10K/class_list.txt',
            img_size=args.img_size,
            max_rank=1,
        )
        dataset_result_path = os.path.join(args.result_path, dataset)

        if os.path.exists(dataset_result_path):
            shutil.rmtree(dataset_result_path)
        os.makedirs(dataset_result_path)


        eval(net, testLoader, dataset_result_path, dataset)
        # eval_classification_accuracy(net, testLoader, num_stages=4)
