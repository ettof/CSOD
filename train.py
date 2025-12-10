import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import argparse
from model.cssam import CSSAM
from dataset.sod_dataset import getSODDataloader
import torch
from tqdm import tqdm
import os
import torch.nn.functional as F
from collections import OrderedDict
import torch.distributed as dist
import time
from utils.loss import LossFunc,compute_loss
from utils.AvgMeter import AvgMeter
from torch import nn
dist.init_process_group(backend="nccl")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--seed", type=int, default = 50
    )
    parser.add_argument(
        "--warmup_period", type = int, default = 5,
    )
    parser.add_argument(
        "--batch_size", type = int, default = 2, required = True
    )
    parser.add_argument(
        "--num_workers", type = int, default = 2
    )
    parser.add_argument(
        "--epochs", type = int, default=80
    )
    parser.add_argument(
        "--lr_rate", type = float, default = 0.0005,
    )
    parser.add_argument(
        "--img_size", type = int, default = 512
    )
    parser.add_argument(
        "--model_size", type = str, default = "large"
    )
    parser.add_argument(
        "--data_path", type = str, default = ''
    )
    parser.add_argument(
        "--txt_path", type = str, default = r'data/CSOD10K/class_list.txt', help="the postfix",
    )
    parser.add_argument(
        "--sam_ckpt", type = str, default = r'checkpoints/sam2.1_hiera_large.pt'
    )
    parser.add_argument(
        "--save_dir", type = str, default = "output/"
    )
    parser.add_argument(
        "--resume", type = str, default = "", help="If you need to train from begining, make sure 'resume' is empty str. If you want to continue training, set it to the previous checkpoint."
    )
    parser.add_argument(
        "--local-rank", type=int, default=-1, help="For distributed training: local_rank"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def trainer(net, dataloader, loss_func, optimizer, local_rank):
    net.train()
    loss_avg = AvgMeter()

    mae_avg = AvgMeter()
    if local_rank == 0:
        print("start trainning")

    start = time.time()

    sigmoid = torch.nn.Sigmoid()
    if local_rank == 0:
        data_generator = tqdm(dataloader)
    else:
        data_generator = dataloader
    for data in data_generator:
        img = data["img"].to(device).to(torch.float32)
        label = data["mask"].to(device).unsqueeze(1)
        class_label = data["label"].to(device)

        optimizer.zero_grad()
        pos_out, out, neg_out, class_out = net(img)
        out = sigmoid(out)
        pos_out = sigmoid(pos_out)
        neg_out = sigmoid(neg_out)

        out_loss = loss_func(out, label)

        pos_out_loss = loss_func(pos_out, label)

        neg_loss = loss_func(neg_out, 1 - label)

        class_loss = compute_loss(class_out, class_label)
        loss = out_loss + pos_out_loss + neg_loss + 0.1*class_loss

        loss_avg.update(loss.item(), img.shape[0])

        img_mae = torch.mean(torch.abs(out - label))

        mae_avg.update(img_mae.item(),n=img.shape[0])

        loss.backward()
        optimizer.step()

    temp_cost=time.time()-start

    print("local_rank:{}, loss:{}, mae:{}, cost_time:{:.0f}m:{:.0f}s".format(local_rank, loss_avg.avg, mae_avg.avg, temp_cost//60, temp_cost%60))

def valer(net, dataloader, local_rank):
    net.eval()
    if local_rank == 0:
        print("start validating")

    start = time.time()
    sigmoid = torch.nn.Sigmoid()
    mae_avg = AvgMeter()

    with torch.no_grad():
        if local_rank == 0:
            data_generator = tqdm(dataloader)
        else:
            data_generator = dataloader

        for data in data_generator:
            img = data["img"].to(device).to(torch.float32)
            ori_label = data['ori_mask'].to(device)
            # class_label = data["label"].to(device)

            _,out, _, _ = net(img)

            out = sigmoid(out[:,-1:,:,:])
            out = torch.nn.functional.interpolate(
                out, [ori_label.shape[1], ori_label.shape[2]], mode='bilinear', align_corners=False
            )

            # Calculate MAE
            img_mae = torch.mean(torch.abs(out - ori_label))
            mae_avg.update(img_mae.item(), n=1)

        temp_cost = time.time() - start
        print("local_rank:{}, val_mae:{:.4f}, cost_time:{:.0f}m:{:.0f}s".format(
            local_rank, mae_avg.avg, temp_cost // 60, temp_cost % 60
        ))

    return mae_avg.avg


def reshapePos(pos_embed, img_size):
    token_size = int(img_size // 16)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
    return pos_embed

def load(net,ckpt, img_size):
    ckpt=torch.load(ckpt,map_location='cpu')
    from collections import OrderedDict
    dict=OrderedDict()
    loaded_params = set()
    for k,v in ckpt['model'].items():
        if 'image_encoder' in k:
            dict[k[14:]] = v
            loaded_params.add(k[14:])
            continue
        if 'pos_embed' in k :
            dict[k] = reshapePos(v, img_size)
            loaded_params.add(k)
            continue
        if "mask_decoder" in k:
            dict["NPG_"+k[4:]] = v
            dict[k[4:]] = v
            dict["PPG_" +k[4:]] = v
            loaded_params.add("PPG_" +k[4:])
            loaded_params.add("NPG_"+k[4:])
            loaded_params.add(k[4:])
            continue
        if "sam_prompt_encoder" in k:
            dict[k] = v
            loaded_params.add(k)

    state = net.load_state_dict(dict, strict=False)
    return state, loaded_params

if __name__ == "__main__":
    args = parse_args()
    if args.local_rank == 0:
        print("start training, batch_size: {}, lr_rate: {}, warmup_period: {}, save to {}".format(args.batch_size, args.lr_rate, args.warmup_period, args.save_dir))
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.local_rank}")

    #Model definition and loading SAM pre-trained weights
    net = CSSAM(args.model_size).to(device)

    if args.resume == "":
        state, loaded_params = load(net, args.sam_ckpt, args.img_size)

    trainLoader = getSODDataloader(args.data_path, args.batch_size, args.num_workers, 'train',label_file=args.txt_path, img_size= args.img_size)
    valLoader = getSODDataloader(
        data_path=args.data_path,
        batch_size=1,
        num_workers=args.num_workers,
        mode='test',
        local_rank=args.local_rank,
        label_file=args.txt_path,
        img_size=args.img_size,
        max_rank=dist.get_world_size()
    )
    loss_func = LossFunc

    hungry_param = []
    full_param = []

    # Freeze parameters in trunk and neck layers
    for k, v in net.named_parameters():
        # Exclude 'neck' or 'trunk' parameters from training
        if "neck" in k:
            v.requires_grad = False
            continue  # Skip to next parameter

        if "trunk" in k and "adapter" not in k and "class" not in k:
            v.requires_grad = False
            continue  # Skip to next parameter

        if "prompt_encoder" in k:
            v.requires_grad = False
            continue  # Skip to next parameter

        if k in loaded_params:
            full_param.append(v)
        else:
            hungry_param.append(v)

    # Print for debugging
    if args.local_rank == 0:
        print("Number of full_param: ", len(full_param))
        print("Number of hungry_param: ", len(hungry_param))
    optimizer = torch.optim.AdamW([{"params": hungry_param, "lr": args.lr_rate },
                                   {"params" : full_param, "lr" : args.lr_rate * 0.1},
                                   ], weight_decay=1e-5)

    best_mae = 1
    best_epoch = 0
    start_epoch = 1

    net=torch.nn.parallel.DistributedDataParallel(net.to(device),device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)

    for i in range(start_epoch, args.epochs + 1):

        #lr_rate setting
        if i <= args.warmup_period:
            _lr = args.lr_rate * i / args.warmup_period
        else:
            _lr = args.lr_rate * (0.98 ** (i - args.warmup_period))

        t = 0
        for param_group in optimizer.param_groups:

            if t == 0:
                param_group['lr'] = _lr
            else:
                param_group['lr'] = _lr * 0.1
            t += 1

        if args.local_rank == 0:
            print("epochs {} start,lr = {}".format(i,_lr))

        trainer(net, trainLoader, loss_func, optimizer, local_rank=args.local_rank)
        local_mae = valer(net, valLoader, local_rank = args.local_rank)

        #average the results from multi-GPU inference
        sum_result = torch.tensor(local_mae).to(device)
        dist.reduce(sum_result, dst = 0, op = dist.ReduceOp.SUM)

        if args.local_rank == 0:
            mae = sum_result.item() / dist.get_world_size()
            print("current mae:{}".format(mae))
            #save the best result
            if(mae < best_mae):
                best_mae = mae
                best_epoch = i
                print("save epoch {} in {}".format(i, "{}/model_epoch{}.pth".format(args.save_dir,i)))
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                torch.save({"model": net.state_dict(),"optimizer":optimizer.state_dict()}, "{}/model_epoch{}.pth".format(args.save_dir,i))
            print("best epoch:{}, mae:{}".format(best_epoch,best_mae))





