import scheduler
import sys
import argparse
import torch
import shutil
import math
import random
import timeit
import os


import torch
import torch.distributed as dist

base_line_batch_size = 256


def GetScheduler(scheduler_configs, param_list) -> scheduler.Scheduler:
    schdl_base = getattr(scheduler, scheduler_configs['type'])
    return schdl_base(scheduler_configs['optimizer'], param_list, **scheduler_configs['kwargs'])


def str2num(src: str) -> list:
    return list(src.encode())


def num2str(num_list: list) -> str:
    return bytes(num_list).decode()


def gather_all_gpu_info(local_gpuid: int, num_all_gpus: int = None) -> list:
    """Gather all gpu info based on DDP backend

    This function is supposed to be invoked in all sub-process.
    """
    if num_all_gpus is None:
        num_all_gpus = dist.get_world_size()

    gpu_info = torch.cuda.get_device_name(local_gpuid)
    gpu_info_len = torch.tensor(len(gpu_info)).cuda(local_gpuid)
    dist.all_reduce(gpu_info_len, op=dist.ReduceOp.MAX)
    gpu_info_len = gpu_info_len.cpu()
    gpu_info = gpu_info + ' ' * (gpu_info_len-len(gpu_info))

    unicode_gpu_info = torch.tensor(
        str2num(gpu_info), dtype=torch.uint8).cuda(local_gpuid)
    info_list = [torch.empty(
        gpu_info_len, dtype=torch.uint8, device=local_gpuid) for _ in range(num_all_gpus)]
    dist.all_gather(info_list, unicode_gpu_info)
    return [num2str(x.tolist()).strip() for x in info_list]


def gen_readme(path, model, gpu_info: list = []):
    if os.path.exists(path):
        highlight_msg(f"Not generate new readme, since '{path}' exists.")
        return path

    model_size = count_parameters(model)/1e6

    msg = [
        "### Basic info",
        "",
        "**This part is auto generated, add your details in Appendix**",
        "",
        "* Model size/M: {:.2f}".format(model_size),
        f"* GPU info \[{len(gpu_info)}\]"
    ]
    gpu_set = list(set(gpu_info))
    gpu_set = {x: gpu_info.count(x) for x in gpu_set}
    gpu_msg = [f"  * \[{num_device}\] {device_name}" for device_name,
               num_device in gpu_set.items()]

    msg += gpu_msg + [""]
    msg += [
        "### Appendix",
        "",
        "* ",
        ""
    ]
    msg += [
        "### WER"
        "",
        "```",
        "",
        "```",
        "",
        "### Monitor figure",
        "![monitor](./monitor.png)",
        ""
    ]
    with open(path, 'w') as fo:
        fo.write('\n'.join(msg))

    return path


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def highlight_msg(msg: str):
    len_str = len(msg)
    if '\n' in msg:
        msg = msg.split('\n')
        msg = '\n'.join(['# ' + line + ' #' for line in msg])
    else:
        msg = '# ' + msg + ' #'
    print('\n' + "#"*(len_str + 4))
    print(msg)
    print("#"*(len_str + 4) + '\n')


def save_ckpt(state, is_best, ckpt_path, filename):
    torch.save(state, '{}/{}'.format(ckpt_path, filename))
    if is_best:
        shutil.copyfile('{}/{}'.format(ckpt_path, filename),
                        '{}/best_model'.format(ckpt_path))


def train(model, tr_dataloader, optimizer, epoch, args, logger):
    model.train()
    prev_t = timeit.default_timer()
    for i, minibatch in enumerate(tr_dataloader):
        try:
            features, input_lengths, labels_padded, label_lengths, path_weights = minibatch
            if (features.size(0) < args.gpu_batch_size):
                continue
            model.zero_grad()
            optimizer.zero_grad()

            loss = model(features, labels_padded, input_lengths, label_lengths)
            if args.ctc_crf:
                partial_loss = torch.mean(loss.cpu())
                weight = torch.mean(path_weights)
                real_loss = partial_loss - weight
            else:
                real_loss = torch.mean(loss.cpu())
            loss.backward()
            optimizer.step()

            t2 = timeit.default_timer()
            if i % 200 == 0 and args.rank == 0:
                logger.debug("epoch: {}, step: {}, time: {}, tr_real_loss: {}, lr: {}".format(
                    epoch, i, t2 - prev_t, real_loss.item(), optimizer.param_groups[0]['lr']))
            prev_t = t2
        except Exception as ex:
            print("rank {} train exception ".format(args.rank), ex)


def validate(model, cv_dataloader, epoch, args, logger):
    # cv stage
    model.eval()
    cv_total_loss = 0
    cv_total_sample = 0
    with torch.no_grad():
        for i, minibatch in enumerate(cv_dataloader):
            try:
                features, input_lengths, labels_padded, label_lengths, path_weights = minibatch
                loss = model(features, labels_padded,
                             input_lengths, label_lengths)
                if args.ctc_crf:
                    partial_loss = torch.mean(loss.cpu())
                    weight = torch.mean(path_weights)
                    real_loss = partial_loss - weight
                else:
                    real_loss = torch.mean(loss.cpu())
                cv_total_loss += real_loss.item() * features.size(0)
                cv_total_sample += features.size(0)
            except Exception as ex:
                print("rank {} cv exception ".format(args.rank), ex)
        cv_loss = cv_total_loss / cv_total_sample
        if args.rank == 0:
            logger.info("epoch: {}, mean_cv_loss: {}".format(epoch, cv_loss))
    return cv_loss


def train_chunk_model(model, reg_model, tr_dataloader, optimizer, epoch, chunk_size, TARGET_GPUS, args, logger):
    prev_t = 0
    for i, minibatch in enumerate(tr_dataloader):
        try:
            logits, input_lengths, labels_padded, label_lengths, path_weights = minibatch
            model.zero_grad()
            optimizer.zero_grad()
            input_lengths = map(lambda x: x.size()[0], logits)
            if sys.version > '3':
                input_lengths = list(input_lengths)

            input_lengths = torch.IntTensor(input_lengths)
            out1_reg, out2_reg, out3_reg = reg_model(
                logits, labels_padded, input_lengths, label_lengths)
            loss, loss_cls, loss_reg = model(logits, labels_padded, input_lengths, label_lengths,
                                             chunk_size, out1_reg.detach(), out2_reg.detach(), out3_reg.detach())

            partial_loss = torch.mean(loss.cpu())
            loss_cls = torch.mean(loss_cls.cpu())
            loss_reg = torch.mean(loss_reg.cpu())

            weight = torch.mean(path_weights)
            real_loss = partial_loss - weight
            loss_cls = loss_cls - weight

            count = min(loss.size(0), len(TARGET_GPUS))
            loss.backward(loss.new_ones(count))

            optimizer.step()
            t2 = timeit.default_timer()
            if i % 200 == 0 and args.rank == 0:
                logger.debug("rank {} epoch:{} step:{} time: {}, tr_real_loss: {},loss_cls: {},loss_reg:{}, lr: {}".format(
                    args.rank, epoch, i, t2 - prev_t, real_loss.item(), loss_cls.item(), loss_reg.item(), optimizer.param_groups[0]['lr']))
            prev_t = t2
            torch.cuda.empty_cache()
        except Exception as ex:
            print("rank {} train exception ".format(args.rank), ex)


def validate_chunk_model(model, reg_model, cv_dataloader, epoch, cv_losses_sum, cv_cls_losses_sum, args, logger):
    count = 0
    for i, minibatch in enumerate(cv_dataloader):
        try:
            #print("cv epoch: {}, step: {}".format(epoch, i))
            logits, input_lengths, labels_padded, label_lengths, path_weights = minibatch

            input_lengths = map(lambda x: x.size()[0], logits)
            if sys.version > '3':
                input_lengths = list(input_lengths)
            input_lengths = torch.IntTensor(input_lengths)

            reg_out1, reg_out2, reg_out3 = reg_model(
                logits, labels_padded, input_lengths, label_lengths)
            loss, loss_cls, loss_reg = model(logits, labels_padded, input_lengths, label_lengths,
                                             args.default_chunk_size, reg_out1.detach(), reg_out2.detach(), reg_out3.detach())

            loss_size = loss.size(0)
            count += loss_size

            partial_loss = torch.mean(loss.cpu())
            weight = torch.mean(path_weights)
            real_loss = partial_loss - weight

            loss_cls = torch.mean(loss_cls.cpu())
            loss_cls = loss_cls - weight
            loss_reg = torch.mean(loss_reg.cpu())

            real_loss_sum = real_loss * loss_size
            loss_cls_sum = loss_cls * loss_size

            cv_losses_sum.append(real_loss_sum.item())
            cv_cls_losses_sum.append(loss_cls_sum.item())
        except Exception as ex:
            print("rank {} cv exception ".format(args.rank), ex)
    return count


def adjust_lr(optimizer, origin_lr, lr, cv_loss, prev_cv_loss, epoch, min_epoch):
    if epoch < min_epoch or cv_loss <= prev_cv_loss:
        pass
    else:
        lr = lr / 10.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_lr_distribute(optimizer, origin_lr, lr, cv_loss, prev_cv_loss, epoch, annealing_epoch, gpu_batch_size, world_size):
    '''
    The hyperparameter setup for the batch size 256
    configuration is the learning rate is set to be 0.1, momentum is
    set as 0.9, and learning rate anneals by p12 every epoch from the
    11th epoch. The training finishes in 16 epochs. Inspired by the
    work proposed in[4], we are able to increase the batch size from
    256 to 2560 without decreasing model accuracy by (1) linearly
    warming up the base learning rate from 0.1 to 1 in the first 10
    epochs and (2) annealing the learning rate by p12 from the 11th
    Epoch. refer Wei Zhang, Xiaodong Cui, Ulrich Finkler, Brian Kingsbury,
    George Saon, David Kung, Michael Picheny "Distributed Deep Learning 
    Strategies For Automatic Speech Recognition"
    '''

    new_lr = lr
    batch_size = gpu_batch_size * world_size
    if epoch < annealing_epoch:
        if batch_size > base_line_batch_size:
            max_lr = (batch_size/base_line_batch_size) * origin_lr
            new_lr = lr + round(max_lr/annealing_epoch, 6)
        else:
            new_lr = origin_lr
    else:
        new_lr = lr / math.sqrt(2)

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return new_lr


def parse_args():
    parser = argparse.ArgumentParser(description="recognition argument")
    parser.add_argument("--model", type=str,
                        help="the model name will be trained")
    parser.add_argument(
        "--arch",
        choices=[
            'BLSTM', 'LSTM', 'VGGBLSTM', 'VGGLSTM', 'LSTMrowCONV', 'TDNN_LSTM',
            'BLSTMN'
        ],
        default='BLSTM')
    parser.add_argument("--ctc_crf", action="store_true", default=False,
                        help="whether to use ctc_crf or not, true for ctc_crf, false for ctc")
    parser.add_argument("--min_epoch", type=int, default=10)
    parser.add_argument("--lamb", type=float, default=0.1)
    parser.add_argument("--hdim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--output_unit", type=int,
                        help='the phone number of your model output, must be specified')
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--feature_size", type=int, default=120)
    parser.add_argument("--origin_lr", type=float, default=0.001)
    parser.add_argument("--annealing_epoch", type=int, default=15)
    parser.add_argument("--data_loader_workers", type=int, default=16)
    parser.add_argument("--reg_weight", type=float, default=0.01)
    parser.add_argument("--stop_lr", type=float, default=0.00001)
    parser.add_argument("--batch_size",  type=int, default=32,
                        help='batch size for non distribute training')
    parser.add_argument("--tr_data_path", type=str, help="training data path")
    parser.add_argument("--dev_data_path", type=str,
                        help="validation data path")
    parser.add_argument("--den_lm_fst_path", type=str, default=None,
                        help="denominator fst path, be used in ctc_crf")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="the checkpoint file path of pretrained model")
    parser.add_argument('--dist_url', type=str, default=None,
                        help='Url used to set up distributed training. It must be the IP of device running the rank 0, \
                        and a free port can be opened and listened by pytorch distribute framework.')
    parser.add_argument('--world_size', type=int, default=0, help='world size must be set to indicate the number of ranks you \
                        will  start to train your model. If the real rank number is less then the value of \
                        --world_size it will be blocked on function init_process_group. Rather than that if a rank \
                        number grater then the value of world_size was given the function init_process_group will \
                        throw a exception to terminate the training. So you need make sure the world size is equal \
                        to the ranks number.')
    parser.add_argument("--start_rank", type=int, default=0, help='This value was used to specify the start rank on the device. \
                        For example, if you have 3 gpu and 3 ranks on device 1 and the 4 gpu and 4 ranks on \
                        device 2. The device 1 has rank 0, then the device 2 must start from rank 3.So you must \
                        set --start_rank 3 on device 2. ')

    parser.add_argument('--gpu_batch_size', default=32, type=int,
                        help='This value was depended on the memory size of your gpu, the biger the value was set, \
                        the higher training speed you can get. The total batch size of your training was \
                        (world size) * (batch size per gpu). The max batch size must not grater then 2560, \
                        or rather the training will not be convergent.')

    parser.add_argument("--spec_augment", action="store_true", default=False,
                        help="whether to use specAugment in training phase or not, true for using")

    parser.add_argument("--regmodel_checkpoint", type=str, default=None,
                        help='the checkpoint file path of reg model')

    parser.add_argument("--default_chunk_size", type=int, default=40,
                        help='default chunk size for low latency model')

    parser.add_argument("--jitter_range", type=int, default=10,
                        help='contextual frame number of low latency model')
    parser.add_argument("--cate", type=int, default=4000,
                        help='the number of pkls the training ark data was diviced and assigned to by utterence length')

    args = parser.parse_args()

    args.csv_file = args.model + ".csv"
    args.figure_file = args.model + ".png"

    return args
