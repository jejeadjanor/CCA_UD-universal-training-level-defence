from collections import defaultdict, deque
import datetime
import time
import umap
import h5py
import torch
import torch.distributed as dist

import errno
import os
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity



class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)



import numpy as np

def PCA_analysis(input: np.ndarray)->(np.ndarray, np.ndarray):
    # input: matrix with shape [n,p] where n is the number of data and p is the dimension
    # output:
    # component: with shape [p,p], where the top row is the most principle component
    # variance: with shape [p,] where the value is the correspoonding variance (sorted in the decending order)
    u, s, vh = np.linalg.svd(input)
    component = vh
    variance = (s*s)/(input.shape[0]-1)

    return component, variance




def my_tpr_tnr(target, pred):
    assert len(target) == len(pred), 'my_accuracy_socre should have same shape input'
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(len(pred)):
        if target[i] == pred[i] == 1:
            TP += 1
        if pred[i] == 1 and target[i] != pred[i]:
            FP += 1
        if target[i] == pred[i] == 0:
            TN += 1
        if pred[i] == 0 and target[i] != pred[i]:
            FN += 1

    TPR, TNR = TP/(TP+FN+1e-12), TN/(TN+FP+1e-12)

    return TPR, TNR


# def feature_distribution_k_d(hdf5_file: str, k: int, target_class: int)->(np.array, np.array):
#     # open the hdf5 file where with (number_data, feature_length+1). the feature_length is 128 and final one is the indics of trainign dataset
#     with h5py.File(hdf5_file, "r") as f:
#         center_feature_matrix = f['{}'.format(target_class)][:, :-1] - np.mean(f['{}'.format(target_class)][:, :-1],
#                                                                                axis=0)  # centerize the data
#
#         score = []
#         eignvs, variances = PCA_analysis(center_feature_matrix)  # the final column is the indics
#         percentage_variance = (1.0 * np.sum(variances[:k]))/np.sum(variances)
#
#         for i in range(k):
#             score.append(np.matmul(center_feature_matrix, np.transpose(eignvs[i])))
#
#         feature_kd = np.stack(score, axis=1)
#         ground_truth = f['{}'.format(target_class)][:, -1]
#
#     return percentage_variance, feature_kd, ground_truth


def feature_distribution_k_d(hdf5_file: str, k: int, target_class: int, method_name: str, n_neighbor=15, min_dist=0.1)->(np.array, np.array):
    # open the hdf5 file where with (number_data, feature_length+1). the feature_length is 128 and final one is the indics of trainign dataset
    with h5py.File(hdf5_file, "r") as f:
        mean_v = np.mean(f['{}'.format(target_class)][:, :-1], axis=0)
        # std_v = np.std(f['{}'.format(target_class)][:, :-1], axis=0)
        center_feature_matrix = f['{}'.format(target_class)][:, :-1] - mean_v  # centerize the data

        indics = f['{}'.format(target_class)][:, -1]
        ground_truth = indics < 0   # less zero means this index is the poisoned data
        if method_name == 'PCA':
            # calculate the 2d vector in first two principle component
            score = []
            eignvs, _ = PCA_analysis(center_feature_matrix)  # the final column is the indics
            for i in range(k):
                score.append(np.matmul(center_feature_matrix, np.transpose(eignvs[i])))
            feature_kd = np.stack(score, axis=1)
        elif method_name == 'UMAP':
            feature_kd = umap.UMAP(n_components=k, n_neighbors=n_neighbor, min_dist=min_dist).fit_transform(center_feature_matrix)
        elif method_name == 'UMAP_cos':
            feature_kd = umap.UMAP(n_neighbors=n_neighbor, min_dist=min_dist,metric='cosine').fit_transform(center_feature_matrix)
        elif method_name == 'TSNE':
            feature_kd = TSNE(n_components=k, learning_rate='auto', init ='random').fit_transform(center_feature_matrix)
        elif method_name == 'TSNE_cos':
            cos_similarity = cosine_similarity(center_feature_matrix)
            cos_distance = 1 - cos_similarity
            feature_kd = TSNE(n_components=k, learning_rate='auto', init='random', metric="precomputed").fit_transform(100.0*(cos_distance - np.min(cos_distance))/np.max(cos_distance))

    return feature_kd, ground_truth, indics


def prediction(labels):
    # count the number for one label appearring in the labels, and the label with most highest one is the benign label.
    # the others are poisoned data
    # label of benign: 0
    # label of poison: 1
    labels_set, counts = np.unique(labels, return_counts=True)
    benign_label = labels_set[np.argmax(counts)] # benign label
    predictions = np.zeros_like(labels)
    predictions[labels!=benign_label] = 1

    return predictions

def interplotion(entropy_th_l, tpr_dif_th, fpr_dif_th, fpr_dif_th_pristine_set_benign_model, fpr_dif_th_pristine_set_poisoned_model,
                 threshold):
    tpr_dif_th_at_10_fpr_val, \
    fpr_dif_th_at_10_fpr_val,\
    fpr_dif_th_pristine_set_benign_model_at_10_fpr_val, \
    fpr_dif_th_pristine_set_poisoned_model_at_10_fpr_val = np.interp(threshold, entropy_th_l, tpr_dif_th), \
                                                           np.interp(threshold, entropy_th_l, fpr_dif_th), \
                                                           np.interp(threshold, entropy_th_l, fpr_dif_th_pristine_set_benign_model),\
                                                           np.interp(threshold, entropy_th_l, fpr_dif_th_pristine_set_poisoned_model)

    return tpr_dif_th_at_10_fpr_val, fpr_dif_th_at_10_fpr_val, fpr_dif_th_pristine_set_benign_model_at_10_fpr_val, fpr_dif_th_pristine_set_poisoned_model_at_10_fpr_val



import matplotlib.pyplot as plt
def draw_figure_with_five_lines(threshold_l, threshold_val, fpr_BCBM_dif_th, fpr_BCPM_dif_th, tpr_dif_th, fpr_dif_th, fpr_BCBM_val_dif_th, save_file):
    import matplotlib
    matplotlib.rcParams.update({'font.size': 12})
    plt.ylabel('percentage')
    plt.xlabel(r'$\theta$')

    plt.plot(threshold_l, np.array(tpr_dif_th)/100.0, label=r'$\overline{TPR}(PC)$')
    plt.plot(threshold_l, np.array(fpr_dif_th)/100.0, label=r'$\overline{FPR}(PC)$')
    plt.plot(threshold_l, np.array(fpr_BCBM_dif_th)/100.0, label=r'$\overline{FPR}(BC_B)$')
    plt.plot(threshold_l, np.array(fpr_BCPM_dif_th)/100.0, label=r'$\overline{FPR}(BC_P)$')

    plt.plot(threshold_l, np.array(fpr_BCBM_val_dif_th)/100.0, '--', label=r'$\overline{FPR}(D_{val}$)')
    _, ymax = plt.gca().get_ylim()
    plt.vlines(x=threshold_val, ymin=-3, ymax=ymax, colors='grey', linestyle='dotted')
    # plt.hlines(y=5, xmin=-0.02, xmax=threshold_val, colors='grey', linestyle='dotted')
    plt.gca().set_ylim(ymax=1.03)
    plt.gca().set_ylim(ymin=-0.03)
    plt.gca().set_xlim(xmin=-0.02)
    # plt.gca().text(threshold_val, -8, r'$\theta^{*}$', fontsize=10)
    # plt.gca().set_aspect(0.025)
    plt.gcf().set_figheight(3)
    plt.gcf().tight_layout()
    tpr_dif_th_at_10_fpr_val, fpr_dif_th_at_10_fpr_val, \
    fpr_dif_th_pristine_set_benign_model_at_10_fpr_val, \
    fpr_dif_th_pristine_set_poisoned_model_at_10_fpr_val = interplotion(threshold_l, tpr_dif_th, fpr_dif_th,
                                                                        fpr_BCBM_dif_th, fpr_BCPM_dif_th,
                                                                        threshold=threshold_val)
    plt.title('(TPR, FPR)({:.2f},{:.2f})PCPD,FPR{:.2f}BCPD,FPR{:.2f}BCBD@th{:.4f}'.format(tpr_dif_th_at_10_fpr_val,
                                                                                          fpr_dif_th_at_10_fpr_val,
                                                                                          fpr_dif_th_pristine_set_poisoned_model_at_10_fpr_val,
                                                                                          fpr_dif_th_pristine_set_benign_model_at_10_fpr_val,
                                                                                          threshold_val))

    plt.legend(loc='upper left')
    plt.savefig(save_file, dpi=1000)
    plt.close()