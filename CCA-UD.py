import random

import h5py
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import umap
from sklearn.metrics import roc_curve, auc
import matplotlib.colors as mcolors
import torch.nn.functional as F

from mymodel import Net

from my_optics import OPTICS
import matplotlib.pyplot as plt
from utils import my_tpr_tnr
import os

debugging_flag = True


def draw_reachability_plot(clust, normalized_feature, labels, save_name, ground_or_pred='G'):
    ordered_labels = labels[clust.ordering_]
    space = np.arange(len(normalized_feature))
    reachability = clust.reachability_[clust.ordering_]
    assert ground_or_pred == 'G' or ground_or_pred == 'P', 'Error: ground_or_pred parameter is set wrongly'
    if ground_or_pred == 'G':
        plt.scatter(space[ordered_labels == 0], reachability[ordered_labels == 0], color='yellow', marker='.',
                    alpha=0.2, label='benign_num_{}'.format(len(space[ordered_labels == 0])))
        plt.scatter(space[ordered_labels == 1], reachability[ordered_labels == 1], color='red', marker='.',
                    alpha=0.2, label='poisoned_num_{}'.format(len(space[ordered_labels == 1])))
        plt.title("RP with ground-truth label")
        plt.legend()
    elif ground_or_pred == 'P':
        colors = mcolors.TABLEAU_COLORS
        c_name = list(colors)
        for label in set(labels):
           plt.scatter(space[ordered_labels == label], reachability[ordered_labels == label], color=colors[c_name[label]], marker='.',
                        alpha=0.2, label='label {}'.format(label))
        plt.title("RP with prediction label")
        plt.legend()

    plt.savefig(save_name+'.jpg')
    # plt.show()
    plt.close()


def draw_space_plot(normalized_feature, labels, save_name, ground_or_pred='G', cluster_entropy=None, tpr=None, fpr=None):
    assert ground_or_pred == 'G' or ground_or_pred == 'P', 'Error: ground_or_pred parameter is set wrongly'
    if ground_or_pred == 'G':
        plt.scatter(normalized_feature[labels == 1][:, 0], normalized_feature[labels == 1][:, 1], color='red',
                    label='poisoned_num_{}'.format(np.sum(labels == 1)), alpha=0.3)
        plt.scatter(normalized_feature[labels == 0][:, 0], normalized_feature[labels == 0][:, 1], color='yellow',
                    label='benign_num_{}'.format(np.sum(labels == 0)), alpha=0.3)
        plt.title("SP with ground-truth label")
    elif ground_or_pred == 'P':
        colors = mcolors.TABLEAU_COLORS
        c_name = list(colors)
        for label in set(labels):
            if label == -1:
                chosen_data = normalized_feature[labels == label]
                plt.scatter(chosen_data[:, 0], chosen_data[:, 1], color=colors[c_name[label]],
                            label='noise', alpha=0.3)
            else:
                chosen_data = normalized_feature[labels == label]
                plt.scatter(chosen_data[:, 0], chosen_data[:, 1], color=colors[c_name[label]],
                            label='label {} with entropy {:.3f}'.format(label, cluster_entropy[label]), alpha=0.3)
    if tpr == None:
        plt.title("SP with prediction label")
    else:
        plt.title("SP with prediction label (tpr {:.3f} and fpr {:.3f})".format(tpr, fpr))
    plt.legend()
    plt.savefig(save_name + '.jpg')
    # plt.show()
    plt.close()


def draw_sdas_suas(clust, sdas, suas, save_name):
    reachability = clust.reachability_[clust.ordering_]
    space = np.arange(len(reachability))
    # get label based on the sdas, suas
    # 0: not steep area
    # 1: steap down areas
    # 2: steep up areas
    labels = np.zeros(len(reachability))
    for D_area in sdas:
        labels[D_area['start']:D_area['end']+1] = 1

    for U_area in suas:
        labels[U_area['start']:U_area['end']+1] = 2


    plt.scatter(space[labels==0], reachability[labels==0], color='grey', marker='.', alpha=1, label='non steep area')
    plt.scatter(space[labels == 1], reachability[labels == 1], color='red', marker='.', alpha=1, label='steep down')
    plt.scatter(space[labels == 2], reachability[labels == 2], color='yellow', marker='.', alpha=1, label='steep up')
    plt.title("steep areas")
    plt.legend()
    # plt.show()
    plt.savefig(save_name+'.jpg', dpi=600)
    plt.close()


def plot_figures(clust, normalized_feature, ground_truth, predition, cluster_entropy, save_path, save_name, validation_flag, tpr=None, fpr=None):
    if validation_flag:
        sub_name = 'validation'
    else:
        sub_name = 'evaluation'

    draw_space_plot(normalized_feature, ground_truth,
                    save_name=os.path.join(save_path + r'\{}\space_plot\ground_truth'.format(sub_name), save_name),
                    ground_or_pred='G')
    draw_space_plot(normalized_feature, predition,
                    save_name=os.path.join(save_path + r'\{}\space_plot\prediction'.format(sub_name), save_name),
                    ground_or_pred='P', cluster_entropy=cluster_entropy, tpr=tpr, fpr=fpr)

    # draw reachability plot
    draw_reachability_plot(clust, normalized_feature, ground_truth,
                           save_name=os.path.join(save_path + r'\{}\reachability_plot\ground_truth'.format(sub_name),
                                                  save_name),
                           ground_or_pred='G')
    draw_reachability_plot(clust, normalized_feature, predition,
                           save_name=os.path.join(save_path + r'\{}\reachability_plot\prediction'.format(sub_name),
                                                  save_name),
                           ground_or_pred='P')

    # draw_sdas_suas(clust, clust.sdas, clust.suas,
    #                os.path.join(save_path + r'\{}\reachability_plot\suas_sdas'.format(sub_name), save_name))


def validation_ave_feature(hdf5_file, model_file, ave_feature, target_class, class_num=10):
    # Check if CUDA is available; otherwise, use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # Debugging output
    model = Net()

    model.load_state_dict(
        # torch.load(model_file, map_location='cpu')['model']
        torch.load(model_file, map_location='cpu', weights_only=False)['model']

    )
    # Move model to the appropriate device
    model.to(device)
    model.eval()
    # load validation dataset of feature
    validation_ds = []
    with h5py.File(hdf5_file, "r") as f:
        data = f['{}'.format(target_class)][:, :-1]
        indics = f['{}'.format(target_class)][:, -1]
        ground_truth = indics < 0

        benign_target_class_vector = np.mean(data[ground_truth == 0][:100], axis=0)  # get feature from target class

        for i in range(class_num):
            if i != target_class:
                data = f['{}'.format(i)][:, :-1]
                validation_ds.append(data[:100])  # get feature from other classes

    trigger_vector = ave_feature - benign_target_class_vector
    validation_ds = np.concatenate(validation_ds)
    temp_with_dev = validation_ds + trigger_vector
    temp_with_dev[temp_with_dev<0]=0
    x = model.dropout2(torch.Tensor(temp_with_dev).to(device))
    x = model.fc2(x)

    output = F.softmax(x, dim=1)
    pred = torch.argmax(output, dim=1)
    acc_target = np.sum(pred.cpu().detach().numpy() == target_class) / pred.shape[0]

    return acc_target


def cluster_estimation(hdf5_file, model_file, data, cluster_labels, target_class, class_num):
    acc_target = {}
    for cluster_label in set(cluster_labels):
        if cluster_label != -1: # the noise data is considered as poisoned data
            ave_feature = np.mean(data[cluster_labels==cluster_label], axis=0)
            acc_target[cluster_label] = validation_ave_feature(hdf5_file, model_file, ave_feature, target_class, class_num=class_num)

    return acc_target


def umap_plus_cluster_estimation_eval(hdf5_file, model_file, target_class, class_num):
    nn, min_dist, eps = 5, 0, 0.8
    method_name = 'dbscan'

    with h5py.File(hdf5_file, "r") as f:
        data = f['{}'.format(target_class)][:, :-1]
        indics = f['{}'.format(target_class)][:, -1]
        ground_truth = indics < 0


    reduced_data = umap.UMAP(n_neighbors=nn, min_dist=min_dist).fit_transform(data)
    cache_data = {}
    cache_data['f'], cache_data['l'], cache_data['i'] = reduced_data, ground_truth, indics

    clust = OPTICS(min_samples=20, p=2, cluster_method=method_name, eps=eps).fit(reduced_data)
    labels = clust.labels_
    # for each cluster get the average feature, and calulate the entropy of misclassification
    acc_target_clusters = cluster_estimation(hdf5_file, model_file, data, labels, target_class, class_num)

    pred = np.zeros_like(ground_truth, dtype=float)
    for label in set(labels):
        if label == -1:
            pred[labels == label] = 1 # outlier is considered as poisoned, its acc should be 1
        else:
            pred[labels == label] = acc_target_clusters[label]

    return pred, ground_truth


def calculate_tpr_minus_fpr(cluster_label_dict, cluster_entropy_dict, ground_truth_dict, threshold):
    ave_tpr, ave_fpr = 0, 0
    cnt = 0
    for key in cluster_label_dict.item():
        cluster_label = cluster_label_dict.item()[key]
        cluster_entropy = cluster_entropy_dict.item()[key]
        ground_truth = ground_truth_dict.item()[key]
        pred = np.zeros_like(ground_truth)
        for label in set(cluster_label):
            if label == -1:
                pred[cluster_label==label] = 1
            else:
                if cluster_entropy[label] > threshold:
                    pred[cluster_label==label] = 0
                else:
                    pred[cluster_label==label] = 1

        tpr, tnr = my_tpr_tnr(ground_truth, pred)
        fpr = 1 - tnr
        ave_tpr += tpr
        ave_fpr += fpr
        cnt += 1

    ave_tpr = (1.0 * ave_tpr) / cnt
    ave_fpr = (1.0 * ave_fpr) / cnt

    return ave_tpr - ave_fpr


if __name__ == '__main__':
    print('corrupted label with gu trigger')
    feature_path = 'feature_and_model/corrupted_label_gu_trigger/feature_corrupted_tri_name_gu_10_types-1_target_class_0_poison_ratio_0.0359842836500576.hdf5'
    model_path = 'feature_and_model/corrupted_label_gu_trigger/corrupted_tri_name_gu_10_types-1_target_class_0_poison_ratio_0.0359842836500576.pt'
    target_class = 0
    class_num = 10

    pred, ground_truth = umap_plus_cluster_estimation_eval(feature_path, model_path, target_class, class_num)
    fpr_l, tpr_l, _ = roc_curve(ground_truth, pred,
                                           pos_label=1)
    roc_auc = auc(fpr_l, tpr_l)
    print('AUC {:.3f}'.format(roc_auc))
    print('corrupted label with ramp trigger')
    feature_path = 'feature_and_model/corrupted_label_ramp_trigger/feature_corrupted_tri_name_ramp_target_class_0_poison_ratio_0.0359842836500576.hdf5'
    model_path = 'feature_and_model/corrupted_label_ramp_trigger/corrupted_tri_name_ramp_target_class_0_poison_ratio_0.0359842836500576.pt'
    target_class = 0

    pred, ground_truth = umap_plus_cluster_estimation_eval(feature_path, model_path, target_class, class_num)
    fpr_l, tpr_l, _ = roc_curve(ground_truth, pred,
                                           pos_label=1)
    roc_auc = auc(fpr_l, tpr_l)
    print('AUC {:.3f}'.format(roc_auc))
    print('clean label with gu trigger')
    feature_path = 'feature_and_model/clean_label_gu_trigger/feature_clean_tri_name_gu_10_types-1_target_class_2_poison_ratio_0.3598428365005759.hdf5'
    model_path = 'feature_and_model/clean_label_gu_trigger/clean_tri_name_gu_10_types-1_target_class_2_poison_ratio_0.3598428365005759.pt'
    target_class = 2

    pred, ground_truth = umap_plus_cluster_estimation_eval(feature_path, model_path, target_class, class_num)
    fpr_l, tpr_l, _ = roc_curve(ground_truth, pred,
                                           pos_label=1)
    roc_auc = auc(fpr_l, tpr_l)
    print('AUC {:.3f}'.format(roc_auc))