# Universal Detection of Backdoor Attacks via Density-based Clustering and Centroids Analysis

In this paper, we propose a Universal Defence based on Clustering and Centroids Analysis (CCA-UD) against backdoor attacks. The goal of the proposed defence is to reveal whether a Deep Neural Network model is subject to a backdoor attack by inspecting the training dataset. CCA-UD first clusters the samples of the training set by means of density-based clustering. Then, it applies a novel strategy to detect the presence of poisoned clusters. The proposed strategy is based on a general misclassification behaviour obtained when the features of a representative example of the analysed cluster are added to benign samples. The capability of inducing a misclassification error is a general characteristic of poisoned samples, hence the proposed defence is attack-agnostic. This masks a significant difference with respect to existing defences, that, either can defend against only some types of backdoor attacks, or are effective only when some conditions on the poisoning ratios adopted by the attacker or the kind of triggering pattern used by the attacker are satisfied. Experiments carried out on several classification tasks and network architectures, considering different types of backdoor attacks, that either corrupt or do not corrupt the labels of the poisoned samples, and triggering signal, including both global and local triggering patterns, as well as sample-specific trigger, reveal that the proposed method is very effective to defend against backdoor attacks in all the cases, always outperforming the state of the art techniques.

This is the implementation of the paper:
~~~
@misc{guo2023universal,
      title={Universal Detection of Backdoor Attacks via Density-based Clustering and Centroids Analysis}, 
      author={Wei Guo and Benedetta Tondi and Mauro Barni},
      year={2023},
      eprint={2301.04554},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
~~~
Download PDF from [ArXiv](https://arxiv.org/abs/2301.04554). 

## Installation

Use the provided *environment.yml* to build the conda environment, then activate it:
~~~
# for linux user
conda env create -f environment.yml
# activate env
conda activate CCAUD_env
~~~

## Backdoored model and corresponding features of poisoned data
Please download the feature hdf5 and model file from this link https://drive.google.com/file/d/1a8P5RzAIqeOC8XPHc2hDXSBePFlJo18Q/view?usp=sharing

After unzip files into 'feature_and_model' folder, you could find three sub-folders

1. The 'clean_label_gu_trigger' folder includes the backdoor model 'clean_tri_name_gu_10_types-1_target_class_2_poison_ratio_0.3598428365005759.pt' 
generated via: clean-label poisoning strategy with gu trigger, target class as 2 and poisoning ratio 0.3598428365005759.
The corresponding features are stored in 'feature_clean_tri_name_gu_10_types-1_target_class_2_poison_ratio_0.3598428365005759.hdf5'

2. The 'corrupted_label_gu_trigger' folder includes the backdoor model 'corrupted_tri_name_gu_10_types-1_target_class_0_poison_ratio_0.0359842836500576.pt'
generated via: corrupted-label posioning strategy with gu trigger, target class as 0 and poisoning ratio 0.0359842836500576.
THe corresponding features are stored in 'feature_corrupted_tri_name_gu_10_types-1_target_class_0_poison_ratio_0.0359842836500576.hdf5'

3. The 'corrupted_label_ramp_trigger' folder includes the backdoor model 'corrupted_tri_name_ramp_target_class_0_poison_ratio_0.0359842836500576.pt'
generated via: corrupted-label poisoning strategy with ramp trigger, target class as 0 and poisoning ratio 0.0359842836500576
The corresponding features are stored in 'feature_corrupted_tri_name_ramp_target_class_0_poison_ratio_0.0359842836500576.hdf5'

More details please check our paper.

## Running test code
```
python CCA-UD.py
```