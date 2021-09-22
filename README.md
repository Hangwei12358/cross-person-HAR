# cross-person-HAR
Code for our GILE model, which is proposed in AAAI-2021 paper "Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition".



### Abstract
>In wearable-sensor-based activity recognition, it is often assumed that the training and the test samples follow the same data distribution. This assumption neglects practical scenarios where the activity patterns inevitably vary from person to person. To solve this problem, transfer learning and domain adaptation approaches are often leveraged to reduce the gaps between different participants. Nevertheless, these approaches require additional information (i.e., labeled or unlabeled data, meta-information) from the target domain during the training stage. In this paper, we introduce a novel method named Generalizable Independent Latent Excitation (GILE) for human activity recognition, which greatly enhances the cross-person generalization capability of the model. Our proposed method is superior to existing methods in the sense that it does not require any access to the target domain information. Besides, this novel model can be directly applied to various target domains without re-training or fine-tuning. Specifically, the proposed model learns to automatically disentangle domain-agnostic and domain-specific features, the former of which are expected to be invariant across various persons. To further remove correlations between the two types of features, a novel Independent Excitation mechanism is incorporated in the latent feature space. Comprehensive experimental evaluations are conducted on three benchmark datasets to demonstrate the superiority of the proposed method over state-of-the-art solutions.



## Datasets Preparation
Three datasets (UCIHAR, Opportunity, and UniMiB SHAR) are utilized in the experiments. The pre-processed outcome can be downloaded from [HERE](https://drive.google.com/drive/folders/1S4oGTs8ChD8ezmxOrnqcboWVWHvT7CdH?usp=sharing). Please save datasets under folder `./data`. The `data_preprocess_[dataset-name].py` can conduct data-preprocessing automatically. 


## Environment Setup
Create an environment named `gile` in Anaconda and install necessary packages. 
```shell script
conda create --name gile python=3.6
conda activate gile
conda install -c pytorch pytorch=1.3
conda install -c conda-forge matplotlib
conda install -c conda-forge scikit-learn
conda install -c conda-forge tqdm
conda install -c pytorch torchvision
```

## Usage
Specify the target domain when running the experiment. For example, 
```python
python main_ucihar.py --beta_d 5.0 --beta_y 5.0 --target_domain 0 
# the opportunity dataset is the largest dataset, which may take longer running time
python main_oppor.py --target_domain S1
python main_shar.py --target_domain 3
```



 If you find any of the codes helpful, kindly cite our paper. 

> ```
>@inproceedings{DBLP:conf/aaai/QianPM21,
> author    = {Hangwei Qian and
>               Sinno Jialin Pan and
>               Chunyan Miao},
>  title     = {Latent Independent Excitation for Generalizable Sensor-based Cross-Person
>               Activity Recognition},
>  booktitle = {Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI}
>               2021, Thirty-Third Conference on Innovative Applications of Artificial
>               Intelligence, {IAAI} 2021, The Eleventh Symposium on Educational Advances
>               in Artificial Intelligence, {EAAI} 2021, Virtual Event, February 2-9,
>               2021},
>  pages     = {11921--11929},
>  publisher = {{AAAI} Press},
>  year      = {2021},
>  url       = {https://ojs.aaai.org/index.php/AAAI/article/view/17416},
>  timestamp = {Sat, 05 Jun 2021 18:11:55 +0200},
>  biburl    = {https://dblp.org/rec/conf/aaai/QianPM21.bib},
>  bibsource = {dblp computer science bibliography, https://dblp.org}
>}
> ```

