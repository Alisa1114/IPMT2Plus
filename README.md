# IPMT++: Improving Few-shot Semantic Segmentation with Contrastive Learning
This repo contains the code for our "*IPMT++: Improving Few-shot Semantic Segmentation with Contrastive Learning*" by Ching Chen, Hsin Lung Wu.

<p align="middle">
  <img src="figure/few_shot_contrastive_learning-separate.png">
</p>
<p align="middle">
  <img src="figure/few_shot_contrastive_learning-combine.png">
</p>


## Dependencies

- Python 3.8
- PyTorch 1.7.0
- cuda 11.0
- torchvision 0.8.1
- tensorboardX 2.14
- cython
- tqdm
- PyYaml
- opencv-python
- pycocotools

## Build Dependencies
```
cd model/ops/
bash make.sh
cd ../../
```

## Datasets

- PASCAL-5<sup>i</sup>:  [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) + [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)

- COCO-20<sup>i</sup>:  [COCO2014](https://cocodataset.org/#download)

```
${IPMT2Plus}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- instances_train2014.json
        |   `-- instances_val2014.json
        |-- train2014
        |   |-- 000000000009.jpg
        |   |-- 000000000025.jpg
        |   |-- 000000000030.jpg
        |   |-- ... 
        `-- val2014
            |-- 000000000139.jpg
            |-- 000000000285.jpg
            |-- 000000000632.jpg
            |-- ... 
```

Then, run  
```
python prepare_coco_data.py
```
to prepare COCO-20^i data.

## Training and Testing

- **Step1** *download backbone models*

  Download the ImageNet pretrained [**backbones**](https://mega.nz/folder/BrhgwSJK#-gSKptOu5G7cWHbkJyvShg) and put them into the `initmodel` directory.

- **Step2** *choose training methods*

  We use git branch to change the training method scripts. We have *two-stage* and *single-stage*.
  ```
  git checkout training_method
  ```

- **Step3** *setting the config*

  Change configuration via the `.yaml` files in `config`.
  
- **Step4** *Training*

  Run for contrastive learning.
  ```
  python train_contrastive.py --config /path/to/config_file
  ```

  Run for few-shot training. For two-stage training, you should put the path to your model weight that is trained from contrastive learning in the `weight` in config files before few-shot training.
  ```
  python train.py --config /path/to/config_file
  ```

- **Step5** *Testing*

  Run for testing.
  ```
  python test.py --config /path/to/config_file
  ```

## Visualization

<p align="middle">
    <img src="figure/fss-vis.png">
</p>

## References

This repo is mainly built based on [CyCTR-PyTorch](https://github.com/YanFangCS/CyCTR-Pytorch) and [IPMT](https://github.com/LIUYUANWEI98/IPMT). Thanks for their great work!
