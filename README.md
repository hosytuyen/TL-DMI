# Model Inversion Robustness: Can Transfer Learning help?

## 1. Setup Environment
This code has been tested with Python 3.7, PyTorch 1.11.0 and Cuda 11.3. 

```
conda create -n MI python=3.7

conda activate MI

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt
```

## 2. Prepare Dataset & Checkpoints

- We follow GMI/KEDMI/LOMMA to prepare dataset
- Please find the checkpoint in the [release tag](https://github.com/hosytuyen/TL-DMI/releases/tag/v1.0)
  
* Extract and place the two folders at `.\datasets` and `.\checkpoints`
  

## 3. Training the classifier with Transfer Learning scheme

- Modify the configuration in `.\config\celeba\classify.json`. To train classifier with our TL scheme. You need to modify as below:
  * `root_path`: path to the saved folder
  * `dataset`: 
    * `model_name`: VGG16
  * `VGG16`:
    * `epochs`: number of training epochs
    * `lr`: learning rate
    * `resume`: checkpoints
    * `freeze_layers`: freeze_layers index that starts unfreeze. For example, if `freeze_layers = 10`, it means we will finetune layers 10 afterward.

  
- Then, run the following command line to get the target model
  ```
  python train_classifier.py
  ```



## 4. Training GAN (Optinal)

SOTA MI attacks work with a general GAN, therefore. However, Inversion-Specific GANs help improve the attack accuracy. In this repo, we provide codes for both training general GAN and Inversion-Specific GAN.

### 4.1. Build a inversion-specific GAN 
* Modify the configuration in
  * `./config/celeba/training_GAN/specific_gan/celeba.json` if training a Inversion-Specific GAN on CelebA
  * `./config/celeba/training_GAN/specific_gan/ffhq.json` if training a Inversion-Specific GAN on FFHQ
  
* Then, run the following command line to get the Inversion-Specific GAN
    ```
    python train_gan.py --configs path/to/config.json --mode "specific"
    ```

### 4.2. Build a general GAN 
* Modify the configuration in
  * `./config/celeba/training_GAN/general_gan/celeba.json` if training a Inversion-Specific GAN on CelebA
  * `./config/celeba/training_GAN/general_gan/ffhq.json` if training a Inversion-Specific GAN on FFHQ
  
* Then, run the following command line to get the General GAN
    ```
    python train_gan.py --configs path/to/config.json --mode "general"
    ```

Pretrained general GAN and Inversion-Specific GAN can be downloaded at https://drive.google.com/drive/folders/1_oyT_JMBym_jse5HcoivFSv4GkpFN5Nz?usp=share_link


## 5. Learn augmented models
We provide code to train augmented models (i.e., `efficientnet_b0`, `efficientnet_b1`, and `efficientnet_b3`) from a ***target model***.
* Modify the configuration in
  * `./config/celeba/training_augmodel/celeba.json` if training an augmented model on CelebA
  * `./config/celeba/training_augmodel/ffhq.json` if training an augmented model on FFHQ
  
* Then, run the following command line to get the General GAN
    ```
    python train_gan.py --configs path/to/config.json
    ```

Pretrained augmented models can be downloaded at https://drive.google.com/drive/folders/12Ib5N9jRkApaVFrUu33S4nexlJwZuCoJ?usp=share_link


## 6. Model Inversion Attack

* Modify the configuration in
  * `./config/celeba/attacking/celeba.json` if training an augmented model on CelebA
  * `./config/celeba/attacking/ffhq.json` if training an augmented model on FFHQ

* Important arguments:
  * `method`: select the method either ***gmi*** or ***kedmi***
  * `variant` select the variant either ***baseline***, ***L_aug***, ***L_logit***, or ***ours***

* Then, run the following command line to attack
    ```
    python recovery.py --configs path/to/config.json
    ```

## 7. Evaluation

After attack, use the same configuration file to run the following command line to get the result:\
```
python evaluation.py --configs path/to/config.json
```



## Reference
<a id="1">[1]</a> 
Zhang, Yuheng, et al. "The secret revealer: Generative model-inversion attacks against deep neural networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.


<a id="2">[2]</a>  Chen, Si, et al. "Knowledge-enriched distributional model inversion attacks." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

<a id="3">[3]</a>  Nguyen, Ngoc-Bao, et al. "Re-thinking Model Inversion Attacks Against Deep Neural Networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
