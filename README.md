# Efficient Image Inpainting using Deep Generative Models

Using the Instance Generation method for Image Inpainting in a limited data setting.

Checkpoints drive - [Google Drive](https://drive.google.com/drive/folders/1N6SkpKG9JjuT5GKuxInMUA3sQeQQQT5B?usp=sharing)

Refer to this directory to download FFHQ - https://github.com/NVlabs/ffhq-dataset

Refer to this directory to download ArtBench - https://github.com/liaopeiyuan/artbench

Model implementations - pix2pix.py and CEGAN.py

## Training InsGen
- Please refer to the Google Drive above for checkpoints
- run `train_CL.py`, with the following commands

```
usage: train_CL.py [-h] [--checkpoint CHECKPOINT] [--epochs EPOCHS] [--momentum MOMENTUM] [--lw_fake_cl_on_g LW_FAKE_CL_ON_G] [--lw_real_cl LW_REAL_CL] [--lw_fake_cl LW_FAKE_CL]
                   [--dataset {ffhq,artbench}] [--partition PARTITION]

training Params

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        restore training from checkpoint
  --epochs EPOCHS       number of training epochs
  --momentum MOMENTUM   momentum to update d_ema
  --lw_fake_cl_on_g LW_FAKE_CL_ON_G
                        weight for gen cl_loss
  --lw_real_cl LW_REAL_CL
                        weight for real instance disc
  --lw_fake_cl LW_FAKE_CL
                        weight for fake instance disc
  --dataset {ffhq,artbench}
                        dataset to run on, default ffhq
  --partition PARTITION
                        dataset partition, default 100
```

## Training Context Encoder GAN
- Please refer to the Google Drive above for checkpoints
- run `train_CEGAN.py`, with the following commands

```
usage: train_CEGAN.py [-h] [--checkpoint CHECKPOINT] [--epochs EPOCHS] [--partition PARTITION]

training Params

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        restore training from checkpoint
  --epochs EPOCHS       number of training epochs
  --partition PARTITION
                        dataset partition
```

Contributors
- Kushagra Agrawal
- Rajeswari Mahapatra
