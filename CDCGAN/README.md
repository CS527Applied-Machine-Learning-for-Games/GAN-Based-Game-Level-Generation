# Conditional DCGAN (CDCGAN)

This directory contains our Conditional Deep Convolutional GAN architecture implementation where we condition the generator on the previously generated game frame.

The GAN was implemented using PyTorch and the model files can be found in `model` subdirectory. We have also saved our best trained models for generating Mario game levels in the `trained_model` directory.

## Data

The mario level data is stored in `levels.json` file. It contains `14 x 28 tile_unit` frames of Super Mario Bros 2 levels.

To train our CDCGAN we split every frame in half, to use first half as condition and second as output. This is achieved in `MarioDataset` class in `data_loader.py`.

## Instructions for training your own GAN

To train the GAN you can run the following command in this directory.

```
python train.py
```

If you want to use cuda then append the `--cuda` argument to the above.


## Generating Levels using trained model

To generate actual mario levels using a trained model you can utilize the `get_level.py`
script. 
For pointing it to your trained model you'll need to change the path parameter in 
```
netG.load_state_dict(torch.load("YOUR_MODEL_PATH"))
```

Make sure you've installed our `image_gen` package included in the repository by running 
```
pip install -e .
```
while in `.ImageGenerators/image_gen` directory.

# Latent Space Exploration (LSE) using CMA-ES

We perform Latent Space Exploration on the noise input to our Conditional DCGAN. The idea is to learn a mapping between certain areas of this noise to certain features. 

Our implementation of CMA-ES can be found in `cmaes.py`. 

## Fitness Functions
Covariance matrix adaptation evolution strategy (CMA-ES) utilized a fitness function to judge how good a certain member of the population is. 

`lse.py` contains a hand crafted fitness function `matrix_eval` that gathers numerous tile frequencies in certain regions of the level, and uses these metrics to optimize for different structural features in the level. For eg. increasing the `sky-tile` count. 

We also have implemented a **KL Divergence** evaluation between two levels based on the tile distributions they contain. This can be found in the `KLTileDistribution` class in `level_kl.py` file. 
You can set the `window_size` parameter for determining tile distribution in the level. 

## Instructions to perform LSE

To perform LSE on a trained CDCGAN run
```
python lse.py
```
This defaults to `1000` population members per iteration, you can configure that by changing the `population_size` param in `__main__`.
You can also configure the level samples that are generated per member by setting `samples_per_member` parameter. 

If you want to define a custom fitness function, then simply define a new function which takes in the generated level as a `numpy array` and returns a custom fitness measure for the level.

## Instructions to Orchestrate Generation of level with feature-noise params

After performing LSE to find the best params for certain features, we can then orchestrate the level generation by running `demo_lse.py`, which loads the saved noise params with the trained generator and then presents the user with a menu to choose the features they want in next `frame_count=6` frames of the level.

Command to run:

```
python demo_lse.py
```