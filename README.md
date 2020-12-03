# Conditional DCGAN (CDCGAN)

This directory contains our Conditional Deep Convolutional GAN architecture implementation where we condition the generator on the previously generated game frame.

The GAN was implemented using PyTorch and the model files can be found in `CDCGAN/model/` subdirectory. We have also saved our best trained models for generating Mario game levels in the `CDCGAN/trained_model/` directory.

## Data

The mario level data is stored in `CDCGAN/levels.json` file. It contains `14 x 28 tile_unit` frames of Super Mario Bros 2 levels.

To train our CDCGAN we split every frame in half, to use first half as condition and second as output. This is achieved in `MarioDataset` class in `CDCGAN/data_loader.py`.

## Instructions for training your own GAN

To train the GAN you can run the following command in `./CDCGAN` directory.

```
python train.py
```

If you want to use cuda then append the `--cuda` argument to the above.


## Generating Levels using trained model

To generate actual mario levels using a trained model you can utilize the `CDCGAN/get_level.py`
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

# Multi-Stage GAN (MSGAN)

This directory contains implementations of our Multi-Stage GAN architectures. The implementation is in TensorFlow and the trained models are in the `MSGAN/models/` directory.

## Data

The data files are present in the `MSGAN/` directory. `example_super_mario_bros_1.json`, `example_super_mario_bros_2.json`, and `example_super_mario_bros_2_cond.json` form the train data for the MSGAN (Original) architecture. `smb1_left.json`, `smb1_right.json`, and `smb1_str_right.json` form the train data for the MSGAN (Combined) architecture with 1:1 Condition:Generation ratio. `smb1_left_2.json`, `smb1_right_2.json`, and `smb1_str_right_2.json` form the train data for the MSGAN (Combined) architecture with 3:1 Condition:Generation ratio.

## Instructions for training your own GAN

Each notebook in the `MSGAN/` directory corresponds to a specific GAN architecture. `GAN_Level_Generation_Structure.ipynb`, `GAN_Level_Generation_Structure_Combined.ipynb`, and `GAN_Level_Generation_Color.ipynb` correspond the MSGAN (Original) Stage-1, MSGAN (Combined) Stage-1, and MSGAN Stage-2 architectures respectively. To train and generate your own models, simply run the necessary notebooks with the required data files (parts where you need to change the path to the data files are clearly marked with `# TODO` prompts).

`GAN_Level_Generation.ipynb` contains our implementation of the baseline architecture outlined in:

*V. Volz, J. Schrum, J. Liu, S. M. Lucas, A. Smith, and S. Risi, "Evolving mario levels in the latent space of a deep convolutional generative adversarial network," in Proceedings of the Genetic and Evolutionary Computation Conference, 2018, pp. 221–228.*

## Generating Levels using trained model

To generate your own levels, use the `GAN_Level_Generation_Results.ipynb` in the `MSGAN/` directory. Load the necessary trained models into the `generator_saved_structure` and `generator_saved_color` variables and run the `generate_level()` function with the loaded models.

# Latent Space Exploration (LSE) using CMA-ES

We perform Latent Space Exploration on the noise input to our Conditional DCGAN. The idea is to learn a mapping between certain areas of this noise to certain features. 

Our implementation of CMA-ES can be found in `CDCGAN/cmaes.py`. 

## Fitness Functions
Covariance matrix adaptation evolution strategy (CMA-ES) utilized a fitness function to judge how good a certain member of the population is. 

`CDCGAN/lse.py` contains a hand crafted fitness function `matrix_eval` that gathers numerous tile frequencies in certain regions of the level, and uses these metrics to optimize for different structural features in the level. For eg. increasing the `sky-tile` count. 

We also have implemented a **KL Divergence** evaluation between two levels based on the tile distributions they contain. This can be found in the `KLTileDistribution` class in `CDCGAN/level_kl.py` file. 
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
