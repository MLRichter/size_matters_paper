# PHD-LAB Experimental Environment for Saturation based Experiments

## Introduction
The phd lab repository contains routines for training of networks, extraction of latent representations,
saturation computation and other experimental and probe training.
For ease of reproduction we will first show you how to reproduce results depicted in the paper or
the appendix.
This readme also provides install instructions as well as a more detailed look at the entire repository in order to fine tune
and expand upon the experiments if necessary.

## Reproducing results from the paper
This section gives you a brief tutorial and the necessary commands
that allow you to reproduce the results of the paper.
Please note that any experiment may take multiple weeks to conclude, depending
on the used models, dataset and postprocessing.

### Prerequisits

#### Hardware
The experiments were conducted on a System with 2 RTX6000 GPUs, 128GB of RAM
and 56 CPU Cores in total.
For most experiments we recommend at least 18 Cores, 32 GB of DDR4 Memory and a GPU with 10GB of VRAM.
We also recommend at least 1TB of free hard drive space, which is used for storing the activation maps of all layers
in order to compute probe performances.


#### Time
Depending on the experiments training can run for multiple days or weeks.
However, given our hardware, in many cases experiments will conclude in less than a day.
The experimental setup will checkpoint the models after each epoch.
Probe performance calculations are also cached on a probe by probe basis.

#### Datasets
The experiments make use of the following datasets:
* Cifar10
* MNIST
* TinyImageNet
* iNaturalist
* ImageNet

Cifar10 and MNIST are downloaded automatically in a tmp-Folder inside
this repository. The path should`allways be ```./phd_lab/tmp/${DATASET}```.
If this folder does not exist, it is created.
Any other dataset is exspected to downloaded manually and placed in the aforementioned folder
TinyImageNet and iNaturalist make use of the ImageFolder-Dataset implementation
by PyTorch. This means that the folder structure should sort them by class.
ImageNet (like MNIST and Cifar10) makes use of the respective dataset-preimplementations
implemented by PyTorch.


#### Overall workflow
The scripts called in the instructions below handle training of models and probes as well as computation of saturation, receptive field a.s.o.
In essence, the script provides the raw data than can be made into the respective plot.
The notebooks consume the raw data as saved by the scripts and will reproduce the plots.
Note that each non-boilerplate cell in the notebooks is responsible for a single plot or a set of plots.
This means in order for a non-boilerplate cell to plot without crashing the exact experiment or set of experiments is required as specified in the cell.

### Section 2

#### Figure 1
* train the model, compute saturation and probe performances by running:
````shell script
PYTHONPATH=".." python probe_meta_execution_script.py --config ../configs/section2/figure1.json -mp 18 -d 4 --device cuda:0 --run-id RES

````
* You can then use the ```size_matters_plots_1.ipynb``` notebook to generate the requested plot

### Section 3.1

#### Table 1

* train the model, compute saturation and probe performances by running:
````shell script
PYTHONPATH=".." python train_model.py --config ../configs/section3.1/table1.json --device cuda:0 --run-id RES
PYTHONPATH=".." python train_model.py --config ../configs/section3.1/table1_2.json --device cuda:0 --run-id RES

````
* Use the CSV-files in the logs of the respective models to obtain the test accuracies at their final epochs.
* Divide the accuracies by the accuracy of the model of the control group (trained on the 224 pixel images without upscaling) to obtain the relative accuracies.

### Section 3.2

#### Figure 2
* train the model, compute saturation and probe performances by running:
````shell script
PYTHONPATH=".." python probe_meta_execution_script.py --config ../configs/section3.2/figure2.json -mp 18 -d 4 --device cuda:0 --run-id RES

````
* You can then use the ```size_matters_plots_1.ipynb``` notebook to generate the requested plot


### Section 3.3

#### Figure 3
* train the model, compute saturation and probe performances by running:
````shell script
PYTHONPATH=".." python probe_meta_execution_script.py --config ../configs/section3.3/figure3.json -mp 18 -d 4 --device cuda:0 --run-id RES

````
* You can then use the ```size_matters_plots_1.ipynb``` notebook to generate the requested plot


### Section 3.4

#### Figure 4
* train the model, compute saturation and probe performances by running:
````shell script
PYTHONPATH=".." python probe_meta_execution_script.py --config ../configs/section3.4/figure4.json -mp 18 -d 4 --device cuda:0 --run-id RES
````
* Compute the receptive field of the layers by running:
````shell script
PYTHONPATH=".." python compute_receptive_field.py --config ../configs/section3.4/figure4.json --device cuda:0 --run-id RES
````
* You can then use the ```size_matters_plots_3.ipynb``` notebook to generate the requested plot.

#### Figure 5
* train the model, compute saturation and probe performances by running:
````shell script
PYTHONPATH=".." python probe_meta_execution_script.py --config ../configs/section3.4/figure5.json -mp 18 -d 4 --device cuda:0 --run-id RES
````
* Compute the receptive field of the layers by running:
````shell script
PYTHONPATH=".." python compute_receptive_field.py --config ../configs/section3.4/figure5.json --device cuda:0 --run-id RES
````
* You can then use the ```size_matters_plots_3.ipynb``` notebook to generate the requested plot


#### Figure 6
* train the model, compute saturation and probe performances by running:
````shell script
PYTHONPATH=".." python pixelwise_probe_meta_execution_script.py --config ../configs/section3.4/figure6.json -mp 56 --device cuda:0 --run-id RES
````

* You can then use the ```size_matters_plots_2.ipynb``` notebook to generate the requested plot.
* Computing the probes consumes a lot of disk space, when you are done we recommend removing the ```latent_representation``` folder.

### Section 3.5

#### Figure 7
* train the model, compute saturation and probe performances by running:
````shell script
PYTHONPATH=".." python probe_meta_execution_script.py --config ../configs/section3.5/figure7.json -mp 18 -d 4 --device cuda:0 --run-id RES
````
* Compute the receptive field of the layers by running:
````shell script
PYTHONPATH=".." python compute_receptive_field.py --config ../configs/section3.5/figure7.json --device cuda:0 --run-id RES
````
* You can then use the ```size_matters_plots_3.ipynb``` notebook to generate the requested plot


#### Figure 8
* train the model, compute saturation and probe performances by running:
````shell script
PYTHONPATH=".." python probe_meta_execution_script.py --config ../configs/section3.5/figure8.json -mp 18 -d 4 --device cuda:0 --run-id RES
````
* Compute the receptive field of the layers by running:
````shell script
PYTHONPATH=".." python compute_receptive_field.py --config ../configs/section3.5/figure8.json --device cuda:0 --run-id RES
````
* You can then use the ```size_matters_plots_3.ipynb``` notebook to generate the requested plot


### Appendix

#### Figure 9
* train the model, compute saturation and probe performances by running:
````shell script
PYTHONPATH=".." python probe_meta_execution_script.py --config ../configs/appendix/figure9.json -mp 18 -d 4 --device cuda:0 --run-id RES

````
* You can then use the ```size_matters_plots_1.ipynb``` notebook to generate the requested plot

#### Figure 10
* train the model, compute saturation and probe performances by running:
````shell script
PYTHONPATH=".." python probe_meta_execution_script.py --config ../configs/appendix/figure10.json -mp 18 -d 4 --device cuda:0 --run-id RES

````
* You can then use the ```size_matters_plots_1.ipynb``` notebook to generate the requested plot

#### Figure 11
* train the model, compute saturation and probe performances by running:
````shell script
PYTHONPATH=".." python probe_meta_execution_script.py --config ../configs/appendix/figure11.json -mp 54 -d 4 --device cuda:0 --run-id RES

````
* You can then use the ```size_matters_plots_1.ipynb``` notebook to generate the requested plot

#### Figure 12
* train the model, compute saturation and probe performances by running:
````shell script
PYTHONPATH=".." python probe_meta_execution_script.py --config ../configs/appendix/figure12.json -mp 18 -d 4 --device cuda:0 --run-id RES

````
* You can then use the ```size_matters_plots_1.ipynb``` notebook to generate the requested plot

#### Figure 13
* train the model, compute saturation and probe performances by running:
````shell script
PYTHONPATH=".." python probe_meta_execution_script.py --config ../configs/appendix/figure13.json -mp 18 -d 4 --device cuda:0 --run-id RES

````
* You can then use the ```size_matters_plots_1.ipynb``` notebook to generate the requested plot

#### Figure 14
* train the model, compute saturation and probe performances by running:
````shell script
PYTHONPATH=".." python probe_meta_execution_script.py --config ../configs/appendix/figure14.json -mp 18 -d 4 --device cuda:0 --run-id RES

````
* You can then use the ```size_matters_plots_1.ipynb``` notebook to generate the requested plot

#### Figure 15
* train the model, compute saturation and probe performances by running:
````shell script
PYTHONPATH=".." python probe_meta_execution_script.py --config ../configs/appendix/figure15.json -mp 18 -d 4 --device cuda:0 --run-id RES
````
* Compute the receptive field of the layers by running:
````shell script
PYTHONPATH=".." python compute_receptive_field.py --config ../configs/appendix/figure15.json --device cuda:0 --run-id RES
````
* You can then use the ```size_matters_plots_3.ipynb``` notebook to generate the requested plot

#### Figure 16
* train the model, compute saturation and probe performances by running:
````shell script
PYTHONPATH=".." python probe_meta_execution_script.py --config ../configs/appendix/figure16.json -mp 18 -d 4 --device cuda:0 --run-id RES
````
* Compute the receptive field of the layers by running:
````shell script
PYTHONPATH=".." python compute_receptive_field.py --config ../configs/appendix/figure16.json --device cuda:0 --run-id RES
````
* You can then use the ```size_matters_plots_3.ipynb``` notebook to generate the requested plot

#### Figure 17
* train the model, compute saturation and probe performances by running:
````shell script
PYTHONPATH=".." python probe_meta_execution_script.py --config ../configs/appendix/figure17.json -mp 18 -d 4 --device cuda:0 --run-id RES
PYTHONPATH=".." python pixelwise_probe_meta_execution_script.py --config ../configs/appendix/figure17.json -mp 54 --device cuda:0 --run-id RES

````
* You can then use the ```size_matters_plots_1.ipynb``` and ```size_matters_plots_2.ipynb``` notebook to generate the requested plot

#### Figure 18
* train the model, compute saturation and probe performances by running:
````shell script
PYTHONPATH=".." python probe_meta_execution_script.py --config ../configs/appendix/figure18.json -mp 18 -d 4 --device cuda:0 --run-id RES
PYTHONPATH=".." python pixelwise_probe_meta_execution_script.py --config ../configs/appendix/figure18.json -mp 54 --device cuda:0 --run-id RES

````
* You can then use the ```size_matters_plots_1.ipynb``` and ```size_matters_plots_2.ipynb``` notebook to generate the requested plot

#### Figure 19
* train the model, compute saturation and probe performances by running:
````shell script
PYTHONPATH=".." python probe_meta_execution_script.py --config ../configs/appendix/figure19.json -mp 18 -d 4 --device cuda:0 --run-id RES
PYTHONPATH=".." python pixelwise_probe_meta_execution_script.py --config ../configs/appendix/figure19.json -mp 54 --device cuda:0 --run-id RES

````
* Compute the receptive field of the layers by running:
````shell script
PYTHONPATH=".." python compute_receptive_field.py --config ../configs/appendix/figure19.json --device cuda:0 --run-id RES
````
* You can then use the ```size_matters_plots_3.ipynb``` and ```size_matters_plots_2.ipynb``` notebook to generate the requested plot






### Troubleshooting and known Issues
* Probe performances are saved not in the order of the models sequential structure. For unknown reasons, this is a windows issue.
Reordering needs to be done manually. Use the names of the layers to sort them correctly.
The plots will still run in this scenario, but the probe performances will look obviously out of order and wrong.
* Failed to Converge after End of an Epoch: This is an issue with the eigenvalue computation algorithm on the GPU. There are
three possible fixes. Sometimes a simple rerun of the epoch (by just restarting the script) helps.
If that does not help deleting the logs of the model and starting from scratch and changing the batch size slighty will resolve the issue.
This rarely happens to TinyImageNet and iNaturalist. The bug was reported to PyTorch.
* Extremely slow probe computation: This is a common issue if the ```-mp``` flag was set to either to many or to few cores.
Ideally the value should match the highest number of layers in your experimental setup OR the number of cores you have available.
This assumes that you have 100GB or more of RAM, which is sufficent for conducting the experiments.
* Do NOT attempt to extract latent representations from models trained on iNaturalist and ImageNet if you want results quickly.
TinyImageNet may take up to 2 days to compute probe performances and the aforementioned datasets
will be worse by at least a factor of 10.

## <a name="install"></a>Installing phd-lab

Phd-lab is written in python. It uses several third-party moduls which
have to be installed in order to run the experiments. The following
two sections provide installation instructions.

### <a name="pip"></a>Installation with pip

The file `requirements.txt` can be used to install all requirements
using `pip` into new, virtual environment (called `phd-lab-env`):
```sh
python3 -m venv phd-lab-env
source phd-lab-env/bin/activate
pip3 install -r requirements.txt
```

Remarks:
* it seems not possible to install `delve` with Python 3.5.2
(the version installed at the IKW)
* at our institute, torch complains that the NVIDIA driver is too old
(found version 10010). However, there seems to be no way to upgrade
this with pip. In this situation you may resort to the 
[conda installation](#conda).
* if not needed anymore, the virtual environment can be deleted by typing
  `rm -R phd-lab-env/`.

### <a name="conda"></a>Installation with conda

When using `conda`, you can use the file `environment.yml` to set up a
new conda environment caled `phd-lab`, containing all required packages:
```sh
conda env create -f environment.yml
conda activate phd-lab
```

Remarks:
* if no longer needed, the environment can be removed by typing
`conda remove --name phd-lab --all`
* at the institute of cognitive science (IKW), the currently installed
  nvidia driver (418.67) allows at best CUDA toolkit vesrion 10.1.
  Use the file `environment-ikw.yml` instead of `environment.yml`
  for an adapted environment.
* To check if torch can use your CUDA version, you can run the following command:
```sh
python -c "import torch; print(torch.cuda.is_available())"
```
  some (but not all) versions of torch also support:
```sh
python -c "import torch; print(torch._C._cuda_isDriverSufficient())"
```

## Configure your Experiments
Models are configures using json-Files. The json files are collected in the ./configs
folder.
```json
{
    "model": ["resnet18", "vgg13", "myNetwork"],
    "epoch": [30],
    "batch_size": [128],

    "dataset": ["Cifar10", "ImageNet"],
    "resolution": [32, 224],

    "optimizer": ["adam", "radam"],
    "metrics": ["Accuracy", "Top5Accuracy", "MCC"],

    "logs_dir": "./logs/",
    "device": "cuda:0",

    "conv_method": ["channelwise"],
    "delta": [0.99],
    "data_parallel": false,
    "downsampling": null
}
```
Note that some elements are written as lists and some are not. 
A config can desribe an arbitrary number of experiments, where the number 
of experiments is the number of possible value combinations. The only 
exception from this rule are the metrics, which are allways provided as a list 
and are used all during every experiment.
In the above example, we train 3 models on 2 datasets using 2 optimizers. 
This result in 3x2x2=12 total experiments.
It is not necessary to set all these parameters everytime. If a parameter is not 
set a default value will be injected.
You can inspect the default value of all configuration keys in ``phd_lab.experiments.utils.config.DEFAULT_CONFIG``.

### <a name="log"></a>Logging
Logging is done in a folder structure. The root folder of the logs is specified 
in ``logs_dir`` of the config file.
The system has the follow save structure

```
+-- logs
|   +-- MyModel
|   |   +-- MyDataset1_64                                               //dataset name followed by input resolution
|   |   |   +-- MyRun                                                   //id of this specific run
|   |   |   |   +--  probe_performance.csv                              //if you compute probe performances this file is added containing accuracies per layer, you may add a prefix to this file
|   |   |   |   +--  projected_results.csv                              //if you projected the networks
|   |   |   |   +--  computational_info.json                            //train_model.py will compute some meta info on FLOPS per inference step and save it as json
|   |   |   |   +--  MyModel-MyDataset1-r64-bs128-e30_config.json       //lets repeat this specific run
|   |   |   |   +--  MyModel-MyDataset1-r64-bs128-e30.csv               //saturation and metrics
|   |   |   |   +--  MyModel-MyDataset1-r64-bs128-e30.pt                //model, lr-scheduler and optimizer states
|   |   |   |   +--  MyModel-MyDataset1-r64-bs128-e30lsat_epoch0.png    //plots of saturation and intrinsic dimensionality
|   |   |   |   +--  MyModel-MyDataset1-r64-bs128-e30lsat_epoch1.png   
|   |   |   |   +--  MyModel-MyDataset1-r64-bs128-e30lsat_epoch2.png    
|   |   |   |   +--  .                                              
|   |   |   |   +--  .                                             
|   |   |   |   +--  .                                              
|   +-- VGG16
|   |   +-- Cifar10_32
.   .   .   .   .     
.   .   .   .   .   
.   .   .   .   .
```

The only exception from this logging structure are the latent
representation, which will be dumped in the folder
`latentent_datasets` in the top level of this repository. The reason
for this is the size of the latent representation on the hard
drive. You likely want to keep your light-weight csv-results in the
logs, but may want to remove extracted latent representations on a
regular basis to free up space.  (They can be
[reextracted from the saved model](#extract) quite easily,
so it's not even a time loss realy)

## Running Experiments

Execution of experiments is fairly straight forward. You can easily
write scripts if you want to deviate from the out-of-the-box
configurations (more on that later).  In the ``phd_lab`` folder you
will find scripts handling different kinds of model training and
analysis.

### <a name="train"></a>Training models
There are 4 overall scripts that will conduct a training if called. It is worth noting that 
each script is calling the same Main-Functionn object in just a slightly different configuration.
They therefore share the same command line arguments and basic execution logic.
The scripts are:
+ ``train_model.py`` train models and compute saturation along the way, adds also a json with information about FLOPs required per image and training
+ ``infer_with_altering_delta.py`` same as train.py, after trainingg concluded the model es evaluated on the test set, while changing the delta-value of all PCA layers.
+ ``extract_latent_representations.py`` extract the latent representation of the train and test set after training has concluded.
+ ``compute_receptive_field.py`` compute the receptive field after training. Non-sequential models must implemented ``noskip: bool`` argument in their consstructor in order for this to work properly.
+ ``probes_meta_execution_script.py`` basically ``extract_latent_representations.py`` and ``train_probes.py`` combined into one file for easier handling. 

All of these scripts have the same arguments:
+ ``--config`` path to the config.json
+ ``--device`` compute device for the model ``cuda:n`` for the nth gpu,``cpu`` for cpu
+ ``--run-id`` the id of the run, may be any string, all experiments of this config will be saved in a subfolder with this id. This is useful if you want to repeat experiments multiple times.

Additionally ``extract_latent_representations.py`` has an additional argument:
+ ``--downsampling`` target height and width of the downsampled feature map. Default value is 4. Adaptive Average Pooling is used for downsampling. In case of ``probes_meta_execution_script.py`` this argument is called ``-d`` instead and may be used multiple times to train probed multiple times on various resolutions.
+ ``--prefix`` if set, the content of this argument will be added as a prefix infront of the filename, separated by underscore. For example``foo_probe_performance.csv``.

#### Checkpointing
All metrics and the model itself are checkpointed after each epoch and the previous weights are overwritten.
The system will automatically resume training at the end of the last finished epoch.
If one or more trainings were completed, these trainings are skipped.
Please note that post-training actions like the extractions of latent representations will still be executed.
Furthermore runs are identified by their run-id. Runs under different run-ids generally do not recognize each other, even if they
are based on the same configuration.

### <a name="extract"></a>Extracting latent representations
Latent representations for an experiment (a specific model and dataset)
can be obtained by the script `extract_latent_representations.py`.

```sh
python extract_latent_representations.py --config ./configs/myconfig.json --device cuda:0 --run-id MyRun --downsample 4
```
The script expects the usual parameters `--config`, `--device`, 
and `--run-id`, and the following additional value:
* `--downsample`: 


This script will feed the full dataset through the model and store
the observed activation patterns for each layer. The data are
stored in the directory `latent_datasets/[experiment]/` and
the files are called `[train|eval]-[layername].p`
```
+-- latent_datasets/
|   +-- ResNet18_XXS_Cifar10_32/
|   |   +-- eval-layer1-0-conv1.p
|   |   +-- eval-layer1-0-conv2.p
|   |   +-- ...
|   |   +-- model_pointer.txt
|   |   +-- train-layer1-0-conv1.p
|   |   +-- train-layer1-0-conv2.p
|   |   +-- ...
.   .
.   .
.   .
```
The `.p` are pickle files containing numpy arrays with the latent
representations.
The file `model_pointer.txt` contains the path to the log files.


### <a name="probe"></a>Probe Classifiers and Latent Representation Extraction
Another operation that is possible with this repository is training
probe classifiers on receptive fields.  Probe Classifiers are
LogisticRegression models. They are trained on the output of a neural
network layer using the original labels.  The performance relative to
the model performance yields an intermediate solution quality for the
trained model.  After [extracting the latent representations](#extract)
you can train the probe classifiers on the latent representation by calling
```sh
python train_probes.py --config ./configs/myconfig.json --prefix "SomePrefix" -mp 4
```

The script `train_probes.py` can take the following arguments:
+ ``--config`` the config the original experiments were conducted on
+ ``-f`` the root folder of the latent representation storage is by default``./latent_representation``
+ ``-mp`` the number of processes spawned by this script. By default
the number of processes equal to the number of cores on your cpu. Note
that the parallelization is done over the number of layers, therefore
more processes than layers will not yield any performance benefits.

The performance of the probe classifiers in stored in the [log
directory](#log) under the name `probe_performances.csv`.

The system uses joblist chaching and will recognize whether a logistic
regression has allready been fitted on a particular latent
representation and skip training if it has, making crash recovery less
painful.


### Using consecutive script calls of scripts to split your workload
All experiments are strictly tied to the run-id and their
configuration. This means that two trained models are considered equal
if they are trained using the same configuration parameters and
run-id, regardless of the called script.  There for you could for
instance run:

```sh
python train_model.py --config ./configs/myconfig.json --device cuda:0 --run-id MyRun
```

followed by 

```sh
python compute_receptive_field.py --config ./configs/myconfig.json --device cuda:0 --run-id MyRun
```

the latter script call will recognize the previously trained models
and just skip to computing the receptive field and add the additional
results to the logs.

### Performing multiple tasks using the meta execution script

You can combine the steps of [train a model](#train), 
[extract pixelwise latent representation](#extract) and 
[train probes](#probe)
using the script `probe_meta_execution_script.py`:
```sh
python probe_meta_execution_script.py --config ../configs/your_config.json -mp ${NumberOfCoresYouWantToUse} -d pixelwise --device cuda:0 --run-id ${YourRunID}
```
* `--config `
* `-d pixelwise`:


## Adding Models / Datasets / Optimizers
You may want to add optimizer, models and datasets to this experimental setup. Basically there is a package for each of
these ingredientes:
+ ````phd_lab.datasets````
+ ````phd_lab.models````
+ ````phd_lab.optimizers````
+ ````phd_lab.metrics````

You can add datasets, model, metrics and optimizers by importing the respective factories in the ````__init__```` file of the respective
packages.
The interfaces for the respective factories are defines as protocols in ````phd_lab.experiments.domain```` or you can
simply orient yourself on the existing once in the package.
If you want to use entirely different registries for datasets, models, optimizers and metrics you can change registry 
by setting different values for:
+ ````phd_lab.experiments.utils.config.MODEL_REGISTRY````
+ ````phd_lab.experiments.utils.config.DATASET_REGISTRY````
+ ````phd_lab.experiments.utils.config.OPTIMIZER_REGISTRY````
+ ````phd_lab.experiments.utils.config.METRICS_REGISTRY````

These registries do not need to be Module or Package-Types, they merely need to have a ````__dict__```` that maps string keys
to the respective factories.
The name in the config file must allways match a factory in order to be a valid configuration.
