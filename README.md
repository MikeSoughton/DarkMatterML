# DarkMatterML
Repository for the code associated with the paper 'Using Machine Learning to disentangle LHC signatures of Dark Matter candidates', [arXiv:1910.06058](https://arxiv.org/abs/1910.06058) (2019). Authors: Charanjit K. Khosa, Veronica Sanz and Michael Soughton.

## Overview
Looking for evidence of Dark Matter is of great importance to the state of High Energy Physics (HEP). Mono-X searches within particle colliders such as the Large Hadron Collider (LHC) are one area where potential Dark Matter particles may be discovered. Our paper explores some of these searches within the LHC and attempts to distinguish between different Dark Matter candidates using Machine Learning. The signals which we consider are **AAAA** . Note that all the samples we consider are pure signal - our task is to characterise between diffeerent potential signals *after* their (theoretical) discovery has been made - although we do consider the how these signals look compared to the background in the appendix.

We first examine using a logistic regression algorithm trained on raw kinematic data for each event. The algorithm is trained for a binary classification task two distinguish between any two signals. We then apply the trained algorithm to new testing data to evaluate its performance. We repeat this for each combination of signals. To determine how well the algorithm can distinguish between different signal types we plot ROC curves for each classification task. We next do the same but using a DNN over the same data instead, the code for which produces similar plots.

We then look at engineering the feature inputs to see how that affect the performance of a classifier. We sample a number $r$ events and produce a 2D histogram in ($\eta$ - $p_T$) space from those events, thereby creating an 'image' which can be fed into a classifier. There will be $N_\text{images} = N_\text{tot}/r$ 'images' to be fed into the classifier, where $N_\text{tot}$ is the total number of events. We have thus decreased the size of the training data but increased the information available within in each training input. We expect then that *if there is still a sufficient amount of 'images' for training*, then the performance may increase. The classifer is now no longer looking at the data on an event-by-event basis but rather at a number of events at a time. There is potentially more power in this approach, although a trade-off can arise as the amount training data can decrease. We also do this in the ideal scenario where we are only working with the pure signals. When using real data without truth labels, you may not know whether an image contains only pure signal or if it also contains background. This could potentially distort the classificatiom, but is not a fundamental problem.

We train a DNN with these 2D histograms and evaluate it over new data, producing ROC curves. We do this for parton-level monojet signals as well as dijet detector-level monojet signals, whereupon we utilse a PCA to find the two most important features to construct histograms from. We then train a CNN with the histograms for the same monojet and dijet signals. We plot ROC curves for both methods, finding that the DNN using 2D histograms performs best.

We finally consider how a statistical criterion $N_\text{SUSY}/N_\text{ALP}$ can be used to set a bound on the ALP cross-section, given some value for the SUSY cross-section, where $N_\text{SUSY}$ is the number of SUSY histograms correctly identified by the classifier and $N_\text{ALP}$ is the number of ALP histograms correctly identified by the classifier. We produce plots to demonstrate this.

Events are generated through [`MadGraph`](https://arxiv.org/abs/1106.0522) along with [`Pythia`](https://arxiv.org/abs/0710.3820) and [`Delphes`](https://arxiv.org/abs/1307.6346) for showering and detector effects. 

## Dependencies

The code is run in `python2.7.17`. The following packages are required:

```
numpy==1.16.5
scipy==1.2.2
tensorflow==2.0.0
scikit-learn==0.20.3
matplotlib==2.2.4
```

These can be installed manually or via the conda yaml file using
```
conda env create --name <env name> -f environment.yml
```
or
```
conda create --name <env_name> --file requirements.txt
```
Note that I would reccomend using the second option as the version of conda used may have problems using the yaml file.

## Code layout

Tensorflow etc..

The paper finds ...

The code does ...

Prerequisities ...

The code is run ...

References, funding and additional info ...

This project was made possible through funding from the Royal Society-SERB Newton International Fellowship (to C.K.K.) and the Science Technology and Facilities Council (STFC) under grant number (to V.S. and M.S.).

## Running the code

### Logistic regression and DNN using kinematic features
The logistic regression and DNN using kinematic features we ran from the command line so we do not provide instructions here, but if you are interested you can follow the instructions in `hepML-master` which utilise the package [hepML](https://github.com/aelwood/hepML). The results obtained as well as scripts to produce ROC plots are stored within `susy-backgd` and `susy-othersig`. 

### DNN and CNN using 2D histograms on monojet data
When running the DNN or CNN using 2D histograms there are a couple of options. To run both the DNN and the CNN using histograms constructed from monojet parton-level data navigate to `monojet`. To train and evaluate the algorithm on the various signals against each other, run
```
python 2D_ROC_curves_sig-sig.py
```
to train and evaluate the algorithm on the various signals against each other. To train and evaluate the algorithm on the various signals against the background run 
```
python 2D_ROC_curves.py
```
These scripts read in the data stored within the directory, train the DNN and CNN classifiers and save the ROC plots produced when using test data.

One can also perform the same analysis using monojet detector-level data within `monojet-delphes-updated` by running the scripts with the same names, but that now use the detector-level data.

### DNN and CNN using 2D histograms on dijet data
To run both the DNN and the CNN using histograms constructed from dijet detector-level data navigate to `dijet-delphes-updated`. The scripts here are structured in the same way as for the monojet case. To train and evaluate the algorithm on the various signals against each other, run
```
python 2D_ROC_curves_sig-sig.py
```
to train and evaluate the algorithm on the various signals against each other. To train and evaluate the algorithm on the various signals against the background run 
```
python 2D_ROC_curves.py
```

There are also scripts to run the PCA analysis used in the paper as well as a script for producing histogram plots for demonstration purposes, however they do not effect the running of the main code.

**Bold**

[Link](https://www.wikipedia.org)

## Section

A section can be referenced through [Section](#section)






## Citation
Please cite the paper as follows in your publications if it helps your research:

@article{Khosa:2019kxd,
    author = "Khosa, Charanjit Kaur and Sanz, Veronica and Soughton, Michael",
    title = "{Using machine learning to disentangle LHC signatures of Dark Matter candidates}",
    eprint = "1910.06058",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    doi = "10.21468/SciPostPhys.10.6.151",
    journal = "SciPost Phys.",
    volume = "10",
    number = "6",
    pages = "151",
    year = "2021"
}

