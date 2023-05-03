# DarkMatterML
Repository for the code associated with the paper 'Using Machine Learning to disentangle LHC signatures of Dark Matter candidates', [arXiv:1910.06058](https://arxiv.org/abs/1910.06058) (2019). Authors: Charanjit K. Khosa, Veronica Sanz and Michael Soughton.

## Overview
Looking for evidence of Dark Matter is of great importance to the state of High Energy Physics (HEP). Mono-X searches within particle colliders such as the Large Hadron Collider (LHC) are one area where potential Dark Matter particles may be discovered. Our paper explores some of these searches within the LHC and attempts to distinguish between different Dark Matter candidates using Machine Learning. The signals which we consider are  . Note that all the samples we consider are pure signal - our task is to characterise between diffeerent potential signals *after* their (theoretical) discovery has been made - although we do consider the how these signals look compared to the background in the appendix.

We first examine using a logistic regression algorithm trained on raw kinematic data for each event. The algorithm is trained for a binary classification task two distinguish between any two signals. We then apply the trained algorithm to new testing data to evaluate its performance. We repeat this for each combination of signals. To determine how well the algorithm can distinguish between different signal types we plot ROC curves for each classification task. We next do the same but using a DNN over the same data instead, the code for which produces similar plots.

We then look at engineering the feature inputs to see how that affect the performance of a classifier. We sample a number $r$ events and produce a 2D histogram in ($\eta$ - $p_T$) space from those events, thereby creating an 'image' which can be fed into a classifier. There will be $N_\text{images} = N_\text{tot}/r$ 'images' to be fed into the classifier, where $N_\text{tot}$ is the total number of events. We have thus decreased the size of the training data but increased the information available within in each training input. We expect then that *if there is still a sufficient amount of 'images' for training*, then the performance may increase. The classifer is now no longer looking at the data on an event-by-event basis but 

Events are generated through [`MadGraph`](https://arxiv.org/abs/1106.0522) along with [`Pythia`](https://arxiv.org/abs/0710.3820) and [`Delphes`](https://arxiv.org/abs/1307.6346) for showering and detector effects. 

## Dependencies

The code is run in `pythonXXX`. The following packages are required:

```

```

These can be installed manually or via the conda yaml file using

```
conda env create --name <env name> -f environment.yml
```

## Code layout

Tensorflow etc..

The paper finds ...

The code does ...

Prerequisities ...

The code is run ...

References, funding and additional info ...

This project was made possible through funding from the Royal Society-SERB Newton International Fellowship (to C.K.K.) and the Science Technology and Facilities Council (STFC) under grant number (to V.S. and M.S.).

**Bold**

[Link](https://www.wikipedia.org)

## Section

A section can be referenced through [Section](#section)






## Citation
Please cite the paper as follows in your publications if it helps your research:

    @article{Khosa:2019kxd,
      author         = "Khosa, Charanjit K. and Sanz, Veronica and Soughton,
                        Michael",
      title          = "{WIMPs or else? Using Machine Learning to disentangle LHC
                        signatures}",
      year           = "2019",
      eprint         = "1910.06058",
      archivePrefix  = "arXiv",
      primaryClass   = "hep-ph",
      SLACcitation   = "%%CITATION = ARXIV:1910.06058;%%"
    }

## License
