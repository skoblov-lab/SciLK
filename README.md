# SciLK: a Scientific natural Language Toolkit

This is a special branch dedicated to the CHEMDNER NER model we've submitted for
publication in the Journal of Cheminformatics. We are deliberately going to 
keep this branch stale as to avoid any breaking API issues. It is thus absolutely
safe to use and rely on in the long run. That being said, we are going to 
support this model and in doing so we might eventually patch user-reported bugs. 
You are thus highly encouraged to create issues in the issue tracker, if you
ever run into them.

### Installation guide

This package has a fair bit of dependencies that should be kept isolated in a 
separate Python environment. We absolutely recommend using `conda` for this,
because it will get you Python 3.6 as well as isolate CUDA-related dependencies, 
should you want to install `tensorflow-gpu`. You can create SciLK-ready conda 
environment by issuing this command:

    $ conda create -n scilk-chem python=3.6
    
You can use `virtualenv` just as well, though take note that SciLK requires
Python 3.6 to run properly. After you've have created and activated a new 
Python 3.6 environment issue these commands:

    (scilk-chem) $ git clone -b chemdner-pub https://github.com/skoblov-lab/SciLK.git
    (scilk-chem) $ pip install --no-cache-dir Scilk/

The second one will install SciLK along with all dependencies. Take note,
that this will get you a CPU-only version of tensorflow. If you are going to
use a GPU, you should remove this version of TF:
     
    (scilk-chem) $ pip uninstall tensorflow
    
and install tensorflow-gpu 1.4.1 (e.g. from `conda` channels). It's
worth pointing out, that you should only consider this option if you want to 
train your own models, because a GPU won't have a significant impact on your
inference speed. If for whatever reason you want to manually install some/all 
dependencies, here they are:

    numpy==1.14.0
    h5py==2.7.1
    fn==0.4.3
    pyrsistent==0.14.2
    scikit-learn==0.19.1
    pandas==0.22.0
    hypothesis==3.56.6
    frozendict==1.2
    joblib==0.11
    tensorflow==1.4.1  # or a GPU-version
    keras==2.1.3
    binpacking==1.3


### Pretrained models

A collection of best-performing pretrained models is available 
[here](https://www.dropbox.com/s/5z4jqlbjgo15m59/chemdner-collection.tgz?dl=0).
You can then refer to `notebooks/usage-demo.ipynb` for a step-by-step guide on
loading and using the models.

### Training new models

You will find all the usage guides you might need in the `notebooks` subdirectory.
Take note, that almost all function used in the core package are heavily
documented and type-annotated.