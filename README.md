# Layzee

This library is aimed to enhance data scientists' daily work efficiency. 
<br><br>
Main functions are listed below:
+ dataframe_observer.py: exploratory data analysis
+ feature_handling.py: feature engineering for one dataset 
+ feature_handling2.py: feature engineering for training set and test set
+ feature_generation.py: several feature generation methods
+ feature_reduction.py: feature reduction by embedding methods
+ feature_drift: visualize and detect feature drift between training set and test set
+ modeling.py: fast modeling for binary/multiclass classification or regression tasks, including model validation & hyper-parameter searching  
+ evaluation.py: evaluate inference result in different aspects
+ splitter_samples: split or sample datasets
<br><br>
  
The following notebooks are quick tutorials for supervised learning:
+ test_modeling_bin.ipynb: binary classification
+ test_modeling_mlt.ipynb: multiclass classification
+ test_modeling_reg.ipynb: regression

### Pip Installation Guide
* Download the package
* [optional] Create a virtual env and activate it
* In terminal, run 
```bash
    $ cd ./Layzee
    $ pip3 install .  ##or##  python setup.py install
```
* To uninstall, run `$ pip3 uninstall layzee`
