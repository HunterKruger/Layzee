# Layzee

This library is aimed to enhance data scientists' daily work efficiency. 
<br><br>
Main functions are listed below:
+ dataframe_observer.py: exploratory data analysis
+ splitter_sampler.py: a simpler way for splitting and sampling datasets
+ feature_handling.py: feature engineering for training set and/or test set
+ feature_generation.py: several feature generation methods
+ feature_reduction.py: for feature reduction, including filtering and embedded methods
+ feature_drift: visualize and detect feature drift between training set and test set
+ modeling.py: fast modeling for binary/multiclass classification and regression tasks, including model validation & hyper-parameter searching  
+ evaluation.py: interpret model result in different aspects
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
    $ pip3 install .
```
* To uninstall, run `$ pip3 uninstall layzee`
