# PharmAI 2

This project aims to detect anomalies in pharmacological profiles using machine learning.

This is a cleaner, simpler rewrite of PharmAI 1 and should be easier to follow.

---

## Motivation

Health-system pharmacists review almost all medication orders for hospitalized patients. Considering that most orders contain no errors, especially in the era of CPOE with CDS,<sup>[1](https://doi.org/10.2146/ajhp060617)</sup> pharmacists have called for technology to enable the triage of routine orders to less extensive review, in order to focus pharmacist attention on unusual orders.<sup>[2](https://doi.org/10.2146/ajhp070671),[3](https://doi.org/10.2146/ajhp080410),[4](https://doi.org/10.2146/ajhp090095)</sup>


## Files

Due to privacy concerns, we are unable to provide the data necessary to run the code in this repository. To run the code, three data files are required.

The first should be the pharmacological profiles data file. This file should be a pickle file containing a dictionary where keys are years and values are lists of pharmacological profiles, each profile being a list of drug identifiers, ideally strings without spaces.

The second file should be the departments data file, which is necessary to analyze results by departement. This file should be a pickle file containing a dictionary where the keys are years and values are lists of departments, corresponding by index to the pharmacological profiles in the first file.

The third file should be the departement categorization file. This file allows the departments to be grouped into patient population categories. This file should be a CSV file with semi-colon separators (`;`) and two columns: `orig_depa` which should be the department ids as listed in the second file, and `cat_depa` which should be the population label.

### Training files (train_basic.py, train_autoenc.py, train_ganomaly.py)

These files are used to train models. Cross-validation can be performed by configuring training and validation folds by year in the `Years to use` section of the parameters. If no validation years are provided, a single training range of years should be provided and a trained model will be saved. If cross-validation is performed, a results log in the form of a CSV file as well as tensorboard logs (for neural networks) will be saved.

### cross_val_results_basic.py

This file is used to plot the cross-validation results of the basic machine learning models obtained from the `train_basic.py` file. The cross-validation logs are used to generate a graph of the ratio of anomalies for selected departements on the validation set for a range of parameters, as well as a graph of the explained variance ratio for a range of TSVD componends in the latent semantic indexing part of the pipeline.

### Training years files (training_years_basic_graph.py, training_years_ganomaly_graph.py)

These files use the cross-validation logs to generate graphs of the validation results for a range of training years.

### Evaluation files (evaluate_basic.py, evaluate_ganomaly.py)

These files are used to evaluate the final models on the test set.

### Evaluation graph giles (evaluation_graphs_basic.py, evaluation_graphs_ganomaly.py)

These file plot graphs from the test set results.

# Prerequisites

Developed using Python 3.7

Requires:

- Joblib
- Numpy
- Pandas
- Scikit-learn
- Scikit-image
- Matplotlib
- Seaborn
- UMAP
- TQDM
- Tensorflow 2.3

# Contributors

Maxime Thibault.

# References

Paper currently under peer review.  
Abstract describing the GANomaly model presented at the Machine Learning for Healthcare 2020 conference  
[Abstract](https://static1.squarespace.com/static/59d5ac1780bd5ef9c396eda6/t/5f245e305efe21770a14b204/1596218928943/112CameraReadySubmission20200722+Clinical+Abstract+MLHC+2020+CAMERA+READY+FINAL.pdf)
[Spotlight presentation](https://www.youtube.com/watch?v=TNT2jOyMaYs&list=PLRqwW7v078faPwD53NgpuDhKq2j3qDgUq&index=23)
[Poster](https://bit.ly/2CWtBrq)

# License

GNU GPL v3

Copyright (C) 2020 Maxime Thibault

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
