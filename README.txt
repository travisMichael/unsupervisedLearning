# Unsupervised Learning

This project explores several unsupervised learning algorithms to classify cardiovascular disease. The project also explores several unsupervised learning algorithms to classify whether a financial loan is good or bad for the company giving out the loan.

The project consists of multiple python programs that require python 3.6.8 or above. The project contains the following python dependencies (numpy, sklearn, matplotlib, and pickle).

The project can be cloned by running the following command:

https://github.com/travisMichael/unsupervisedLearning.git

Before running any of the programs in this project, download the datasets from the following locations. Both datasets are owned by Kaggle, so you must be a Kaggle member to download them. Signing up for Kaggle is free. Make sure to place both datasets inside the parent directory of this project.

https://www.kaggle.com/wendykan/lending-club-loan-data

https://www.kaggle.com/sulianova/cardiovascular-disease-dataset

Once both datasets have been placed inside the project, then the pre-processing programs can be run for each dataset. To run the pre-processing programs, run the following commands:

python data_transform.py cardio

python data_transform.py loan

Once the pre-processing programs have completed, then the programs to train the models can be run. To train the models, run the following commands:

To generate the plots from the analysis paper, run the following command:

python generate.py figure_1_and_2

python generate.py table_1_and_2_and_3_and_4

python generate.py table_5

python generate.py table_6

python generate.py figure_3

python generate.py figure_4_and_5

python generate.py table_7

python generate.py figure_6

python generate.py table_8
