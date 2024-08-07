# Centralized and Federated Heart Disease Classification Models Using UCI Dataset and their Shapley-value Based Intepretability
The objective of this work is to extend the existing federated learning benchmark and dataset suite FLamby (https://github.com/owkin/FLamby), specifically for the UCI heart disease dataset (https://archive.ics.uci.edu/dataset/45/heart+disease). In addition, we provide Shapley value interpretability analysis to find the features in the data most contributing to heart disease presence. We feed the centralized dataset to several binary classification models, which are hyperparameter tuned and run on 10 different seeds, and federate the best performers. We compare our results to the results obtained by the default model used in FLamby (logistic regression, using a learning rate of 0.001 and batch size of 4). We find that the SVM model performs best in both central and federated settings, achieving test accuracies of 83.3% and 73.8%, respectively. Our interpretability analysis indicates that the three most influential features in the data are also cited in medical literature as being strong indicators of heart disease or failure, providing credence to our findings. The paper can be found here: 

A secondary objective of this work is to address the misuse of duplicated data by other researchers in the heart disease classification task. Several papers use a modified version of the Cleveland dataset, obtained on Kaggle (https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset), which contains more than 700 duplicated data points, resulting in highly inflated accuracies. The results obtained in this work use the unmodified dataset, and we only perform standard preprocessing such as removing missing values and binarizing the labels. 

The code for the seven tested centrally-trained binary classification models are provided ("model"_heartdisease.py), as well as updated model.py, metric.py,  and loss.py that incorporate the federated 1LNN and SVM models. The requirements for running these codes are in Requirements.pdf. The centrally-trained models can be run as is with the necessary packages installed, which is even easier on Google Colab. 
