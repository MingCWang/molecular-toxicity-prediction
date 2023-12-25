## Project 3
Long Nguyen, Ming-Shih Wang

### Introduction:
This project aims to fine-tune and optimize a method to generate accurate predictions for the toxicity of molecules in the Tox21 dataset. Although we spent the majority of our time using data manipulation for data imbalances, fine-tuning, and trying different methods and models, due to time constraints, we were unable to find a way to overcome the hurdle of a 0.7855% validation score.

### Contributions
Long Nguyen, Ming-Shih Wang: data preprocessing, model fine-tuning, different-model searching, report, training and testing

### Preprocess The Data

#### Balancing Each Property Individually:
The data was extremely imbalanced — the 0:1 proportion was close to 0.95% for some, and so we focused much of our time on accounting for this, as this was a likely contributor to the overfitting of our model. The key is to resample in a way that improves the representation of underrepresented labels without significantly distorting the distribution of other labels. Here is our approach. 

Given this is a multi-label classification problem, the distribution of the 12 labels in each data entry is dependent on each other; Which means that simply replicating the minority class would not produce a balanced distribution. Therefore, through trial and error, we found the best combination of data to replicate, which produced fairly balanced data as shown in the graphs below.

                 	Left: Before oversampling 				  Right: After oversampling

#### Integrating SMILES and Molecular Data
Another issue we thought was the cause of overfitting, in addition to the “12-task” imbalance was the lack of data. To account for this, we tried adding more features using the rdkit to utilize SMILES and Molecular data. 




By using RDKit and MolVS libraries, the code standardizes molecules from their SMILES representation and extracts their parent structure. Then, it generates Morgan fingerprints, a form of circular fingerprint that encapsulates the molecular structure. These fingerprints are integrated into our dataset by expanding them to match the graph representation of each molecule, thereby augmenting the original node features. This enrichment with detailed chemical information aims to provide a more comprehensive dataset, potentially enhancing the predictive accuracy of our models in tasks such as molecular property prediction. However, this caused more overfitting, quickly going from 68% to 84%, to 93% within 3 epochs for the training ROC-AUC score — and leaving the validation score still to be around 75%.

#### Weighted Loss Function:
To account for data imbalance, we also calculated the weights in each label class respectively. 
The original loss calculation simply converts the two-dimensional toxicity label dataset into a huge single-dimension array while excluding all the nan values. This would not work if class weights were applied, we were experiencing dimension mismatches because of the missing values. To account for the dimension difference, because the labels are in a binary format that represents the absence of the property associated with the label, I replaced nan values with 0. 
After class weights calculation, I applied them to a custom loss calculation function using the BCEWithLogitsLoss()function that works better with binary values.

#### Model Optimization and Fine-Tuning:
Needless to say, our optimizations did not stop in preprocessing, as we implemented a random search for tuning combinations of layers and hyperparameters. 

- Models: GraphConv, GATConv, Regular Neural Network, GNN, GCNConv, BatchNorm, ReLu
- Hyperparameters: dropout_rate, fingerprint_dim, num_classes, hidden_classes, num_rdkit_features

However, we continued to hit a plateau of around 78.98% with these models and combinations of layers and hyperparameter tuning. 
In the end, we used the model shown above, with a combination of Graph Convolutional Layers, Batch Normalizations, Linears, and layers that utilize the fingerprint data. The most optimal hyperparameters were found to be: 

hidden_channels = 32, dropout_rate = 0.6, fingerprint_dim = 8192, num_node_features = 9

### Conclusion and Takeaways
All in all, this project was much harder than expected. Admittedly, if there was more time, we would spend more of it understanding the fundamentals of the complex data, as all the efforts of preprocessing with the imbalances, data concatenation, smiles, and molecular data, and all the model and hyperparameter optimization only resulted in a negligible increase in AUC-ROC score. Ultimately, we learned that the APIs of Pytorch, Keras, and scikit-learn make it accessible and quite easy to implement machine learning models and techniques, truly understanding the data, the intricacies of how models work is crucial in achieving a high success rate.

