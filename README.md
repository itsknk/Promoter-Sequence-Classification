## DNA Sequence Classification

This project demonstrates a comprehensive approach to classifying DNA sequences using various machine learning models. Specifically, the dataset used is the promoter gene sequences obtained from the UCI Machine Learning Repository.

### Introduction

The goal of this project is to classify DNA sequences, specifically identifying promoter regions in DNA. Promoters are sequences of DNA that initiate transcription of a particular gene, playing a crucial role in gene expression. Correct identification of promoters is significant for understanding gene regulation and function.

### Dataset

The dataset, `promoters.data`, is sourced from the UCI Machine Learning Repository. It contains 106 DNA sequences, each 57 nucleotides long, along with their class labels indicating whether they are promoters or non-promoters. The dataset can be accessed [here](https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data).

### Data Preprocessing

1. **Loading Data**: The data is loaded into a Pandas DataFrame, with columns for class, sequence ID, and the DNA sequence.
2. **Cleaning**: The sequences contain tab characters, which are removed for consistency.
3. **Dictionary Creation**: Each sequence is converted into a list of nucleotides, with the class label appended to the end. This structure is transformed into a DataFrame.

### Feature Engineering

To prepare the data for machine learning models:

1. **Label Encoding**: Class labels are converted to numerical format using `LabelEncoder`.
2. **One-Hot Encoding**: Each nucleotide position in the sequences is one-hot encoded, creating binary features for each nucleotide at each position.

### Models Used

A variety of machine learning models were employed to classify the DNA sequences:

- RandomForestClassifier
- AdaBoostClassifier
- GradientBoostingClassifier
- LogisticRegression
- XGBClassifier (XGBoost)
- LGBMClassifier (LightGBM)
- CatBoostClassifier
- MLPClassifier (Neural Network)

### Evaluation Metrics

The models were evaluated using several metrics:

- Accuracy
- Cross-Validation Accuracy
- F1 Score
- Precision
- Recall

Confusion matrices, ROC curves, and Precision-Recall curves were also generated for a more in-depth evaluation.

### Results

The performance of each model is summarized in a table showing training time, test time, accuracy, F1 score, precision, and recall.

### Visualizations

- **Confusion Matrices**: Display the performance of each model in distinguishing between promoters and non-promoters.
- **ROC Curves**: Illustrate the trade-off between true positive rate and false positive rate.
- **Precision-Recall Curves**: Show the trade-off between precision and recall for different thresholds.

### Feature Importance

The feature importance for the RandomForestClassifier was plotted to identify which nucleotide positions are most significant in predicting promoter regions.

### Ensemble Learning

An ensemble model using soft voting was created by combining the predictions of all the individual models. This ensemble model showed improved accuracy over individual models.

### Prediction on New Sequence

An example prediction was demonstrated using a new DNA sequence, showcasing the practical application of the trained model.

### How to Run

1. **Clone the Repository**: Clone the repository to your local machine.
    ```bash
    git clone https://github.com/itsknk/Promoter-Sequence-Classification.git
    ```
2. **Install Dependencies**: Install the required Python packages.
3. **Download Dataset**: Ensure the `promoters.data` file is located in the appropriate directory.
4. **Run the Notebook**: Execute the Jupyter Notebook to preprocess data, train models, and generate results.
    ```bash
    jupyter notebook classify.ipynb
    ```
### License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/itsknk/Promoter-Sequence-Classification/blob/master/LICENSE) file for details.
