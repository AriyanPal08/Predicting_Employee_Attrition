# Employee Attrition Prediction

This project explores employee data and builds supervised classification models to estimate whether an employee is likely to leave the organisation. The target variable is `Attrition`, where `Yes` means the employee left and `No` means the employee did not leave.

The analysis is organised as one reproducible Jupyter notebook. It uses a fixed `random_state=42` whenever a process involves randomness.

## Project workflow

### Task 1 - Data understanding and exploration

Load the employee dataset and examine its structure before making any modelling decisions.

- Report dataset dimensions, columns, and feature data types.
- Examine the `Attrition` class distribution.
- Check for missing values and fully duplicated rows.
- Produce numerical and categorical summary statistics.
- Identify constant-value columns and explore selected relationships with attrition.
- Record observations that are calculated directly from the dataset.

### Task 2 - Data preprocessing

Prepare the dataset for future model development without training a model.

- Confirm that there are no missing values.
- Remove constant columns and the employee identifier because they are not useful predictive inputs.
- Encode `Attrition` as a binary target (`No` = 0, `Yes` = 1).
- One-hot encode nominal categorical predictor columns.
- Standard-scale numerical predictor columns for scale-sensitive algorithms.
- Create an 80/20 stratified train-test split with `random_state=42`.
- Fit preprocessing transformations on the training set only to prevent data leakage.

### Task 3 - Model development

Train at least four classification models on the preprocessed training data.

- Use clearly documented hyperparameters for each model.
- Use `random_state=42` for every model or procedure that supports randomness.
- Generate class predictions and probability scores on the test set.
- Keep the same train-test split for every model so that results are comparable.

### Task 4 - Model evaluation

Evaluate every trained model using metrics that are appropriate for the imbalanced attrition target.

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Full confusion matrix

Present the results in one comparison table and discuss the strengths, limitations, and appropriate use cases of each model. Accuracy alone is not used to judge model quality.

### Task 5 - Model selection and reflection

Select a final model only after reviewing the evidence from Task 4.

- Justify the recommendation using precision, recall, F1-score, ROC-AUC, interpretability, and practical deployment needs.
- Discuss potential overfitting or underfitting using training and test behaviour.
- Suggest realistic improvements that could be investigated in a later iteration.

## Repository structure

```text
.
|-- Task_1_Data_Understanding_and_Exploration.ipynb
|-- WA_Fn-UseC_-HR-Employee-Attrition.csv  # local dataset; ignored by Git
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## Setup and run

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place the employee attrition CSV file beside the notebook.
4. Start Jupyter and run the notebook cells in order:

   ```bash
   jupyter notebook
   ```

The dataset file is intentionally excluded from version control. Do not add credentials, API keys, or other secrets to the notebook or repository.
