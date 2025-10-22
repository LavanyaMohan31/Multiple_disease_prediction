# Multiple_disease_prediction

Install the required libraries using pip install and saved in requirement.txt

Creating seperate IPYNB file for disease prediction of
         Kidney 
         Liver
         parkinson


Loading the data from google sheet to dataframe

Data Investigation * shape and size * checking for null values * duplicates * value counts

Data cleaning - dropping unwanted columns

Seperating Feature and Target

In Feature , seperating numerical and categorical column

Feature selection * Doing EDA to find relation between feature and target * chi - square test for categorical * 2-sample T-test for numerical * Dropping the unwanted features and selecting the required ones

Feature Scaling and Encoding * standard Scaling - numeric * One-Hot Encoding - categorical

Pipeline creation Smote - to balance the data

Model Creation - trying with different models Selecting the best model and dump it as pickle file

Creating the dashboard using streamlit Predict the model in streamlit with user input
