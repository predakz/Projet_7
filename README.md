# Projet_7# Projet_7

Project: deploy an online scoring app in order to predict a client's ability to repay a loan and decide if the loaan can be granted.

Data: https://www.kaggle.com/c/home-credit-default-risk/data

In this folder:
-> notebook_P7.ipynb is the jupyter notebook containing all the preprocessing operations applied to data and the training of different models including the one chosen for the API
-> mlruns contains most important runs logged by MLflow while training the models (open a terminal and use the command "mlflow ui" to visualize their informations)
-> Pickles contain Python objects including the model (called "randfor.pkl") and tools for data preprocessing or generating visual explanations for the results
-> dashboard.py is the Python script generating the application's dashboard using streamlit (to be runned locally using "streamlit run dashboard.py")
-> P7_api.py is the Python script deploying the API using flask (to be runned locally using "python P7_api.py" and accessing it via http://localhost:5000/predict)
-> Evidently_report_full.html is the HTML data drift analysis table