# Informations

The folders:
-  **http_requests_nasa** 
- **http_requests_wikipedia**

Contains the NASA and Wikipedia datasets corresponding to the prediction intervals. For example, trace60.csv contains the data needed to make predictions for the next 60 minutes.

Original dataset:
 - Wikipédia: http://www.wikibench.eu/wiki/2007-09/
 - NASA: http://ita.ee.lbl.gov/html/contrib/


- **algorithms**
  
Contains the source codes of Machine Learning models (MLP, GRU, ARIMA, Prophet) and their hyperparameter optimizers (*Grid Search* (**grid_search.py**), *Tree of parzen estimators (TPE)* (**hypeas_gru .py** and **hypeas_mlp.py**) for local execution.

To execute the algorithms, you will need an environment with the following libraries below:

- **Keras** https://keras.io/
- **SKlearn** https://scikit-learn.org
- **Hyperopt** http://hyperopt.github.io/hyperopt/
- **Statsmodels** https://www.statsmodels.org/stable/index.html
- **Prophet** https://facebook.github.io/prophet/
- **Matplotlib** https://matplotlib.org/
- **Pandas** https://pandas.pydata.org/
- **Numpy** http://www.numpy.org/

The algorithms below are available as an alternative for online execution:
- **MLP_exec.ipynb**
- **GRU_exec.ipynb**
- **ARIMA_exec.ipynb**
- **Prophet_exec.ipynb**

They correspond to the source code of each Machine Learning model. They are in a similar format to the Jupyter notebook.
When accessing the file page, select *Open in Colab* to access the model in execution mode in the *Google Colaboratory* environment.
