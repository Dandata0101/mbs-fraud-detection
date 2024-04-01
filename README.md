# mbs-fraud-detection
[![Author - DanRamirez](https://img.shields.io/badge/Author-DanRamirez-2ea44f?style=for-the-badge)](https://github.com/Dandata0101)
![Python - Version](https://img.shields.io/badge/PYTHON-3.11-red?style=for-the-badge&logo=python&logoColor=white)


**Jupyter Reports** :blue_book:
1. [Enron Email analysis :email::chart_with_upwards_trend:](https://github.com/Dandata0101/mbs-fraud-detection/blob/main/Classwork_day01.ipynb)
2. [Top Features & analysis :green_book:](https://github.com/Dandata0101/mbs-fraud-detection/blob/main/Classwork_day02.ipynb)
3.  [ML Models analyses :robot:](https://github.com/Dandata0101/mbs-fraud-detection/blob/main/Classwork_day03.ipynb)

&nbsp;

All pythons code have been place into functions to make it easier show ouputs in Jupyter notebooks:

[Script Directory :file_folder:](https://github.com/Dandata0101/mbs-fraud-detection/tree/main/scripts)
1. [Csv to Parquet conversion :floppy_disk:](https://github.com/Dandata0101/mbs-fraud-detection/blob/main/scripts/csvtopaquet.py)
   - converts csv files int parquet files to reduce size by more than half.
2. [Clean Data :broom:](https://github.com/Dandata0101/mbs-fraud-detection/blob/main/scripts/dataclean.py)
   - cleans column names, fills blanks with zero and creates dummie variables for Categorical columns.
3. [email analysis fx :email::chart_with_upwards_trend:](https://github.com/Dandata0101/mbs-fraud-detection/blob/main/scripts/emailfx.py)
   - Text mining functions to find key phrases in email content and returns filters and content in a word clould.
4. [Charts :chart_with_upwards_trend:](https://github.com/Dandata0101/mbs-fraud-detection/blob/main/scripts/distributionchart.py)
5. [AI Email summary :robot::email:](https://github.com/Dandata0101/mbs-fraud-detection/blob/main/scripts/emailsummary.py)
   - Uses openai API to summarize any email content into a concise paragraph (see setup for more detail).
6. [ML Models :robot::chart_with_upwards_trend:](https://github.com/Dandata0101/mbs-fraud-detection/blob/main/scripts/models.py)
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - KNN
   - Gaussian
7. [LightGBM Models :robot::chart_with_upwards_trend:](https://github.com/Dandata0101/mbs-fraud-detection/blob/main/scripts/lgbmmodels.py)
   - used a `binary_logloss`  function to return the top features used to predict the `Y` variables and provides the option to return top `n` features in a shap chart. 


## setup and Installment requirements

### Required package installation 

```bash
# Install dependencies from a requirements file
pip install -r requirements.txt
```

### AI Email summary :robot::email:
In order for the AI Email summary :robot::email: to work, follow these steps:
1. go to https://platform.openai.com/docs/overview and create an account
2. to go the API Key and create a key for your project
![Openai](https://github.com/Dandata0101/mbs-fraud-detection/blob/main/03-images/openai.png "api keys")

3. create an environment file `.env` and the following in the file:

```
openaikey=yourapihere
```
note: make sure to add `.env` to your `.gitignore` file. 

