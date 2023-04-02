import kaggle
#############################################################
# import os
# os.environ['KAGGLE_USERNAME'] = '<your Kaggle username>'
# os.environ['KAGGLE_KEY'] = '<your Kaggle API key>'
##############################################################
# go to your kaggle 'Account' > API > Create New API Token.
# Copy the downloaded kaggle.json file to C:\Users\your_user\.kaggle.
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
api.dataset_download_files('aptos2019-blindness-detection', path='./Dataset', unzip=True)

#if dataset is zipped
# import zipfile
# with zipfile.ZipFile('Dataset/aptos2019-blindness-detection.zip', 'r') as zipref:
#     zipref.extractall('Dataset/')