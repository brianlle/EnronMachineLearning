
# coding: utf-8



import sys
import csv
import ast
import pickle
sys.path.append("../tools/")

import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier




### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Load the dictionary containing the text learning feature from csv file


'''
with open("your_author_prediction_data.pkl", "r") as data_file:
    email_prediction_data = pickle.load(data_file)
'''


with open('your_author_prediction_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader: # only 1 row expected
        email_prediction_data = row
        
# convert csv data from form key: '[x, y]' to key: [x, y]
for email in email_prediction_data:
    email_prediction_data[email] = ast.literal_eval(email_prediction_data[email])
    
#add new feature data to data_dict

for person in data_dict:
    if data_dict[person]['email_address'] in email_prediction_data:
        email = data_dict[person]['email_address']
        if email_prediction_data[email][1] != 0:    
            data_dict[person]['email_predicted_proportion'] = float(email_prediction_data[email][0])                                                               / float(email_prediction_data[email][1])
        else: data_dict[person]['email_predicted_proportion'] = 0  #for those with an address listed but no emails
    else:
        data_dict[person]['email_predicted_proportion'] = 'NaN'   #for those with no email address listed


# remove outlier that is 'TOTAL' row

data_dict.pop('TOTAL')



features_list = ['poi','salary', 'bonus',
                'from_poi_to_this_person', 'from_this_person_to_poi',
                'email_predicted_proportion']

# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')


# store dataset and classifier for testing by tester.py

dump_classifier_and_data(clf, my_dataset, features_list)




