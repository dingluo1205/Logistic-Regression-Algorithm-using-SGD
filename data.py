import csv
import random

def load_csv(filename):
    lines = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def load_adult_data():
    return load_csv("adult.data")

# Note: Possibly use different data for training and validation to get a more accurate result, 
# but remember that in the last part your model will be trained on the full training data
# load_adult_data() and be tested on a test dataset you don't have access to.

def split_data(train,val):
    data = load_adult_data()
    random.shuffle(data)
    length = len(data)
    if train == 1:
        num = int(length*0.75)
        return data[:num]
    else:
        num = length-int(length*0.75)
        return data[num:]


def load_adult_train_data():
#    return load_adult_data()
     return split_data(1,0)

def load_adult_valid_data():
#    return load_adult_data()
     return split_data(0,1)

