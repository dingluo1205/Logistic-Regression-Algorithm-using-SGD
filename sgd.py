from math import exp
from math import sqrt
import random

# TODO: Calculate logistic
def logistic(x):
    return 1/(1+exp(-x))
    

# TODO: Calculate dot product of two lists
def dot(x, y):
    s = 0
    for i in range(0,len(x)):
        s = s + x[i]*y[i]
    return s

# TODO: Calculate prediction based on model
def predict(model, point):
    return logistic(dot(model,point['features']))

# TODO: Calculate accuracy of predictions on data
def accuracy(data, predictions):
    correct = 0
    for i in range(0,len(data)):
        correct = correct + data[i]['label'] - predictions[i]
    return 1-abs(float(correct)/len(data))

# TODO: Update model using learning rate and L2 regularization
def update(model, point, delta, rate, lam):
    prediction = predict(model,point)
    new_delta = [0]*len(delta)
    p = logistic(prediction)
    for i in range(0,len(model)):
        label = [point['label']]*len(point['features'])
        overall = 0
        for j in range(0,len(point['features'])):
            overall = overall + point['features'][j]*(point['label']-p)
        new_delta[i] = rate*(-lam*model[i]+ overall )-model[i]*delta[i]
        model[i] = model[i] - new_delta[i]
    return model,new_delta

def initialize_model(k):
    return [random.gauss(0, 1) for x in range(k)]
#    return [0 for x in range(k)]

# TODO: Train model using training data
def train(data, epochs, rate, lam):
    model = initialize_model(len(data[0]['features']))
    min_loss = 99999
 #   a = accuracy(model,)
    
    delta = [0]*len(data[0]['features'])
    for t in range(0,epochs):
        for point in data:
            error = ((predict(model,point))-point['label'])
            loss = error*error+lam*sqrt(dot(model,model)) 
            if loss < min_loss:
                model = update(model,point,delta,rate,lam)[0]
                delta = update(model,point,delta,rate,lam)[1]
                min_loss = loss
   #     if 
    return model
        
def extract_features(raw):
    data = []
    for r in raw:
        point = {}
        point["label"] = (r['income'] == '>50K')
        features = []
        features.append(1.)
        features.append(float(r['age'])/100)
        features.append(float(r['education_num'])/20)
        features.append(r['marital'] == 'Married-civ-spouse')
        #TODO: Add more feature extraction rules here!
        features.append(r['sex']=='Male')
        features.append(float(r['hr_per_week'])/100)
        point['features'] = features
        data.append(point)
    
    

    return data

# TODO: Tune your parameters for final submission
def submission(data):
    new_model= [0]*len(data[0]['features'])
    a = 0
# change the learning rates and regularization parameters so that the model accuracy will also be greater than 0.75
    for rate in range(0,10):
        for lam in range(0,10):
            model = train(data,2,float(rate)/100,float(lam)/10)
            prediction = [0]*len(data[0]['features'])
            for point in data:
                prediction.append(predict(model,point))
            acc = accuracy(data,prediction)
            if acc > 0.75 and acc>a:
                new_model = model       
                a = acc

    return new_model
    
