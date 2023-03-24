#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request


# In[2]:


import pickle


# In[3]:


import numpy as np


# In[4]:


app = Flask(__name__,template_folder='.')


# In[5]:
def hypothesis(x,theta):
    y_=0.0 #(y hat)
    n=x.shape[0]
    for i in range(n):
        y_+=(theta[i]*x[i])
    return y_


svm_model=pickle.load(open('model_svm','rb'))
mlp_model=pickle.load(open('model_mlp','rb'))
lr_model=pickle.load(open('model_lr','rb'))
rf_model=pickle.load(open('model_rf','rb'))
dt_model=pickle.load(open('model_dt','rb'))


# In[6]:


@app.route('/',methods=['GET'])
def main():
        return render_template("AQI.html")

# In[ ]:

# In[7]:

@app.route('/', methods=['POST'])
def predict():
    #print(request.form)
    if request.method == 'POST':


        final=[]
        final_lr=[]


        final.append(float(request.form['pm2']))
        final.append(float(request.form['pm10']))
        final.append(float(request.form['no']))
        final.append(float(request.form['no2']))
        final.append(float(request.form['nox']))
        final.append(float(request.form['nh3']))
        final.append(float(request.form['co']))
        final.append(float(request.form['so2']))
        final.append(float(request.form['o3']))
        final.append(float(request.form['benzene']))
        final.append(float(request.form['toluene']))
        final.append(float(request.form['xylene']))
        

        req_model = str(request.form['model'])
        
        if req_model == "Support Vector Regression":
            final = np.array(final)
            arr = final.reshape(1, -1)
            prediction = svm_model.predict(arr)
            prediction = prediction[0]

        if req_model == "Multi-Layer Perceptron":
            final = np.array(final)
            arr = final.reshape(1, -1)
            prediction = mlp_model.predict(arr)
            prediction = prediction[0]

        if req_model == "Linear Regression":
            theta,error_list = lr_model
            final_lr.append(1)
            for x in range(0,len(final)):
                final_lr.append(final[x])
            final_lr = np.array(final_lr)
            prediction = hypothesis(final_lr,theta)

        if req_model == "Random Forest Regressor":
            final = np.array(final)
            arr = final.reshape(1, -1)
            prediction = rf_model.predict(arr)
            prediction = prediction[0]

        if req_model == "Decision Tree Regressor":
            final = np.array(final)
            arr = final.reshape(1, -1)
            prediction = dt_model.predict(arr)
            prediction = prediction[0]



    prediction=round(prediction,2)
    if prediction<=50:
        return render_template("AQI.html",pred ='AQI is {}: \nGood; \nMinimal Impact.'.format(prediction))
    if prediction>=51 and prediction<=100:
        return render_template("AQI.html",pred ='AQI is {}: \nSatisfactory; \nMinor breathing discomfort to sensitive people.'.format(prediction))
    if prediction>=101 and prediction<=200:
        return render_template("AQI.html",pred ='AQI is {}: \nModerate; \nBreathing discomfort to the people with lungs, asthma and heart disease.'.format(prediction))
    if prediction>=201 and prediction<=300:
        return render_template("AQI.html",pred ="AQI is {}: \nPoor; \nBreathing discomfort to most people on prolonged exposure.".format(prediction))
    if prediction>=301 and prediction<=400:
        return render_template("AQI.html",pred ='AQI is {}: \nVery Poor; \nRespiratory illness on prolonged exposure.'.format(prediction))  
    if prediction>=401:
        return render_template("AQI.html",pred ='AQI is {}: \nSevere; \nAffects healthy people and seriously impacts those with the existing disease.'.format(prediction))  



# In[ ]:


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




