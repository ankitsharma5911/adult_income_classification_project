from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('form.html')


@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data = CustomData(
            age=float(request.form.get("age")),
            workclass=(request.form.get('workclass')),
            education=(request.form.get("education")),
            marital_status=(request.form.get("marital_status")),
            occupation=(request.form.get("occupation")),
            relationship=(request.form.get("relationship")),
            race=(request.form.get("race")),
            gender=(request.form.get("gender")),
            capital_gain=float(request.form.get("capital_gain")),
            capital_loss=float(request.form.get("capital_loss")),
            hours_per_week=float(request.form.get("hours_per_week"))
            )
            
        
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        
        pred=predict_pipeline.predict(final_new_data)

        results = ''

        if pred == 0:
            results='salary is less then 50k'
        else:
            results = 'salary is greater then 50k'

        return render_template('form.html',final_result=results)
    

if __name__ =='__main__':
        app.run(host='0.0.0.0',port=5000,debug=True)