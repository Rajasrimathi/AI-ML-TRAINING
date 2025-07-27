from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_performance.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        study_hours = float(request.form['study_hours'])
        attendance = float(request.form['attendance_percentage'])
        prev_grade = float(request.form['previous_grade_numeric'])

        input_data = [[study_hours, attendance, prev_grade]]
        prediction = model.predict(input_data)

        return render_template('result.html', prediction=prediction[0])
    except:
        return "Invalid input. Please enter valid numbers."

if __name__== '__main__':
    app.run(debug=True)