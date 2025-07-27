from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/book', methods=['POST'])
def book():
    pickup = request.form['pickup']
    drop = request.form['drop']
    distance = abs(len(pickup) - len(drop)) + 5  # mock distance logic
    fare = distance * 15  # ₹15 per unit

    return render_template('result.html', pickup=pickup, drop=drop, fare=fare)

if __name__ == '__main__':
    app.run(debug=True)