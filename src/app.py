from flask import Flask, render_template, request
from predict import predict_intent

app = Flask(__name__, template_folder='../templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        intent, confidence = predict_intent(user_input)
        return render_template('result.html', user_input=user_input, intent=intent, confidence=confidence)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)