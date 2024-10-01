from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/dashboard')
def graphs():
    return render_template("dashboard.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')