from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the NASA SMRU Applicatsdfgadsfasdfadsion / ahora si seridfgsdgadfgaos mis perros digalo !"

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)
