from flask import Flask, render_template, url_for

app = Flask("YAGG",template_folder='app/templates', static_folder='app/static')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)