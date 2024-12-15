from flask import Flask, render_template, request, jsonify
from chat import gen_response, load_model
from langdetect import detect

app = Flask(__name__)

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    user_query = request.get_json().get("message")
    load_model()

    response = gen_response(user_query)
    print(response)    

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)


## uploaded file excel
## insert sqlite for checking data
## checking data in sqlite for find a same data; checking a format
## 