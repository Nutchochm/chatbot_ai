from flask import Flask, render_template, request, jsonify
from chat import gen_response, load_model, loadtyphoon2, typhoon2chat
from langdetect import detect
#, template_folder='/var/www/chatbot/templates'
app = Flask(__name__)
#load_model()
model, token = loadtyphoon2()

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    user_query = request.get_json().get("message")
    response = typhoon2chat(user_query, model, token)

    #responses = jsonify({"answer": response})
    #response = gen_response(user_query)
    print(response)    
    return jsonify({"answer": response})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)


## uploaded file excel
## insert sqlite for checking data
## checking data in sqlite for find a same data; checking a format
## 