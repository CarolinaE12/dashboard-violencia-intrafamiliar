from flask import Flask, render_template

app = Flask(__name__)

POWERBI_URL = "https://app.powerbi.com/view?r=eyJrIjoiNDgwNWJmZTItNjRjOS00NThhLWIyNzEtMGQxYWViNWY2ZjgzIiwidCI6IjA3ZGE2N2EwLTFmNDMtNGU4Yy05NzdmLTVmODhiNjQ3MGVlNiIsImMiOjR9"

@app.route("/")
def index():
    return render_template("index.html", powerbi_url=POWERBI_URL)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)