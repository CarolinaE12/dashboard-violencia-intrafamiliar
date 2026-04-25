from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

# Datos de prueba (luego los cambias por Orange)
@app.route("/api/data")
def data():
    return jsonify({
        "kpis": {
            "casos_total": 125,
            "tasa": 8.5,
            "reincidencia": 12,
            "municipios": 3
        },
        "tipos": [
            {"tipo": "Física", "valor": 40},
            {"tipo": "Psicológica", "valor": 30},
            {"tipo": "Económica", "valor": 20},
            {"tipo": "Sexual", "valor": 10}
        ],
        "tendencia": [
            {"mes": "Ene", "valor": 10},
            {"mes": "Feb", "valor": 20},
            {"mes": "Mar", "valor": 15},
            {"mes": "Abr", "valor": 30}
        ]
    })

if __name__ == "__main__":
    app.run(debug=True)