from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Carregar o modelo treinado
modelo = joblib.load('modelo_treinado.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        idade_anos = int(request.form['idade_anos'])
        sexo = request.form['sexo']
        estado_conjugal_atual = request.form['estado_conjugal_atual']
        anos_de_estudo = int(request.form['anos_de_estudo'])
        dirige = int(request.form.get('dirige', 0))
        freq_horta_crua = int(request.form['freq_horta_crua'])
        bebida_alcoolica = int(request.form['bebida_alcoolica'])
        exercicio_fisico = int(request.form['exercicio_fisico'])
        fumante = int(request.form['fumante'])
        imc = float(request.form['imc'])
        cor = request.form['cor']

        # Criar um dicionário com os dados do formulário
        data_dict = {
            'idade_anos': [idade_anos],
            'sexo': [sexo],
            'estado_conjugal_atual': [estado_conjugal_atual],
            'anos_de_estudo': [anos_de_estudo],
            'dirige': [dirige],
            'freq_horta_crua': [freq_horta_crua],
            'bebida_alcoolica': [bebida_alcoolica],
            'exercicio_fisico': [exercicio_fisico],
            'fumante': [fumante],
            'imc': [imc],
            'cor': [cor]
        }

        # Converter o dicionário para um DataFrame
        data_df = pd.DataFrame(data_dict)

        # Fazer a previsão
        prediction = modelo.predict(data_df)
        probability = modelo.predict_proba(data_df)[0][1]

        # Converter previsão para resposta interpretável
        result = "Pressão Alta" if prediction == 1 else "Sem Pressão Alta"

        # Retornar resultado e probabilidade
        return jsonify({
            'prediction_text': f'Resultado: {result}',
            'probability': probability
        })

if __name__ == '__main__':
    app.run(debug=True)
