from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

modelo_rf = joblib.load('modelo_rf.pkl')  
modelo_lr = joblib.load('modelo_reducao_gases.pkl') 

reverse_label_mapping = {0: 'Baixo', 1: 'Medio', 2: 'Alto', 3: 'Muito Alto'}

@app.route('/prever_politica', methods=['POST'])
def prever_politica():
    """
    Faz previsões de política com base nos dados fornecidos.
    """
    try:
        dados = request.json

        campos_obrigatorios = ['Year', 'Solar (terawatt-hours)', 'Wind (terawatt-hours)',
                               'Hydropower (terawatt-hours)', 'Other renewables (terawatt-hours)']
        for campo in campos_obrigatorios:
            if campo not in dados:
                return jsonify({'erro': f"Campo obrigatório '{campo}' não encontrado"}), 400

        entrada = pd.DataFrame([{
            'Year': dados['Year'],
            'Solar (terawatt-hours)': dados['Solar (terawatt-hours)'],
            'Wind (terawatt-hours)': dados['Wind (terawatt-hours)'],
            'Hydropower (terawatt-hours)': dados['Hydropower (terawatt-hours)'],
            'Other renewables (terawatt-hours)': dados['Other renewables (terawatt-hours)']
        }])

        y_pred = modelo_rf.predict(entrada)
        classe_predita = reverse_label_mapping[y_pred[0]]

        return jsonify({
            'Ano': dados['Year'],
            'Classe Predita': classe_predita
        }), 200

    except Exception as e:
        return jsonify({'erro': str(e)}), 500

@app.route('/prever_emissao', methods=['POST'])
def prever_emissao():
    """
    Calcula a diferença entre previsões:
    1. Usando todas as variáveis (Year, Other_renewables, Solar, Wind, Hydropower).
    2. Usando apenas o Year, a predição somado com as outras variáveis.
    """
    try:
        dados = request.json

        campos_obrigatorios = ['Year', 'Other_renewables', 'Solar', 'Wind', 'Hydropower']
        for campo in campos_obrigatorios:
            if campo not in dados:
                return jsonify({'erro': f"Campo obrigatório '{campo}' não encontrado"}), 400

        year = dados['Year']
        other_renewables = dados['Other_renewables']
        solar = dados['Solar']
        wind = dados['Wind']
        hydropower = dados['Hydropower']

        full_features = pd.DataFrame([[year, other_renewables, solar, wind, hydropower]],
                                     columns=['Year', 'Other_renewables', 'Solar', 'Wind', 'Hydropower'])

        full_prediction = modelo_lr.predict(full_features)[0]

        zero_prediction = full_prediction + (other_renewables + solar + wind + hydropower)

        difference = zero_prediction - full_prediction

        return jsonify({
            'Ano': year,
            'Predicao_com_renovaveis': full_prediction,
            'Predicao_sem_renovaveis': zero_prediction,
            'Diferenca': difference
        }), 200

    except Exception as e:
        return jsonify({'erro': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
