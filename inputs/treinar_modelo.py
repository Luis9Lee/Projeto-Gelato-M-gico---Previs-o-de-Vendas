MODELO-----

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn

# 1. Configurar o MLflow para rastreamento (Tracking)
mlflow.set_experiment("Projeto_Gelato_Magico")

def treinar_modelo():
    with mlflow.start_run():
        # Carregar dados do diretório inputs
        df = pd.read_csv('../inputs/dados_vendas.csv')
        X = df[['temperatura']]
        y = df['vendas']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Treinar o modelo de regressão
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)

        # Previsões e Métricas
        previsoes = modelo.predict(X_test)
        mae = mean_absolute_error(y_test, previsoes)
        rmse = mean_squared_error(y_test, previsoes, squared=False)

        # Registrar Parâmetros e Métricas no MLflow [cite: 120]
        mlflow.log_param("algoritmo", "Regressão Linear")
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)

        # Salvar o modelo no Registry
        mlflow.sklearn.log_model(modelo, "modelo_gelato_v1")
        
        print(f"Modelo treinado! MAE: {mae}")

if __name__ == "__main__":
    treinar_modelo()
    
SIMULAÇÂO -----

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

# Simulação do carregamento de dados (pasta /inputs)
data = {
    'temperatura': [22, 25, 28, 32, 35, 18, 20],
    'vendas': [110, 140, 200, 280, 350, 80, 95]
}
df = pd.DataFrame(data)

# Configuração do Experimento no MLflow
mlflow.set_experiment("Gelato_Magico_Demand")

with mlflow.start_run(run_name="Treino_Regressao_Linear"):
    # 1. Definição do Modelo
    model = LinearRegression()
    X = df[['temperatura']]
    y = df['vendas']
    
    # 2. Treino
    model.fit(X, y)
    
    # 3. Previsão e Métricas
    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    # 4. Registo de Parâmetros e Métricas no MLflow Tracking
    mlflow.log_param("tipo_modelo", "LinearRegression")
    mlflow.log_metric("mae", mae) # Erro Médio Absoluto
    mlflow.log_metric("r2_score", r2) # Coeficiente de Determinação
    
    # 5. Registo do Modelo (Registry)
    mlflow.sklearn.log_model(model, "modelo_sorvete_v1")
    
    print(f"Simulação concluída! MAE: {mae:.2f}, R2: {r2:.2f}")
