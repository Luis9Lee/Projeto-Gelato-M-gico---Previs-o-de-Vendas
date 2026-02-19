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
