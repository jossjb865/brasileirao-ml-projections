# Brasileirão Série A – ML Market Projections

Sistema de proyección de mercados de fútbol para el **Campeonato Brasileiro Série A** (`comp_4795`) usando **datos reales exclusivos** de TheStatsAPI.

## Requisitos

- Python 3.11+
- `THESTATS_API_KEY` configurada como secret de GitHub o variable de entorno

## Competición

- **competition_id**: `comp_4795`
- Nombre oficial: Campeonato Brasileiro Série A

## Estructura

Ver árbol completo en la raíz del repositorio.

## Flujo automatizado (GitHub Actions)

1. `daily_data_sync.yml` → descarga partidos + stats + odds reales
2. `train_ensemble.yml` → re-entrenamiento del ensemble (semanal)
3. `predict_and_publish.yml` → predicciones diarias

Disparable desde la app de GitHub en el móvil (`workflow_dispatch`).

## Modelos del ensemble

- Poisson
- XGBoost
- CatBoost
- Decision Tree
- LSTM_Momentum
- LSTM_Result
- LSTM_Model
