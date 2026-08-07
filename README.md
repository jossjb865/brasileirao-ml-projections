# Brasileirão Série A – ML Market Projections v2.0

Sistema de Machine Learning para proyección de resultados del Campeonato Brasileiro Série A (comp_4795) usando datos reales de TheStatsAPI.

## Arquitectura v2.0

La versión 2.0 representa una refactorización completa del sistema de proyecciones, migrando desde un prototipo con modelos sintéticos hacia un pipeline de Machine Learning de producción riguroso y sin sesgos de información.

### Principales cambios y mejoras:

- **Eliminados los 3 falsos modelos LSTM**: Se removieron las implementaciones LSTM anteriores que repetían vectores estáticos en lugar de procesar secuencias temporales reales, reemplazándolas por un ensemble robusto y fundamentado estadísticamente.
- **Eliminado data leakage**: Las estadísticas post-partido (como posesión, remates directos o faltas ocurridas durante el juego) ya no se utilizan como features de entrada para predecir dicho partido, garantizando que el modelo opere únicamente con información disponible previa al pitazo inicial.
- **Añadido ELO rating por equipo**: Implementación de un sistema de puntuación ELO dinámico que actualiza la fuerza relativa de cada equipo tras cada jornada, ajustando por ventaja de localía y diferencia de goles.
- **Añadido backtesting walk-forward y cross-validation temporal**: Evaluación mediante validación cruzada respetando la línea de tiempo histórica (time-series split) y simulaciones walk-forward para evitar la contaminación entre pasado y futuro.
- **Añadido optimización de pesos del ensemble con scipy**: Reemplazo de pesos fijos por optimización numérica (`scipy.optimize`) orientada a minimizar la pérdida logarítmica (Log Loss) in-sample y out-of-sample.
- **Añadido calibración de probabilidades**: Calibración isotónica y de Platt (Logistic Regression) sobre las probabilidades predichas para asegurar que un 70% de probabilidad estimada se traduzca efectivamente en un 70% de aciertos reales.
- **Añadido detección de value bets**: Comparación automática de las probabilidades calibradas del modelo contra las cuotas implícitas de las casas de apuestas (odds de TheStatsAPI) para identificar apuestas con valor esperado positivo (EV+).

## Modelos del Ensemble

El sistema combina las predicciones de múltiples modelos heterogéneos para maximizar la precisión y reducir la varianza:

| Modelo | Tipo | Descripción |
|--------|------|-------------|
| Poisson | Goles → 1X2 | Dixon-Coles con PoissonRegressor |
| XGBoost | Multiclass 1X2 | Gradient boosting con features de forma |
| CatBoost | Multiclass 1X2 | Gradient boosting con manejo de categóricos |
| Decision Tree | Multiclass 1X2 | Baseline interpretable |
| Logistic Regression | Multiclass 1X2 | Baseline calibrado |

## Feature Engineering

El pipeline de extracción de características genera únicamente datos pre-partido sin data leakage. Las features construidas incluyen:

- **Rolling goals (5 y 10 partidos)**:
  - Goles a favor promedio (`home_gf_roll5`, `away_gf_roll5`, `roll10`).
  - Goles en contra promedio (`home_ga_roll5`, `away_ga_roll5`, `roll10`).
  - Diferencia de goles rolling (`home_gd_roll5`, `away_gd_roll5`, `roll10`).
- **Rolling points (5 y 10 partidos)**:
  - Puntos obtenidos acumulados en las últimas 5 y 10 jornadas (`home_pts_roll5`, `away_pts_roll5`, `roll10`).
- **Forma reciente (W/D/L streak)**:
  - Racha de victorias, empates y derrotas consecutivas de cada equipo en las jornadas previas.
- **ELO rating por equipo**:
  - Rating ELO actualizado jornada a jornada para medir la fuerza relativa global de local y visitante.
- **Racha de local/visitante**:
  - Métricas de rendimiento diferenciadas exclusivamente jugando en casa (Home Form) y fuera de casa (Away Form).
- **Días de descanso**:
  - Tiempo de descanso transcurrido entre el partido actual y el compromiso inmediatamente anterior de cada club.
- **Head-to-head histórico**:
  - Resultados, victorias y promedio de goles en los enfrentamientos directos previos entre ambos equipos.
- **Diferencias (home - away)**:
  - Cálculo explícito del diferencial (`home - away`) para todas las features rolling y métricas de rendimiento (ej. `gf_diff_roll5`, `pts_diff_roll5`, `elo_diff`).

## Pipeline

El flujo de trabajo se divide en 4 etapas principales secuenciales:

1. **Data Sync** → Descarga datos reales de TheStatsAPI
2. **Feature Engineering** → Construye features sin data leakage
3. **Training** → Entrena ensemble con CV temporal y backtesting
4. **Prediction** → Genera predicciones con value detection

## GitHub Actions

El repositorio cuenta con tres flujos automatizados de GitHub Actions encargados de mantener las predicciones actualizadas:

- **Daily Data Sync**: Executado diariamente a las **05:00 UTC** (`.github/workflows/daily_data_sync.yml`). Descarga los datos más recientes de partidos, resultados y odds desde TheStatsAPI y genera el dataset de features.
- **Train Ensemble**: Ejecutado los **Lunes a las 06:00 UTC** (`.github/workflows/train_ensemble.yml`). Re-entrena el ensemble completo con los nuevos resultados de la jornada previa, optimiza pesos y guarda los artefactos de modelos actualizados.
- **Predict & Publish**: Ejecutado diariamente a las **06:30 UTC** (`.github/workflows/predict_and_publish.yml`). Carga los modelos entrenados, genera las proyecciones para los próximos encuentros y publica los resultados en la carpeta `outputs/`.

## Uso Local

```bash
export THESTATS_API_KEY='tu_clave'
pip install -r requirements.txt
python -m src.data.fetch
python -m src.data.features
python -m src.train
python -m src.predict
```

## Métricas de Evaluación

Para evaluar y validar el rendimiento predictivo del sistema se utilizan las siguientes métricas:

- **Accuracy**: Proporción de resultados predichos correctamente (1X2) sobre el total de partidos jugados. Sirve como referencia base de clasificación.
- **Log Loss**: Mide la incertidumbre y penaliza duramente las predicciones probabilísticas confiadas pero incorrectas. Es la función objetivo principal para la optimización de pesos del ensemble.
- **Brier Score**: Error cuadrático medio entre la probabilidad predicha para cada resultado y el resultado real (one-hot encoded). Evalúa la precisión global de las probabilidades asignadas.
- **Calibration Error**: Mide la discrepancia promedio entre la probabilidad estimada por el modelo y la frecuencia observada de los eventos en bins de probabilidad. Un bajo error de calibración garantiza que las cuotas calculadas sean matemáticamente fiables para encontrar apuestas de valor.

## Configuración

La configuración del sistema se administra centralizadamente a través de archivos YAML situados en `configs/`:

- `configs/competition.yaml`: Contiene los metadatos de la competición (ID `comp_4795`, nombre "Campeonato Brasileiro Série A", país "Brazil"), las temporadas a descargar (`current`, `previous`), los endpoints de TheStatsAPI, timeouts, reintentos y las rutas de almacenamiento interno para archivos Parquet (`matches.parquet`, `features.parquet`, etc.).
- `configs/model_hyperparams.yaml`: Define los hiperparámetros específicos para cada algoritmo del ensemble (Alpha en Poisson, profundidad y tasa de aprendizaje en XGBoost/CatBoost, parámetros del Decision Tree, etc.), la ventana rolling por defecto (`rolling_window: 5`), y los pesos iniciales/optimizados asignados a cada modelo dentro del ensemble.

## Licencia

MIT
