# Brasileirão Série A – ML Market Projections

Sistema completo de Machine Learning para proyección de mercados del **Campeonato Brasileiro Série A** (`competition_id = comp_4795`) usando **exclusivamente datos reales** de [TheStatsAPI](https://www.thestatsapi.com).

## Ensemble

| Modelo            | Tipo                  | Peso por defecto |
|-------------------|-----------------------|------------------|
| Poisson           | Goles → 1X2           | 0.18             |
| XGBoost           | Multiclass 1X2        | 0.22             |
| CatBoost          | Multiclass 1X2        | 0.22             |
| Decision Tree     | Multiclass 1X2        | 0.08             |
| LSTM_Momentum     | Secuencia de forma    | 0.10             |
| LSTM_Result       | Secuencia de resultados | 0.08           |
| LSTM_Model        | Secuencia + contexto  | 0.12             |

## Requisitos

- Python **3.11+**
- Secret de GitHub: `THESTATS_API_KEY` (ya configurado)
- Plan de TheStatsAPI con cuota suficiente

## Estructura del repositorio

```
brasileirao-ml-projections/
├── .github/workflows/
│   ├── daily_data_sync.yml
│   ├── train_ensemble.yml
│   └── predict_and_publish.yml
├── configs/
│   ├── competition.yaml
│   └── model_hyperparams.yaml
├── src/
│   ├── api_client.py
│   ├── data/
│   │   ├── fetch.py
│   │   ├── features.py
│   │   └── store.py
│   ├── models/
│   │   ├── poisson.py
│   │   ├── xgboost_model.py
│   │   ├── catboost_model.py
│   │   ├── decision_tree.py
│   │   ├── lstm_momentum.py
│   │   ├── lstm_result.py
│   │   └── lstm_model.py
│   ├── ensemble.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── data/          # generado en runtime
├── models/        # artefactos entrenados
├── outputs/       # predicciones
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Uso local (opcional)

```bash
export THESTATS_API_KEY="tu_clave"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Descargar datos reales
python -m src.data.fetch

# 2. Construir features
python -m src.data.features

# 3. Entrenar ensemble
python -m src.train

# 4. Generar predicciones de próximos partidos
python -m src.predict
```

## Automatización con GitHub Actions (desde el móvil)

1. Abre la **app de GitHub** en el móvil.
2. Entra al repositorio → pestaña **Actions**.
3. Selecciona el workflow deseado:
   - **Daily Data Sync** → descarga partidos + stats + odds reales
   - **Train Ensemble** → re-entrena todos los modelos
   - **Predict & Publish** → genera predicciones y sube artefactos
4. Pulsa **Run workflow** (workflow_dispatch).

Los workflows también se ejecutan automáticamente:
- Sync de datos: todos los días 05:00 UTC
- Entrenamiento: todos los lunes 06:00 UTC
- Predicciones: todos los días 06:30 UTC

Los resultados (parquets/CSVs) quedan disponibles como **Artifacts** descargables desde la misma pantalla de Actions.

## Competición

- **competition_id**: `comp_4795`
- Nombre: Campeonato Brasileiro Série A
- Fuente: TheStatsAPI (Bearer token)

## Notas importantes

- No se usan datos de muestra. Todo proviene de la API en tiempo real.
- Los modelos LSTM usan una aproximación de secuencia basada en features rolling (práctica y estable con el volumen típico de una liga).
- Las predicciones son probabilísticas (1X2). No garantizan beneficios.
- Respeta los rate limits de tu plan de TheStatsAPI.
- Los artefactos de datos y modelos se guardan como GitHub Artifacts (no se suben al repositorio git).

## Licencia

MIT
```

---

### 3. Archivos residuales mínimos

#### `src/data/__init__.py` (ya indicado, vacío)

```python
# package
```

#### `src/models/__init__.py` (ya indicado, vacío)

```python
# package
```

#### Confirmación de `src/__init__.py`

```python
__version__ = "1.0.0"
```

---

### 4. Orden recomendado de primera ejecución (GitHub Actions)

1. Lanza **Daily Data Sync** (desde el móvil o web).
2. Cuando termine, lanza **Train Ensemble**.
3. Cuando termine, lanza **Predict & Publish**.
4. Descarga el artifact `brasileirao-predictions`.

A partir de ahí el schedule se encarga del resto.

---

**Repositorio completo generado en 5 secciones.**

Resumen de lo entregado:

| Sección | Contenido |
|---------|-----------|
| 1 | Estructura, `.gitignore`, `requirements.txt`, `competition.yaml`, `api_client.py`, `utils.py`, README inicial |
| 2 | `store.py`, `fetch.py` (datos reales `comp_4795`), `features.py` |
| 3 | Poisson, XGBoost, CatBoost, DecisionTree, 3 LSTMs + `ensemble.py` |
| 4 | `train.py`, `predict.py`, `model_hyperparams.yaml`, 3 workflows de GitHub Actions |
| 5 | `pyproject.toml`, README completo, instrucciones de uso desde móvil |

El sistema está listo para clonar, configurar el secret `THESTATS_API_KEY` (ya lo tienes) y ejecutarse.
