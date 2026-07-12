# Streamlit ML Analytics PoC

Uruchomienie:

```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

Aplikacja automatycznie szuka plików:

```text
outputs/online_retail_customer_segments.parquet
outputs/online_retail_product_text_clusters.parquet
outputs/nasa_asteroids_classification_predictions.parquet
outputs/nasa_asteroids_logistic_pipeline.joblib
outputs/consumer_complaints_topics.parquet
```

Gdy pliku nie ma, odpowiednia zakładka używa wyraźnie oznaczonych
danych demonstracyjnych.

Ścieżki można nadpisać zmiennymi środowiskowymi:

```text
ML_ANALYTICS_OUTPUT_DIR
CUSTOMER_RESULTS_PATH
PRODUCT_RESULTS_PATH
ASTEROID_PREDICTIONS_PATH
ASTEROID_MODEL_PATH
COMPLAINTS_RESULTS_PATH
```

Zakładki:

- **Klienci** — segment, profil, podobni klienci i rekomendacje.
- **Asteroidy** — prawdopodobieństwo, podobne obiekty i formularz.
- **Skargi klientów** — tematy, wyszukiwanie i prompt dla LLM.