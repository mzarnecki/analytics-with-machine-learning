"""
Cykliczny pipeline analityczny: Online Retail -> cechy klientów -> ML
-> anomaly detection -> dane dla aplikacji Streamlit.

Wersja demonstracyjna dla Apache Airflow 3.x i TaskFlow API.

Oczekiwane artefakty:
- źródło danych:
  /opt/airflow/data/incoming/online_retail_latest.parquet
- model segmentacji:
  /opt/airflow/models/customer_segmentation_pipeline.joblib
- model wykrywania anomalii:
  /opt/airflow/models/customer_anomaly_pipeline.joblib

Modele powinny być kompletnymi pipeline'ami scikit-learn, które zawierają
preprocessing oraz finalny estimator. Dzięki temu zadanie scoringu nie
odtwarza ręcznie transformacji z notebooka.

Ważne:
- Airflow przekazuje między zadaniami tylko małe metadane i ścieżki.
- Dane tabelaryczne są zapisywane do Parquet, a nie do XCom.
- "Odświeżenie Streamlit" oznacza atomowe opublikowanie nowego pliku,
  który aplikacja odczytuje jako aktualne dane.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from airflow.sdk import dag, get_current_context, task


WARSAW_TZ = ZoneInfo("Europe/Warsaw")

BASE_DIR = Path(
    os.getenv(
        "ANALYTICAL_PIPELINE_BASE_DIR",
        "/opt/airflow/data",
    )
)
MODEL_DIR = Path(
    os.getenv(
        "ANALYTICAL_PIPELINE_MODEL_DIR",
        "/opt/airflow/models",
    )
)

SOURCE_PATH = Path(
    os.getenv(
        "ONLINE_RETAIL_SOURCE_PATH",
        str(BASE_DIR / "incoming" / "online_retail_latest.parquet"),
    )
)
SEGMENTATION_MODEL_PATH = Path(
    os.getenv(
        "SEGMENTATION_MODEL_PATH",
        str(MODEL_DIR / "customer_segmentation_pipeline.joblib"),
    )
)
ANOMALY_MODEL_PATH = Path(
    os.getenv(
        "ANOMALY_MODEL_PATH",
        str(MODEL_DIR / "customer_anomaly_pipeline.joblib"),
    )
)

REQUIRED_COLUMNS = {
    "invoice_no",
    "stock_code",
    "quantity",
    "invoice_date",
    "unit_price",
    "customer_id",
}

MODEL_FEATURES = [
    "recency_days",
    "number_of_orders",
    "total_revenue",
    "average_order_value",
    "number_of_products",
    "purchase_frequency_30d",
    "return_rate",
]


def ensure_directories() -> None:
    """Utwórz katalogi używane przez kolejne zadania."""
    for directory in [
        BASE_DIR / "raw",
        BASE_DIR / "quality",
        BASE_DIR / "features",
        BASE_DIR / "predictions",
        BASE_DIR / "results",
        BASE_DIR / "streamlit",
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def current_partition() -> str:
    """Zwróć datę logicznego uruchomienia w formacie YYYYMMDD."""
    context = get_current_context()
    logical_date = context["logical_date"]

    # logical_date jest obiektem świadomym strefy czasowej.
    local_date = logical_date.astimezone(WARSAW_TZ)
    return local_date.strftime("%Y%m%d")


@dag(
    dag_id="analytical_ml_pipeline",
    description=(
        "Daily Online Retail analytics, ML scoring, anomaly detection "
        "and publishing data for Streamlit."
    ),
    schedule="0 6 * * *",
    start_date=datetime(2026, 1, 1, tzinfo=WARSAW_TZ),
    catchup=False,
    max_active_runs=1,
    tags=["analytics", "polars", "ml", "streamlit"],
    default_args={
        "owner": "analytics",
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },
)
def analytical_ml_pipeline():
    """
    DAG uruchamiany codziennie o 06:00 czasu Europe/Warsaw.

    Kolejność zadań wynika z przekazywania ścieżek przez TaskFlow API:

    fetch_new_data
    -> validate_data
    -> build_customer_features
    -> run_segmentation_model
    -> detect_anomalies
    -> publish_streamlit_data
    """

    @task()
    def fetch_new_data() -> str:
        """
        Pobierz lub skopiuj nowy snapshot danych.

        W wersji produkcyjnej zadanie może:
        - wykonać zapytanie do hurtowni,
        - pobrać plik z object storage,
        - wywołać API,
        - uruchomić eksport z systemu źródłowego.

        W tym przykładzie kopiujemy gotowy plik Parquet do katalogu raw.
        """
        ensure_directories()

        if not SOURCE_PATH.exists():
            raise FileNotFoundError(
                f"Nie znaleziono źródła danych: {SOURCE_PATH}"
            )

        partition = current_partition()
        destination = (
            BASE_DIR / "raw" / f"online_retail_{partition}.parquet"
        )

        temporary = destination.with_suffix(".parquet.tmp")
        shutil.copy2(SOURCE_PATH, temporary)
        temporary.replace(destination)

        return str(destination)

    @task()
    def validate_data(raw_path: str) -> dict[str, str | int | float]:
        """
        Sprawdź schemat i podstawowe reguły jakości.

        Nieprawidłowe dane przerywają workflow przed uruchomieniem modelu.
        """
        import polars as pl

        ensure_directories()
        raw_path_obj = Path(raw_path)

        lazy_frame = pl.scan_parquet(raw_path_obj)
        schema_columns = set(lazy_frame.collect_schema().names())

        missing_columns = REQUIRED_COLUMNS - schema_columns
        if missing_columns:
            raise ValueError(
                "Brakuje wymaganych kolumn: "
                f"{sorted(missing_columns)}"
            )

        report = (
            lazy_frame.select(
                pl.len().alias("rows_total"),
                pl.col("customer_id")
                .is_null()
                .sum()
                .alias("missing_customer_id"),
                pl.col("invoice_date")
                .is_null()
                .sum()
                .alias("missing_invoice_date"),
                (pl.col("quantity") <= 0)
                .sum()
                .alias("non_positive_quantity"),
                (pl.col("unit_price") <= 0)
                .sum()
                .alias("non_positive_price"),
            )
            .collect()
            .row(0, named=True)
        )

        rows_total = int(report["rows_total"])
        if rows_total == 0:
            raise ValueError("Plik źródłowy nie zawiera rekordów.")

        missing_customer_share = (
            float(report["missing_customer_id"]) / rows_total
        )

        # Przykładowa reguła jakości. W realnym projekcie próg powinien
        # wynikać z kontraktu danych i historii źródła.
        if missing_customer_share > 0.50:
            raise ValueError(
                "Ponad 50% rekordów nie ma customer_id: "
                f"{missing_customer_share:.1%}"
            )

        partition = current_partition()
        report_path = (
            BASE_DIR / "quality" / f"quality_{partition}.json"
        )
        report_path.write_text(
            json.dumps(
                {
                    **report,
                    "missing_customer_share": missing_customer_share,
                    "source_path": raw_path,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        return {
            "raw_path": raw_path,
            "quality_report_path": str(report_path),
            "rows_total": rows_total,
            "missing_customer_share": missing_customer_share,
        }

    @task()
    def build_customer_features(
        validation_result: dict[str, str | int | float],
    ) -> str:
        """
        Przekształć pozycje zamówień w tabelę:
        jeden klient = jeden rekord.
        """
        import polars as pl

        ensure_directories()
        raw_path = str(validation_result["raw_path"])

        transactions = (
            pl.scan_parquet(raw_path)
            .with_columns(
                pl.col("invoice_no")
                .cast(pl.String, strict=False)
                .str.strip_chars(),
                pl.col("stock_code")
                .cast(pl.String, strict=False)
                .str.strip_chars(),
                pl.col("customer_id")
                .cast(pl.String, strict=False)
                .str.strip_chars(),
                pl.col("quantity").cast(pl.Float64, strict=False),
                pl.col("unit_price").cast(pl.Float64, strict=False),
                pl.col("invoice_date").cast(pl.Datetime, strict=False),
            )
            .with_columns(
                pl.col("invoice_no")
                .str.to_uppercase()
                .str.starts_with("C")
                .fill_null(False)
                .alias("is_cancelled"),
            )
            .with_columns(
                (
                    pl.col("is_cancelled")
                    | (pl.col("quantity").fill_null(0) < 0)
                ).alias("is_return"),
                (
                    pl.col("quantity") * pl.col("unit_price")
                ).alias("line_revenue"),
            )
        )

        return_stats = (
            transactions.filter(
                pl.col("customer_id").is_not_null()
            )
            .group_by("customer_id")
            .agg(
                pl.len().alias("all_line_count"),
                pl.col("is_return")
                .sum()
                .alias("return_line_count"),
            )
        )

        valid_sales = transactions.filter(
            pl.col("customer_id").is_not_null()
            & pl.col("invoice_date").is_not_null()
            & ~pl.col("is_cancelled")
            & (pl.col("quantity") > 0)
            & (pl.col("unit_price") > 0)
        )

        max_invoice_date = (
            valid_sales.select(
                pl.col("invoice_date").max()
            )
            .collect()
            .item()
        )
        if max_invoice_date is None:
            raise ValueError(
                "Po czyszczeniu nie pozostały poprawne transakcje."
            )

        snapshot_date = max_invoice_date + timedelta(days=1)

        customer_features = (
            valid_sales.group_by("customer_id")
            .agg(
                pl.col("invoice_date")
                .min()
                .alias("first_purchase_date"),
                pl.col("invoice_date")
                .max()
                .alias("last_purchase_date"),
                pl.col("invoice_no")
                .n_unique()
                .alias("number_of_orders"),
                pl.col("line_revenue")
                .sum()
                .alias("total_revenue"),
                pl.col("stock_code")
                .n_unique()
                .alias("number_of_products"),
            )
            .with_columns(
                (
                    pl.lit(snapshot_date)
                    - pl.col("last_purchase_date")
                )
                .dt.total_days()
                .alias("recency_days"),
                (
                    (
                        pl.col("last_purchase_date")
                        - pl.col("first_purchase_date")
                    )
                    .dt.total_days()
                    + 1
                ).alias("active_days"),
            )
            .with_columns(
                (
                    pl.col("total_revenue")
                    / pl.col("number_of_orders")
                ).alias("average_order_value"),
                (
                    pl.col("number_of_orders")
                    / pl.max_horizontal(
                        pl.col("active_days"),
                        pl.lit(1),
                    )
                    * 30
                ).alias("purchase_frequency_30d"),
            )
            .join(
                return_stats,
                on="customer_id",
                how="left",
            )
            .with_columns(
                pl.col("all_line_count").fill_null(0),
                pl.col("return_line_count").fill_null(0),
            )
            .with_columns(
                pl.when(pl.col("all_line_count") > 0)
                .then(
                    pl.col("return_line_count")
                    / pl.col("all_line_count")
                )
                .otherwise(0.0)
                .alias("return_rate"),
            )
            .select(
                "customer_id",
                *MODEL_FEATURES,
                "first_purchase_date",
                "last_purchase_date",
            )
        )

        partition = current_partition()
        output_path = (
            BASE_DIR
            / "features"
            / f"customer_features_{partition}.parquet"
        )

        customer_features.sink_parquet(output_path)
        return str(output_path)

    @task()
    def run_segmentation_model(features_path: str) -> str:
        """
        Uruchom zapisany pipeline segmentacji.

        Oczekiwany artefakt:
        customer_segmentation_pipeline.joblib

        Pipeline powinien zawierać wszystkie transformacje używane podczas
        treningu, np. log1p, skalowanie i K-Means.
        """
        import joblib
        import numpy as np
        import polars as pl

        ensure_directories()

        if not SEGMENTATION_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Nie znaleziono modelu segmentacji: "
                f"{SEGMENTATION_MODEL_PATH}"
            )

        features = pl.read_parquet(features_path)
        model_input = features.select(MODEL_FEATURES).to_pandas()

        segmentation_pipeline = joblib.load(
            SEGMENTATION_MODEL_PATH
        )
        cluster_ids = segmentation_pipeline.predict(model_input)

        result = features.with_columns(
            pl.Series(
                "cluster_id",
                np.asarray(cluster_ids, dtype=int),
            )
        )

        # Jeżeli pipeline udostępnia transform(), finalny K-Means zwraca
        # odległości do centroidów. Zapisujemy najmniejszą z nich.
        if hasattr(segmentation_pipeline, "transform"):
            distances = segmentation_pipeline.transform(model_input)
            distances = np.asarray(distances)

            if distances.ndim == 2:
                result = result.with_columns(
                    pl.Series(
                        "centroid_distance",
                        distances.min(axis=1),
                    )
                )

        partition = current_partition()
        output_path = (
            BASE_DIR
            / "predictions"
            / f"customer_segments_{partition}.parquet"
        )
        result.write_parquet(output_path)

        return str(output_path)

    @task()
    def detect_anomalies(predictions_path: str) -> str:
        """
        Uruchom zapisany pipeline anomaly detection.

        Oczekiwany artefakt:
        customer_anomaly_pipeline.joblib

        Przykładowym finalnym estymatorem może być IsolationForest.
        """
        import joblib
        import numpy as np
        import polars as pl

        ensure_directories()

        if not ANOMALY_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Nie znaleziono modelu anomalii: "
                f"{ANOMALY_MODEL_PATH}"
            )

        data = pl.read_parquet(predictions_path)
        model_input = data.select(MODEL_FEATURES).to_pandas()

        anomaly_pipeline = joblib.load(ANOMALY_MODEL_PATH)
        anomaly_labels = anomaly_pipeline.predict(model_input)

        # Dla IsolationForest etykieta -1 oznacza anomalię.
        is_anomaly = np.asarray(anomaly_labels) == -1

        result = data.with_columns(
            pl.Series("is_anomaly", is_anomaly)
        )

        if hasattr(anomaly_pipeline, "score_samples"):
            # Odwracamy znak: większy wynik = bardziej nietypowy rekord.
            anomaly_score = -np.asarray(
                anomaly_pipeline.score_samples(model_input)
            )
            result = result.with_columns(
                pl.Series("anomaly_score", anomaly_score)
            )

        partition = current_partition()
        output_path = (
            BASE_DIR
            / "results"
            / f"customer_results_{partition}.parquet"
        )
        result.write_parquet(output_path)

        return str(output_path)

    @task()
    def publish_streamlit_data(results_path: str) -> dict[str, str]:
        """
        Atomowo opublikuj wynik dla aplikacji Streamlit.

        Aplikacja może zawsze czytać:
        /opt/airflow/data/streamlit/current_customer_results.parquet

        Nie trzeba restartować Streamlit. Wystarczy ponownie odczytać dane
        lub wyczyścić/ustawić TTL cache po stronie aplikacji.
        """
        import polars as pl

        ensure_directories()

        source = Path(results_path)
        current_path = (
            BASE_DIR
            / "streamlit"
            / "current_customer_results.parquet"
        )
        temporary_path = current_path.with_suffix(".parquet.tmp")

        shutil.copy2(source, temporary_path)
        temporary_path.replace(current_path)

        result_frame = pl.scan_parquet(current_path)
        summary = (
            result_frame.select(
                pl.len().alias("customers"),
                pl.col("cluster_id")
                .n_unique()
                .alias("clusters"),
                pl.col("is_anomaly")
                .sum()
                .alias("anomalies"),
            )
            .collect()
            .row(0, named=True)
        )

        manifest_path = (
            BASE_DIR / "streamlit" / "manifest.json"
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "published_at": datetime.now(
                        tz=WARSAW_TZ
                    ).isoformat(),
                    "data_path": str(current_path),
                    "source_results_path": results_path,
                    **summary,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        return {
            "streamlit_data_path": str(current_path),
            "manifest_path": str(manifest_path),
        }

    raw_path = fetch_new_data()
    validation_result = validate_data(raw_path)
    features_path = build_customer_features(validation_result)
    predictions_path = run_segmentation_model(features_path)
    results_path = detect_anomalies(predictions_path)
    publish_streamlit_data(results_path)


analytical_ml_pipeline()
