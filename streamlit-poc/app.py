from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="ML Analytics PoC",
    page_icon="📊",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
OUTPUT_DIR = Path(
    os.getenv(
        "ML_ANALYTICS_OUTPUT_DIR",
        str(REPO_ROOT / "outputs"),
    )
)

CUSTOMER_FEATURES = [
    "recency_days",
    "number_of_orders",
    "total_revenue",
    "average_order_value",
    "number_of_products",
    "purchase_frequency_30d",
    "return_rate",
]

ASTEROID_EXCLUDED_COLUMNS = {
    "source_index",
    "actual_target",
    "target",
    "hazardous",
    "predicted_probability",
    "predicted_class_0_5",
    "predicted_class",
}


def first_existing_path(
    environment_variable: str,
    candidates: Iterable[Path],
) -> Path | None:
    configured = os.getenv(environment_variable)

    if configured:
        configured_path = Path(configured)
        if configured_path.exists():
            return configured_path

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


@st.cache_data(show_spinner=False)
def read_parquet(path: str) -> pd.DataFrame:
    return pl.read_parquet(path).to_pandas()


@st.cache_resource(show_spinner=False)
def load_joblib(path: str):
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def build_text_index(texts: tuple[str, ...]):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def create_demo_customers() -> pd.DataFrame:
    rng = np.random.default_rng(42)

    profiles = [
        {
            "segment_name": "Klienci okazjonalni",
            "recency_days": 120,
            "number_of_orders": 2,
            "total_revenue": 180,
            "average_order_value": 90,
            "number_of_products": 4,
            "purchase_frequency_30d": 0.5,
            "return_rate": 0.02,
        },
        {
            "segment_name": "Klienci regularni",
            "recency_days": 25,
            "number_of_orders": 14,
            "total_revenue": 2200,
            "average_order_value": 160,
            "number_of_products": 38,
            "purchase_frequency_30d": 2.2,
            "return_rate": 0.04,
        },
        {
            "segment_name": "Klienci wysokowartościowi",
            "recency_days": 12,
            "number_of_orders": 31,
            "total_revenue": 9200,
            "average_order_value": 310,
            "number_of_products": 74,
            "purchase_frequency_30d": 4.5,
            "return_rate": 0.05,
        },
        {
            "segment_name": "Klienci traceni",
            "recency_days": 260,
            "number_of_orders": 6,
            "total_revenue": 980,
            "average_order_value": 165,
            "number_of_products": 16,
            "purchase_frequency_30d": 0.7,
            "return_rate": 0.03,
        },
        {
            "segment_name": "Klienci hurtowi",
            "recency_days": 18,
            "number_of_orders": 9,
            "total_revenue": 14800,
            "average_order_value": 1750,
            "number_of_products": 95,
            "purchase_frequency_30d": 1.4,
            "return_rate": 0.08,
        },
    ]

    rows = []

    for cluster_id, profile in enumerate(profiles):
        for index in range(24):
            rows.append(
                {
                    "customer_id": f"DEMO-{cluster_id}-{index:03d}",
                    "cluster_id": cluster_id,
                    "segment_name": profile["segment_name"],
                    "recency_days": max(
                        1,
                        int(
                            rng.normal(
                                profile["recency_days"],
                                max(
                                    5,
                                    profile["recency_days"] * 0.18,
                                ),
                            )
                        ),
                    ),
                    "number_of_orders": max(
                        1,
                        int(
                            rng.normal(
                                profile["number_of_orders"],
                                max(
                                    1,
                                    profile["number_of_orders"] * 0.2,
                                ),
                            )
                        ),
                    ),
                    "total_revenue": max(
                        10,
                        rng.normal(
                            profile["total_revenue"],
                            profile["total_revenue"] * 0.2,
                        ),
                    ),
                    "average_order_value": max(
                        5,
                        rng.normal(
                            profile["average_order_value"],
                            profile["average_order_value"] * 0.15,
                        ),
                    ),
                    "number_of_products": max(
                        1,
                        int(
                            rng.normal(
                                profile["number_of_products"],
                                max(
                                    2,
                                    profile["number_of_products"] * 0.2,
                                ),
                            )
                        ),
                    ),
                    "purchase_frequency_30d": max(
                        0.05,
                        rng.normal(
                            profile["purchase_frequency_30d"],
                            max(
                                0.1,
                                profile["purchase_frequency_30d"] * 0.18,
                            ),
                        ),
                    ),
                    "return_rate": float(
                        np.clip(
                            rng.normal(
                                profile["return_rate"],
                                0.015,
                            ),
                            0,
                            0.35,
                        )
                    ),
                    "is_anomaly": index == 0,
                }
            )

    return pd.DataFrame(rows)


def create_demo_products() -> pd.DataFrame:
    products = [
        "JUMBO BAG RED RETROSPOT",
        "JUMBO BAG PINK POLKADOT",
        "LUNCH BAG SPACEBOY DESIGN",
        "SHOPPING BAG VINTAGE ROSE",
        "WHITE HANGING HEART T-LIGHT HOLDER",
        "GLASS STAR FROSTED T-LIGHT HOLDER",
        "SILVER HANGING HEART CANDLE HOLDER",
        "SET OF THREE CAKE TINS PANTRY DESIGN",
        "CERAMIC MUG BLUE POLKADOT",
        "VINTAGE KITCHEN STORAGE JAR",
        "WOODEN PICTURE FRAME WHITE FINISH",
        "VINTAGE PHOTO FRAME GOLD",
        "CHRISTMAS TREE DECORATION SILVER",
        "WOODEN STAR CHRISTMAS DECORATION",
        "WOODEN TOY TRAIN SET",
        "CHILDRENS APRON SPACEBOY",
    ]

    return pd.DataFrame(
        {
            "stock_code": [
                f"DEMO-P{index:03d}"
                for index in range(len(products))
            ],
            "description": products,
        }
    )


def create_demo_asteroids() -> pd.DataFrame:
    rng = np.random.default_rng(43)
    rows = 240

    absolute_magnitude = rng.normal(22.0, 2.3, rows)
    diameter_min = np.exp(rng.normal(-2.0, 0.65, rows))
    diameter_max = diameter_min * rng.uniform(1.5, 2.4, rows)
    relative_velocity = np.exp(rng.normal(9.6, 0.35, rows))
    miss_distance = np.exp(rng.normal(15.2, 0.8, rows))
    moid = np.exp(rng.normal(-3.0, 0.9, rows))
    eccentricity = np.clip(rng.beta(2.0, 3.5, rows), 0.01, 0.98)
    semi_major_axis = rng.uniform(0.7, 4.5, rows)
    inclination = np.clip(rng.gamma(2.0, 6.0, rows), 0, 70)
    orbital_period = np.exp(rng.normal(6.3, 0.45, rows))

    raw_score = (
        -0.8 * (absolute_magnitude - 22)
        - 1.2 * np.log1p(moid)
        + 0.08 * (inclination - inclination.mean())
        + rng.normal(0, 0.7, rows)
    )
    probability = 1 / (1 + np.exp(-raw_score))
    target = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "source_index": np.arange(rows),
            "absolute_magnitude": absolute_magnitude,
            "est_dia_in_km_min": diameter_min,
            "est_dia_in_km_max": diameter_max,
            "relative_velocity_km_per_hr": relative_velocity,
            "miss_dist_kilometers": miss_distance,
            "minimum_orbit_intersection": moid,
            "eccentricity": eccentricity,
            "semi_major_axis": semi_major_axis,
            "inclination": inclination,
            "orbital_period": orbital_period,
            "actual_target": target,
            "predicted_probability": probability,
            "predicted_class_0_5": (
                probability >= 0.5
            ).astype(int),
        }
    )


def load_customers() -> tuple[pd.DataFrame, str]:
    path = first_existing_path(
        "CUSTOMER_RESULTS_PATH",
        [
            OUTPUT_DIR
            / "online_retail_customer_segments.parquet",
            OUTPUT_DIR
            / "online_retail_customer_features.parquet",
            Path(
                "/opt/airflow/data/streamlit/"
                "current_customer_results.parquet"
            ),
        ],
    )

    if path:
        frame = read_parquet(str(path)).copy()

        if "cluster_id" not in frame.columns:
            frame["cluster_id"] = 0

        if "segment_name" not in frame.columns:
            frame["segment_name"] = frame["cluster_id"].map(
                lambda value: f"Segment {value}"
            )

        return frame, str(path)

    return create_demo_customers(), "dane demonstracyjne"


def load_products() -> tuple[pd.DataFrame, str]:
    path = first_existing_path(
        "PRODUCT_RESULTS_PATH",
        [
            OUTPUT_DIR
            / "online_retail_product_text_clusters.parquet",
        ],
    )

    if path:
        return read_parquet(str(path)).copy(), str(path)

    return create_demo_products(), "dane demonstracyjne"


def load_asteroids() -> tuple[pd.DataFrame, str]:
    path = first_existing_path(
        "ASTEROID_PREDICTIONS_PATH",
        [
            OUTPUT_DIR
            / "nasa_asteroids_classification_predictions.parquet",
        ],
    )

    if path:
        return read_parquet(str(path)).copy(), str(path)

    return create_demo_asteroids(), "dane demonstracyjne"


def load_asteroid_model():
    path = first_existing_path(
        "ASTEROID_MODEL_PATH",
        [
            OUTPUT_DIR
            / "nasa_asteroids_logistic_pipeline.joblib",
        ],
    )

    if path:
        return load_joblib(str(path)), str(path)

    return None, "brak zapisanego modelu"



def numeric_columns(
    frame: pd.DataFrame,
    preferred: Iterable[str] | None = None,
) -> list[str]:
    candidates = (
        list(preferred)
        if preferred is not None
        else frame.select_dtypes(
            include=np.number
        ).columns.tolist()
    )

    return [
        column
        for column in candidates
        if (
            column in frame.columns
            and pd.api.types.is_numeric_dtype(frame[column])
            and frame[column].notna().sum() > 1
        )
    ]


def nearest_rows(
    frame: pd.DataFrame,
    selected_position: int,
    feature_columns: list[str],
    top_k: int = 5,
) -> pd.DataFrame:
    clean = frame[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    clean = clean.fillna(clean.median(numeric_only=True))

    scaler = StandardScaler()
    matrix = scaler.fit_transform(clean)

    distances = np.linalg.norm(
        matrix - matrix[selected_position],
        axis=1,
    )
    order = np.argsort(distances)

    selected_indices = [
        index
        for index in order
        if index != selected_position
    ][:top_k]

    result = frame.iloc[selected_indices].copy()
    result["distance"] = distances[selected_indices]
    return result


def top_terms(
    texts: list[str],
    number_of_terms: int = 8,
) -> list[str]:
    if not texts:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)
    weights = np.asarray(matrix.mean(axis=0)).ravel()
    vocabulary = np.asarray(
        vectorizer.get_feature_names_out()
    )

    top_indices = weights.argsort()[::-1][
        :number_of_terms
    ]
    return vocabulary[top_indices].tolist()


customers, customers_source = load_customers()
products, products_source = load_products()
asteroids, asteroids_source = load_asteroids()
asteroid_model, asteroid_model_source = (
    load_asteroid_model()
)

st.title("Interaktywne demo analiz ML")
st.caption(
    "Połączenie danych, modeli, wizualizacji i interakcji "
    "użytkownika w prostej aplikacji PoC."
)

with st.sidebar:
    st.header("Źródła danych")
    st.write("**Klienci:**", customers_source)
    st.write("**Produkty:**", products_source)
    st.write("**Asteroidy:**", asteroids_source)
    st.write("**Model asteroid:**", asteroid_model_source)

customer_tab, asteroid_tab = st.tabs(
    [
        "👥 Klienci",
        "☄️ Asteroidy",
    ]
)

with customer_tab:
    st.header("Segmentacja klientów Online Retail")

    customer_id_column = (
        "customer_id"
        if "customer_id" in customers.columns
        else customers.columns[0]
    )

    customer_options = customers[
        customer_id_column
    ].astype(str).tolist()

    selected_customer_id = st.selectbox(
        "Wybierz klienta",
        customer_options,
    )

    customer_positions = np.flatnonzero(
        customers[customer_id_column]
        .astype(str)
        .to_numpy()
        == selected_customer_id
    )
    selected_position = int(customer_positions[0])
    selected_customer = customers.iloc[
        selected_position
    ]

    segment_name = selected_customer.get(
        "segment_name",
        f"Segment {selected_customer.get('cluster_id', '-')}",
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Segment",
        str(segment_name),
    )
    metric_columns[1].metric(
        "Liczba zamówień",
        f"{selected_customer.get('number_of_orders', 0):,.0f}",
    )
    metric_columns[2].metric(
        "Łączny przychód",
        f"{selected_customer.get('total_revenue', 0):,.2f}",
    )
    metric_columns[3].metric(
        "Recency",
        f"{selected_customer.get('recency_days', 0):,.0f} dni",
    )

    available_customer_features = numeric_columns(
        customers,
        CUSTOMER_FEATURES,
    )

    if available_customer_features:
        selected_cluster = selected_customer.get(
            "cluster_id",
            0,
        )
        cluster_frame = customers[
            customers["cluster_id"]
            == selected_cluster
        ]

        segment_median = cluster_frame[
            available_customer_features
        ].median(numeric_only=True)

        comparison = pd.DataFrame(
            {
                "cecha": available_customer_features,
                "klient": [
                    selected_customer[feature]
                    for feature in available_customer_features
                ],
                "mediana_segmentu": [
                    segment_median[feature]
                    for feature in available_customer_features
                ],
            }
        )
        comparison["relacja_do_segmentu"] = (
            comparison["klient"]
            / comparison["mediana_segmentu"].replace(
                0,
                np.nan,
            )
        )

        st.subheader("Charakterystyka klienta")
        st.dataframe(
            comparison.round(3),
            use_container_width=True,
            hide_index=True,
        )

        chart_data = comparison.set_index("cecha")[
            ["relacja_do_segmentu"]
        ]
        chart_data = chart_data.replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(0)
        st.bar_chart(chart_data)

        st.subheader("Najbardziej podobni klienci")
        similar_customers = nearest_rows(
            customers,
            selected_position,
            available_customer_features,
            top_k=5,
        )

        display_columns = [
            column
            for column in [
                customer_id_column,
                "segment_name",
                "cluster_id",
                "total_revenue",
                "number_of_orders",
                "recency_days",
                "distance",
            ]
            if column in similar_customers.columns
        ]

        st.dataframe(
            similar_customers[
                display_columns
            ].round(3),
            use_container_width=True,
            hide_index=True,
        )

    if "is_anomaly" in customers.columns:
        if bool(
            selected_customer.get(
                "is_anomaly",
                False,
            )
        ):
            st.warning(
                "Klient został oznaczony jako nietypowy "
                "i może wymagać dodatkowej weryfikacji."
            )

    st.subheader("Rekomendacje content-based")

    description_column = next(
        (
            column
            for column in [
                "description",
                "Description",
            ]
            if column in products.columns
        ),
        None,
    )

    if description_column:
        product_descriptions = (
            products[description_column]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        seed_product = st.selectbox(
            "Wybierz produkt kupowany lub oglądany "
            "przez klienta",
            product_descriptions,
        )

        vectorizer, product_matrix = (
            build_text_index(
                tuple(product_descriptions)
            )
        )
        _ = vectorizer
        seed_position = product_descriptions.index(
            seed_product
        )
        scores = cosine_similarity(
            product_matrix[seed_position],
            product_matrix,
        ).ravel()
        order = np.argsort(scores)[::-1]

        recommendations = [
            {
                "produkt": product_descriptions[index],
                "podobieństwo": scores[index],
            }
            for index in order
            if index != seed_position
        ][:5]

        st.dataframe(
            pd.DataFrame(
                recommendations
            ).round(3),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "To prosta rekomendacja na podstawie tekstu "
            "produktu. Pełna rekomendacja klienta powinna "
            "dodatkowo uwzględniać jego historię zakupów."
        )

with asteroid_tab:
    st.header("Klasyfikacja asteroid")

    probability_column = next(
        (
            column
            for column in [
                "predicted_probability",
                "probability",
                "hazardous_probability",
            ]
            if column in asteroids.columns
        ),
        None,
    )

    target_column = next(
        (
            column
            for column in [
                "actual_target",
                "target",
                "hazardous",
            ]
            if column in asteroids.columns
        ),
        None,
    )

    identifier_column = next(
        (
            column
            for column in [
                "source_index",
                "neo_reference_id",
                "name",
            ]
            if column in asteroids.columns
        ),
        None,
    )

    if identifier_column is None:
        asteroids = asteroids.reset_index(
            names="source_index"
        )
        identifier_column = "source_index"

    selected_asteroid_id = st.selectbox(
        "Wybierz obiekt z danych testowych",
        asteroids[identifier_column]
        .astype(str)
        .tolist(),
    )

    asteroid_positions = np.flatnonzero(
        asteroids[identifier_column]
        .astype(str)
        .to_numpy()
        == selected_asteroid_id
    )
    selected_asteroid_position = int(
        asteroid_positions[0]
    )
    selected_asteroid = asteroids.iloc[
        selected_asteroid_position
    ]

    asteroid_metrics = st.columns(3)

    if probability_column:
        probability = float(
            selected_asteroid[
                probability_column
            ]
        )
        asteroid_metrics[0].metric(
            "Prawdopodobieństwo",
            f"{probability:.1%}",
        )
        asteroid_metrics[1].metric(
            "Predykcja przy progu 0,5",
            (
                "potencjalnie niebezpieczna"
                if probability >= 0.5
                else "pozostała"
            ),
        )

    if target_column:
        asteroid_metrics[2].metric(
            "Etykieta rzeczywista",
            str(
                int(
                    selected_asteroid[
                        target_column
                    ]
                )
            ),
        )

    asteroid_feature_columns = [
        column
        for column in numeric_columns(asteroids)
        if column not in ASTEROID_EXCLUDED_COLUMNS
    ]

    if asteroid_feature_columns:
        st.subheader("Najbliższe podobne obiekty")
        similar_asteroids = nearest_rows(
            asteroids,
            selected_asteroid_position,
            asteroid_feature_columns,
            top_k=5,
        )

        asteroid_display_columns = [
            column
            for column in [
                identifier_column,
                probability_column,
                target_column,
                "absolute_magnitude",
                "minimum_orbit_intersection",
                "relative_velocity_km_per_hr",
                "distance",
            ]
            if (
                column
                and column
                in similar_asteroids.columns
            )
        ]

        st.dataframe(
            similar_asteroids[
                asteroid_display_columns
            ].round(4),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Formularz predykcji")

    if (
        asteroid_model is not None
        and hasattr(
            asteroid_model,
            "predict_proba",
        )
    ):
        model_features = list(
            getattr(
                asteroid_model,
                "feature_names_in_",
                [],
            )
        )

        if model_features:
            input_values = {}

            with st.form(
                "asteroid_prediction_form"
            ):
                left, right = st.columns(2)

                for index, feature in enumerate(
                    model_features
                ):
                    container = (
                        left
                        if index % 2 == 0
                        else right
                    )

                    if (
                        feature in asteroids.columns
                        and pd.api.types.is_numeric_dtype(
                            asteroids[feature]
                        )
                    ):
                        default_value = float(
                            pd.to_numeric(
                                asteroids[feature],
                                errors="coerce",
                            ).median()
                        )
                        input_values[feature] = (
                            container.number_input(
                                feature,
                                value=default_value,
                                format="%.6f",
                            )
                        )
                    else:
                        values = (
                            asteroids[feature]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                            if feature
                            in asteroids.columns
                            else ["unknown"]
                        )
                        input_values[feature] = (
                            container.selectbox(
                                feature,
                                values or ["unknown"],
                            )
                        )

                submitted = (
                    st.form_submit_button(
                        "Oblicz prawdopodobieństwo"
                    )
                )

            if submitted:
                model_input = pd.DataFrame(
                    [input_values]
                )
                model_probability = float(
                    asteroid_model.predict_proba(
                        model_input
                    )[0, 1]
                )
                st.success(
                    "Prawdopodobieństwo klasy "
                    f"pozytywnej: "
                    f"{model_probability:.1%}"
                )
        else:
            st.info(
                "Model został wczytany, ale nie "
                "udostępnia nazw cech."
            )
    else:
        st.info(
            "Brak zapisanego pipeline'u klasyfikacji. "
            "Uruchom notebook 02, aby utworzyć plik "
            "`outputs/"
            "nasa_asteroids_logistic_pipeline.joblib`."
        )

        if (
            asteroid_feature_columns
            and probability_column
        ):
            st.caption(
                "Poniższy formularz jest trybem "
                "demonstracyjnym. Wynik jest "
                "interpolowany na podstawie najbliższych "
                "obiektów z zapisanych predykcji."
            )

            selected_features_for_demo = (
                asteroid_feature_columns[:6]
            )
            demo_input = {}

            with st.form(
                "asteroid_demo_form"
            ):
                form_columns = st.columns(2)

                for index, feature in enumerate(
                    selected_features_for_demo
                ):
                    default_value = float(
                        pd.to_numeric(
                            asteroids[feature],
                            errors="coerce",
                        ).median()
                    )
                    demo_input[feature] = (
                        form_columns[
                            index % 2
                        ].number_input(
                            feature,
                            value=default_value,
                            format="%.6f",
                            key=f"demo_{feature}",
                        )
                    )

                demo_submitted = (
                    st.form_submit_button(
                        "Oblicz wynik demonstracyjny"
                    )
                )

            if demo_submitted:
                reference = asteroids[
                    selected_features_for_demo
                ].apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                reference = reference.fillna(
                    reference.median(
                        numeric_only=True
                    )
                )

                scaler = StandardScaler()
                reference_scaled = (
                    scaler.fit_transform(
                        reference
                    )
                )
                input_scaled = scaler.transform(
                    pd.DataFrame(
                        [demo_input]
                    )
                )

                distances = np.linalg.norm(
                    reference_scaled
                    - input_scaled,
                    axis=1,
                )
                nearest = np.argsort(
                    distances
                )[:10]
                weights = 1 / np.maximum(
                    distances[nearest],
                    1e-6,
                )
                demo_probability = float(
                    np.average(
                        asteroids.iloc[
                            nearest
                        ][probability_column].astype(
                            float
                        ),
                        weights=weights,
                    )
                )

                st.warning(
                    "Wynik demonstracyjny: "
                    f"{demo_probability:.1%}"
                )

st.divider()
