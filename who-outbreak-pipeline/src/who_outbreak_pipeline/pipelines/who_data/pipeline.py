from kedro.pipeline import Pipeline, node, pipeline

# Import your node groups
from .nodes import fetch_who_data, fetch_future_who_data
from .nodes_clean import clean_who_data, split_by_year
from .nodes_features import engineer_features
from .nodes_model import train_model, evaluate_model, predict_future
from .nodes_viz import aggregate_who_data, summarize_who_trends


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [

            # --------------------------------------------------------------
            # 1) FETCH WHO HISTORICAL DATA
            # --------------------------------------------------------------
            node(
                func=fetch_who_data,
                inputs=dict(
                    indicator_codes="params:who.indicator_codes",
                    output_path="params:who.raw_output_path",
                ),
                outputs="who_raw_data",
                name="fetch_who_data_node",
            ),

            # 2) CLEAN historical data
            node(
                func=clean_who_data,
                inputs="who_raw_data",
                outputs="who_clean_data",
                name="clean_who_data_node",
                tags=["clean"],
            ),

            # 3) FEATURE ENGINEER historical
            node(
                func=engineer_features,
                inputs="who_clean_data",
                outputs="who_feature_data",
                name="engineer_features_node",
                tags=["features"],
            ),

            # 4) SPLIT data
            node(
                func=split_by_year,
                inputs="who_feature_data",
                outputs=["train_df", "test_df"],
                name="split_by_year_node",
            ),

            # --------------------------------------------------------------
            # 5) TRAIN MODEL → returns (model, model_columns)
            # --------------------------------------------------------------
            node(
                func=train_model,
                inputs="train_df",
                outputs=["who_model", "who_model_columns"],
                name="train_model_node",
                tags=["model"],
            ),

            # --------------------------------------------------------------
            # 6) EVALUATE MODEL
            # --------------------------------------------------------------
            node(
                func=evaluate_model,
                inputs=dict(
                    model="who_model",
                    test_df="test_df",
                    model_columns="who_model_columns",
                ),
                outputs=["who_model_info", "who_predictions"],
                name="evaluate_model_node",
                tags=["model"],
            ),

            # --------------------------------------------------------------
            # FUTURE PIPELINE (2023–2025)
            # --------------------------------------------------------------

            # 7) Fetch future WHO data
            node(
                func=fetch_future_who_data,
                inputs=dict(indicator_codes="params:who.indicator_codes"),
                outputs="who_future_raw",
                name="fetch_future_who_node",
                tags=["future"],
            ),

            # 8) Clean future WHO data
            node(
                func=clean_who_data,
                inputs="who_future_raw",
                outputs="who_future_clean",
                name="clean_future_who_node",
                tags=["future", "clean"],
            ),

            # 9) Feature engineer future WHO data
            node(
                func=engineer_features,
                inputs="who_future_clean",
                outputs="who_future_features",
                name="engineer_future_features_node",
                tags=["future", "features"],
            ),

            # --------------------------------------------------------------
            # 10) PREDICT FUTURE VALUES
            # --------------------------------------------------------------
            node(
                func=predict_future,
                inputs=dict(
                model="who_model",
                future_df="who_future_features",
                model_columns="who_model_columns"
                ),
                outputs="who_future_predictions"
            ),


            # --------------------------------------------------------------
            # VISUALIZATION & SUMMARY
            # --------------------------------------------------------------

            # 11) Aggregate historical for visualization
            node(
                func=aggregate_who_data,
                inputs="who_clean_data",
                outputs="who_aggregated_data",
                name="aggregate_who_data_node",
                tags=["agg"],
            ),

            # 12) Summary report
            node(
                func=summarize_who_trends,
                inputs=dict(
                    df_agg="who_aggregated_data",
                    model_info="who_model_info",
                    preds_df="who_predictions",
                    out_html="params:who.summary_html",
                ),
                outputs=["who_summary_country", "who_summary_region"],
                name="summarize_who_trends_node",
                tags=["viz", "report"],
            ),
        ]
    )
