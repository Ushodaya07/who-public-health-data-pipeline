from who_outbreak_pipeline.pipelines.who_data import pipeline as who_data_pipeline

def register_pipelines():
    """Register the WHO Outbreak pipeline as the default Kedro pipeline."""
    who_data = who_data_pipeline.create_pipeline()

    return {
        "__default__": who_data,
        "who_data": who_data,
    }
 