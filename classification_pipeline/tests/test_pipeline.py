from classification_pipeline import ClassificationPipeline, PipelineConfig
from classification_pipeline.data import DATA_PATH


def test_full_pipeline(tmp_path):
    # copy dataset into temp dir for isolation
    import shutil

    test_path = tmp_path / "diabetes.csv"
    shutil.copy(DATA_PATH, test_path)

    pconfig = PipelineConfig(
        model_name="dtree",
        model_params={"max_depth": 5, "random_state": 42},
    )

    pipe = ClassificationPipeline(config=pconfig)
    pipe.load_and_clean()
    pipe.fit()
    pipe.evaluate()
    pipe.plot_confusion()
    assert pipe.eval_results["accuracy"] > 0.65  # reasonable baseline
