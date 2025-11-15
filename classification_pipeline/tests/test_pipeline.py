from classification_pipeline.pipeline import ClassificationPipeline
from classification_pipeline.data import DATA_PATH


def test_full_pipeline(tmp_path):
    # copy dataset into temp dir for isolation
    import shutil

    test_path = tmp_path / "diabetes.csv"
    shutil.copy(DATA_PATH, test_path)

    pipe = ClassificationPipeline()
    pipe.load_and_clean(test_path)
    pipe.fit()
    pipe.evaluate()
    assert pipe.eval_results["accuracy"] > 0.65  # reasonable baseline
