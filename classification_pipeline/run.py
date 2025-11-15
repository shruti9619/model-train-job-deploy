# run.py
import yaml
from argparse import ArgumentParser

from classification_pipeline import ClassificationPipeline, PipelineConfig


def argument_parser():
    parser = ArgumentParser(description="Run Classification Pipeline")
    parser.add_argument(
        "--config-path",
        type=str,
        default="dtree_config.yaml",
        help="Path to the configuration YAML file.",
    )
    return parser


if __name__ == "__main__":
    args = argument_parser().parse_args()
    config_path = args.config_path
    config = yaml.safe_load(open(config_path))
    print(config)
    pconfig = PipelineConfig(
        model_name=config.get("model_name", "dtree"),
        model_params=config.get("model_params", {}),
    )
    p = ClassificationPipeline(pconfig)
    p.load_and_clean()
    p.fit()
    p.evaluate()
    # p.plot_confusion()
