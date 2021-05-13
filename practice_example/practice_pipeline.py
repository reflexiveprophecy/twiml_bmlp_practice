#python 3.8.4 

# %%
import os
from typing import Text
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    Evaluator,
    ExampleValidator,
    Pusher,
    ResolverNode,
    SchemaGen,
    StatisticsGen,
    Trainer,
    Transform,
)

from tfx.orchestration.local import local_dag_runner
from tfx.extensions.google_cloud_ai_platform.trainer import executor \
        as aip_trainer_executor
from consumer_complaint.config import config
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.proto import pusher_pb2, trainer_pb2, example_gen_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.utils.dsl_utils import external_input
from tfx.orchestration import metadata, pipeline


# %%
def init_components(data_dir, module_file,
                    serving_model_dir=None,
                    ai_platform_training_args=None,
                    ai_platform_serving_args=None,
                    training_steps = 50000,
                    eval_steps = 10000):

    """
    This function is to initialize tfx components
    """

    if serving_model_dir and ai_platform_serving_args:
        raise NotImplementedError(
            "Can't set ai_platform_serving_args and serving_model_dir at "
            "the same time. Choose one deployment option."
        )

    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(
                    name="train", hash_buckets=99
                ),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=1),
            ]
        )
    )

    example_gen = CsvExampleGen(input_base=data_dir, output_config=output)

    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=False,
    )

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=module_file,
    )

    training_kwargs = {
        "module_file": module_file,
        "examples": transform.outputs["transformed_examples"],
        "schema": schema_gen.outputs["schema"],
        "transform_graph": transform.outputs['transform_graph'],
        "train_args": trainer_pb2.TrainArgs(num_steps = training_steps),
        "eval_args": trainer_pb2.EvalArgs(num_steps = eval_steps),
    }


    if ai_platform_training_args:

        training_kwargs.update(
            {
                "custom_executor_spec": executor_spec.ExecutorClassSpec(
                    aip_trainer_executor.GenericExecutor
                ),
                "custom_config": {
                    aip_trainer_executor.TRAINING_ARGS_KEY: ai_platform_training_args  # noqa
                },
            }
        )
    else:
        training_kwargs.update(
            {
                "custom_executor_spec": executor_spec.ExecutorClassSpec(
                    GenericExecutor
                )
            }
        )

    trainer = Trainer(**training_kwargs)
    
    #compile all components in a list
    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
    ]
    return components


# %%
def init_pipeline(components, 
                pipeline_root: Text, 
                direct_num_workers: int) -> pipeline.Pipeline:

    beam_arg = [
        f"--direct_num_workers={direct_num_workers}",
    ]
    tfx_pipeline = pipeline.Pipeline(
        pipeline_name=config.PIPELINE_NAME,
        pipeline_root=config.PIPELINE_ROOT,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            config.METADATA_PATH
        ),
        beam_pipeline_args=beam_arg,
    )
    return tfx_pipeline



# %%
if __name__ == "__main__":
    tfx_components = init_components(config.DATA_DIR_PATH,
                                config.MODULE_FILE_PATH,
                                config.SERVING_MODEL_DIR,
                                )
# %%
    tfx_pipeline = init_pipeline(tfx_components, config.PIPELINE_ROOT, 4)
    


# %%
    #the pipeline doesn't work in ipykernel, so you would have to run 
    # this in terminal 
    local_dag_runner.LocalDagRunner().run(tfx_pipeline)




