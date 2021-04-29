#python3.8.4
#./venv/bin/python

# %%
import pandas as pd
import os
from consumer_complaint.config import config
import tensorflow_data_validation as tfdv 
from sklearn.model_selection import train_test_split
from tfx.orchestration import pipeline
from tfx.components import CsvExampleGen, \
                            StatisticsGen, \
                            SchemaGen, \
                            ExampleValidator
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext


# %%
def csv_statistics_generator(file_path):
    """
    Generate statistics for the csv dataset
    """
    csv_stats = tfdv.generate_statistics_from_csv(data_location = file_path,
                                                delimiter=',')
    csv_schema = tfdv.infer_schema(csv_stats)
    tfdv.display_schema(csv_schema)
    return csv_stats, csv_schema


# %%
def tfrecord_statis_generator(file_path):
    """
    Generate statistics for the tfrecord dataset
    """
    tfrecord_stats = tfdv.generate_statistics_from_tfrecord(data_location = file_path)
    tfrecord_schema = tfdv.infer_schema(tfrecord_stats)
    tfdv.display_schema(tfrecord_schema)
    return tfrecord_stats, tfrecord_schema

# %%
def train_val_split(file_path, shuffle_split = True):
    """
    Train test split from sklearn function
    """    
    data = pd.read_csv(file_path, encoding = 'utf-8')
    if shuffle_split:
        train_data, val_data = train_test_split(data, test_size = 0.1, 
                                                random_state= 42, shuffle= True)
        train_data.to_csv(os.path.join(config.DATA_SPLITS_DIR_PATH, "shuffled_train_data.csv"), index = False)
        val_data.to_csv(os.path.join(config.DATA_SPLITS_DIR_PATH, "shuffled_val_data.csv"), index = False)
    else:
        train_data, val_data = data.iloc[0:-5000, :], data.iloc[-5000:, :]
        train_data.to_csv(os.path.join(config.DATA_SPLITS_DIR_PATH, "loc_train_data.csv"), index = False)
        val_data.to_csv(os.path.join(config.DATA_SPLITS_DIR_PATH, "loc_val_data.csv"), index = False)
    return train_data, val_data


# %%
def csv_statistics_validator(stats, schema):
    """
    Validate statistics from a csv dataset
    """
    stats_anomalies = tfdv.validate_statistics(statistics = stats, schema = schema)
    tfdv.display_anomalies(stats_anomalies)
    return stats_anomalies

# %%
def tfdv_skew_validator(feature_name, train_stats, serve_stats, schema, threshold):
    """
    Validate skew for the csv dataset
    """
    #this doesn't display skew anomalies as the book shows
    tfdv.get_feature(schema, feature_name).skew_comparator.infinity_norm.threshold = threshold
    skew_anomalies = tfdv.validate_statistics(statistics = train_stats,
                                                schema = schema,
                                                serving_statistics = serve_stats)
    tfdv.display_anomalies(skew_anomalies)
    return skew_anomalies

def tfdv_drift_validator(feature_name, train_stats, previous_stats, schema, threshold):
    """
    Validate drift for the csv dataset
    """
    #this doesn't display drift anomalies as the book shows
    tfdv.get_feature(schema, feature_name).drift_comparator.infinity_norm.threshold = threshold
    drift_anomalies = tfdv.validate_statistics(statistics=train_stats, 
                                                schema=schema, 
                                                previous_statistics= previous_stats
                                                )
    tfdv.display_anomalies(drift_anomalies)
    return drift_anomalies


# %%
if __name__ == '__main__':
    #train val split
    train_val_split(file_path = config.DATA_FILE_PATH)

# %%
    #generating train val stats and schema, and then visualize it
    # data_stats, data_schema = csv_statistics_generator(file_path = config.DATA_FILE_PATH)
    train_stats, train_schema = csv_statistics_generator(file_path = config.TRAIN_FILE_PATH)
    val_stats, val_schema = csv_statistics_generator(file_path = config.VAL_FILE_PATH)
    tfdv.visualize_statistics(lhs_statistics = val_stats, rhs_statistics=train_stats,
                            lhs_name = 'VAL_DATASET', rhs_name = 'TRAIN_DATASET')    

# %%
    #check anomalies in train and val 
    train_anomalies = csv_statistics_validator(stats = train_stats, schema = train_schema)
    val_anomalies = csv_statistics_validator(stats = val_stats, schema = val_schema)


# %%
    skew_anomalies = tfdv_skew_validator(feature_name = 'company',
                                        train_stats = train_stats,
                                        serve_stats = val_stats,
                                        schema = train_schema,
                                        threshold = 0.01)

# %%
    drift_anomalies = tfdv_drift_validator(feature_name = 'company',
                                            train_stats = train_stats,
                                            previous_stats = val_stats,
                                            schema = train_schema,
                                            threshold = 0.01
                                            )




