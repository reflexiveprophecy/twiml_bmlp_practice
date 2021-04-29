#python3.8.4
#./venv/bin/python
"""
This file serves as a practice for Chapter 3 Data Ingestion
Make sure you change the /consumer_complaint/config/config.py
to define your own directories and set your credentials. 
"""

# %%
import os
import tensorflow as tf
import pandas as pd
import tfx
import csv
import numpy as np
from consumer_complaint.config import config
from tfx.utils.dsl_utils import external_input
from tfx.components import CsvExampleGen, ImportExampleGen
from tfx.extensions.google_cloud_big_query.example_gen.component import BigQueryExampleGen
from tfx.proto import example_gen_pb2
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config.GOOGLE_CREDENTIAL_PATH


# %%
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def clean_rows(row):
    if not row["zip_code"]:
        row["zip_code"] = "99999"
    return row

def convert_zipcode_to_int(zipcode):
    if isinstance(zipcode, str) and "XX" in zipcode:
        zipcode = zipcode.replace("XX", "00")
    int_zipcode = int(zipcode)
    return int_zipcode


# %%
def tfrecord_data_writer(file_path):
    tfrecord_filename = config.RECORD_FILE_PATH
    tf_record_writer = tf.io.TFRecordWriter(tfrecord_filename)

    with open(file_path, encoding = 'utf-8') as csv_file:
        reader = csv.DictReader(csv_file, delimiter = ',', quotechar = '"')
        for row in reader:
            row = clean_rows(row)
            example = tf.train.Example(features = tf.train.Features(feature = {
                'product': _bytes_feature(row['product'].encode('utf-8')),
                'sub_product': _bytes_feature(row['sub_product'].encode('utf-8')),
                'issue': _bytes_feature(row['issue'].encode('utf-8')),
                'sub_issue': _bytes_feature(row['sub_issue'].encode('utf-8')),
                'consumer_complaint_narrative': _bytes_feature(row['consumer_complaint_narrative'].encode('utf-8')),
                'company': _bytes_feature(row['company'].encode('utf-8')),
                'state': _bytes_feature(row['state'].encode('utf-8')),
                'zip_code': _int64_feature(convert_zipcode_to_int(row["zip_code"])),
                'company_response': _bytes_feature(row['company_response'].encode('utf-8')),
                'timely_response': _bytes_feature(row['timely_response'].encode('utf-8')),
                'consumer_disputed': _bytes_feature(row['consumer_disputed'].encode('utf-8'))
            }))
            tf_record_writer.write(example.SerializeToString())
        tf_record_writer.close()

    return tf_record_writer


# %%
def data_split(file_path):
    """splitting data before feeding into CsvExampleGen"""
    output_config = example_gen_pb2.Output(
        split_config = example_gen_pb2.SplitConfig(splits = [
            example_gen_pb2.SplitConfig.Split(name = 'train', hash_buckets = 6),
            example_gen_pb2.SplitConfig.Split(name = 'eval', hash_buckets = 2),
            example_gen_pb2.SplitConfig.Split(name = 'test', hash_buckets = 2)
        ]))
    split_example = CsvExampleGen(input_base = file_path, output_config = output_config)
    return split_example


# %%
def existing_data_split(file_path):
    """preserving existing data splits with existing subdirectories"""
    input_config = example_gen_pb2.Input(splits = [
        example_gen_pb2.Input.Split(name = 'train', pattern = 'train/*'),
        example_gen_pb2.Input.Split(name = 'eval', pattern = 'eval/*'),
        example_gen_pb2.Input.Split(name = 'test', pattern = 'test/*')
    ])
    existing_split_example = CsvExampleGen(input_base = file_path, input_config = input_config)
    return existing_split_example


# %%
def span_data_split(file_path):
    """data split with span(data snapshot that can replicate existing data records"""
    input_config = example_gen_pb2.Input(splits = [
        example_gen_pb2.Input.Split(pattern = 'export-{SPAN}/*')
    ])
    span_example = CsvExampleGen(input_base = file_path, input_config = input_config)
    return span_example


# %%

if __name__ == '__main__':
    context = InteractiveContext(pipeline_root=config.PIPELINE_ROOT)
    
# %%
    complaint_df = pd.read_csv(config.DATA_FILE_PATH, encoding = 'utf-8')

# %%
    #ImportExampleGen with TFRecord
    complaint_tfrecord = tfrecord_data_writer(file_path = config.DATA_FILE_PATH)
    example_gen = ImportExampleGen(input_base = config.RECORD_DIR_PATH)
    context.run(example_gen)
    

# %%
    #Plain simple csv file for CsvExampleGen
    example_gen = CsvExampleGen(input_base = config.DATA_DIR_PATH)
    context.run(example_gen)


# %%
    #Data Split
    split_example_gen = data_split(file_path = config.DATA_SPLITS_DIR_PATH)
    context.run(split_example_gen)


# %%
    #Existing Data Split
    #Won't run through as there is no train folder
    # existing_split_example_gen = existing_data_split(file_path = FILE_DIR_PATH)
    # context.run(existing_split_example_gen)

# %%
    #Spanning Datasets
    span_split_example_gen = span_data_split(file_path = config.DATA_SPLITS_DIR_PATH)
    span_split_example_gen
    

# %%
    #Big Query Data Ingestion
    query = """
    SELECT * 
    FROM `consumer-complaint-310721.consumer_complaint.consumer_complaint_data` 
    LIMIT 100
    """
    bigquery_example_gen = BigQueryExampleGen(query = query)
    context.run(bigquery_example_gen, beam_pipeline_args=["--project={}".format(config.GCP_PROJECT_ID), 
                                                        "--temp_location={}".format(config.TEMP_GCS_LOCATION)])




# %%



