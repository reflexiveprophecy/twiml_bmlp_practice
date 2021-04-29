#python 3.8.4
#./venv/bin/python
import os 
#Project Paths
PACKAGE_DIR = os.path.dirname(os.path.dirname(__file__))
ROOT_DIR = os.path.dirname(PACKAGE_DIR)
FILE_NAME = 'consumer_complaints_with_narrative.csv'

#Data Paths
DATA_DIR_PATH = os.path.join(PACKAGE_DIR, 'files', 'data')
DATA_FILE_PATH = os.path.join(DATA_DIR_PATH, FILE_NAME)

#Splitted Data Paths
DATA_SPLITS_DIR_PATH = os.path.join(PACKAGE_DIR, 'files', 'data_splits')
TRAIN_FILE_PATH = os.path.join(DATA_SPLITS_DIR_PATH, 'shuffled_train_data.csv')
VAL_FILE_PATH = os.path.join(DATA_SPLITS_DIR_PATH, 'shuffled_val_data.csv')

#TF Record Paths
RECORD_NAME = 'consumer_complaint.tfrecord'
RECORD_DIR_PATH = os.path.join(PACKAGE_DIR, 'files','tf_record')
RECORD_FILE_PATH = os.path.join(RECORD_DIR_PATH, RECORD_NAME)

#TFX Pipeline
PIPELINE_ROOT = os.path.join(ROOT_DIR, 'pipeline_root')
PIPELINE_NAME = "consumer_complaint_pipeline"
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")

#Model Directory
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, "serving_model", PIPELINE_NAME)

#GOOGLE BIG QUERY
GCP_PROJECT_ID = 'consumer-complaint-310721'
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
GOOGLE_APPLICATION_CREDENTIALS = "consumer-complaint-service.json"
TEMP_GCS_LOCATION = "gs://tfx_test_04142021"
GOOGLE_CREDENTIAL_PATH = os.path.join(PROJECT_DIR, 'credentials', GOOGLE_APPLICATION_CREDENTIALS)

#Practice_Example Paths
PRACTICE_EXAMPLE_DIR_PATH = os.path.join(ROOT_DIR, 'practice_example')
MODULE_FILE_PATH = os.path.join(PRACTICE_EXAMPLE_DIR_PATH, 'module.py')

#Training Parameters
TRAIN_STEPS = 50000
EVAL_STEPS = 10000

