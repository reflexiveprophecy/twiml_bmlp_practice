#python 3.8.4
#./venv/bin/python
import os 
#DEFINING DIRECTORIES
DIRNAME = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
FILE_NAME = 'consumer_complaints_with_narrative.csv'
RECORD_NAME = 'consumer_complaint.tfrecord'
FILE_PATH = os.path.join(DIRNAME, 'data_root', FILE_NAME)
FILE_DIR_PATH = os.path.join(DIRNAME, 'data_root')
RECORD_DIR_PATH = os.path.join(DIRNAME, 'record_root')
RECORD_FILE_PATH = os.path.join(RECORD_DIR_PATH, RECORD_NAME)
PIPELINE_ROOT = os.path.join(DIRNAME, 'tfx')

#GOOGLE BIG QUERY
GCP_PROJECT_ID = 'consumer-complaint-310721'
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
GOOGLE_APPLICATION_CREDENTIALS = "consumer-complaint-service.json"
TEMP_GCS_LOCATION = "gs://tfx_test_04142021"
GOOGLE_CREDENTIAL_PATH = os.path.join(PROJECT_DIR, 'credentials', GOOGLE_APPLICATION_CREDENTIALS)

