#python 3.8.4
#./venv/bin/python
# %%
import os
import pandas as pd 
from google.cloud import bigquery
from pandas.io import gbq
from consumer_complaint.config import config

# %%
class BigQueryConnection:
    def __init__(self, project_id):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.GOOGLE_CREDENTIAL_PATH
        self.client = bigquery.Client()
        self.project_id = project_id

    def get_public_sql_result(self, query):
        query_job = self.client.query(query = query)
        print("The query data:")
        for row in query_job:
            # Row values can be accessed by field name or index.
            print("name={}, count={}".format(row[0], row["total_people"]))

    def get_private_sql_df(self, query):
        data_df = gbq.read_gbq(query = query, 
                                project_id = self.project_id)
        return data_df


# %%
if __name__ == '__main__':
    consumer_complaint_df = pd.read_csv(config.FILE_PATH, encoding = 'utf-8')

# %%
    consumer_complaint_df.to_gbq(destination_table = "consumer_complaint.consumer_complaint_data",
                                project_id = config.GCP_PROJECT_ID,
                                if_exists = 'replace')


# %%
    big_query = BigQueryConnection(project_id = config.GCP_PROJECT_ID)
    query = """
    SELECT * 
    FROM `consumer-complaint-310721.consumer_complaint.consumer_complaint_data` 
    LIMIT 1000;
    """
    result_df = big_query.get_private_sql_df(query = query)



# %%



