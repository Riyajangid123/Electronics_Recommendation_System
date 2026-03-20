import pandas as pd

class DataIngestion:
    def Ingest_Data(self,path):
        data=pd.read_csv(path)
        return data