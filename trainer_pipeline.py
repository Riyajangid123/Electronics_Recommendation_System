from src.components.DataIngestion import DataIngestion
from src.components.DataPreprocessing import DataPreprocessing
from src.components.recommender import Recommender

class Train_pipeline:
    def run_pipeline(self):
        Ingestion=DataIngestion()
        data=Ingestion.Ingest_Data(r"D:\Electronics_recommendation_system\data\All Electronics.csv")

        print("Data Ingestion Complete")

        preprocessing=DataPreprocessing(data)
        preprocessed_data=preprocessing.preprocess_data()

        print("Data Preprocessing Done")

        trainer=Recommender(preprocessed_data)
        trainer.save_model()

        print("Model Training Done!")
if __name__=="__main__":

    pipeline=Train_pipeline()
    pipeline.run_pipeline()
