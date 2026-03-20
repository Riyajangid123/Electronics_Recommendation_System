from src.components.DataIngestion import DataIngestion
from src.components.DataPreprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer

class Train_pipeline:
    def run_pipeline(self):
        Ingestion=DataIngestion()
        data=Ingestion.Ingest_Data(r"C:\Users\DELL\OneDrive\Desktop\Electronics_recommendation_system\data\All Electronics.csv")

        print("Data Ingestion Complete")

        preprocessing=DataPreprocessing()
        x,y,transformer=preprocessing.preprocess_data(data)

        print("Data Preprocessing Done")

        trainer=ModelTrainer()
        trainer.TrainModel(x,y,transformer)

        print("Model Training Done!")
if __name__=="__main__":

    pipeline=Train_pipeline()
    pipeline.run_pipeline()
