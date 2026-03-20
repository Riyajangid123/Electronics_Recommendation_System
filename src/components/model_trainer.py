import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score,KFold
from sklearn.neighbors import KNeighborsRegressor,NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib
from src.components.DataIngestion import DataIngestion

class ModelTrainer:
    def TrainModel(self,x,y,transformer):

        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
        model_pipeline=Pipeline(steps=[("preprocessor",transformer),
                                 ("model",NearestNeighbors(metric="cosine",algorithm="brute"))])
        
        accuracy_pipeline = Pipeline(steps=[
            ("preprocessor", transformer),
            ("regressor", KNeighborsRegressor(n_neighbors=5, metric="cosine", algorithm="brute"))
        ])

        accuracy_pipeline.fit(x_train,y_train)
        ingestor=DataIngestion()
        y_pred=accuracy_pipeline.predict(x_test)
        rmse=np.sqrt(mean_squared_error(y_test,y_pred))
        print(f"Accuracy Score {rmse}")

        model_pipeline.fit(x)
        df=ingestor.Ingest_Data(r"C:\Users\DELL\OneDrive\Desktop\Electronics_recommendation_system\data\All Electronics.csv")
        os.makedirs("models", exist_ok=True)

        joblib.dump(model_pipeline,"models/model.pkl")
        joblib.dump(x, "models/product_data.pkl")  

        print("Model Saved Successfully")

        return model_pipeline
