import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.base import BaseEstimator,TransformerMixin

class OutlierCapping(BaseEstimator,TransformerMixin):
        def fit(self,x,y=None):
            x=np.array(x)
            self.q1=np.percentile(x,25,axis=0)
            self.q3=np.percentile(x,75,axis=0)
            self.IQR=self.q3-self.q1
            return self
        def transform(self,x):
            x=np.array(x)
            lower=self.q1-1.5*self.IQR
            upper=self.q3+1.5*self.IQR
            return np.clip(x,lower,upper)

class DataPreprocessing:
    def preprocess_data(self,data):
        data["discount_price"]=data["discount_price"].astype("str").str.replace("₹","").str.replace(",","").str.strip()
        data["discount_price"]=pd.to_numeric(data["discount_price"],errors="coerce")
        data["ratings"]=pd.to_numeric(data["ratings"], errors="coerce")
        data["no_of_ratings"]=data["no_of_ratings"].astype("str").str.replace(",","").str.strip()
        data["no_of_ratings"]=pd.to_numeric(data["no_of_ratings"],errors="coerce")
        data["actual_price"]=data["actual_price"].astype("str").str.replace("₹","").str.replace(",","").str.strip()
        data["actual_price"]=pd.to_numeric(data["actual_price"],errors="coerce")

        data.dropna(subset=['ratings', 'no_of_ratings'], inplace=True)

        data["discount_price"]=data["discount_price"].fillna(data["actual_price"])

        data['actual_price'] = data['actual_price'].fillna(data['actual_price'].median())
        data['discount_price'] = data['discount_price'].fillna(data['discount_price'].median())

        x = data[['actual_price', 'no_of_ratings', 'discount_price',
          'sub_category', 'main_category']]
        y=data['ratings']

        

        num_col=x.select_dtypes(exclude='object').columns
        cat_col=x.select_dtypes(include='object').columns

        num_pipeline=Pipeline(steps=[('Imputer',SimpleImputer(strategy='median')),
                                 ('Outlier',OutlierCapping()),
                                 ('Scaler',StandardScaler())])
        cat_pipeline=Pipeline(steps=[("Imputer",SimpleImputer(strategy="most_frequent")),
                                     ("Encoder",OneHotEncoder(handle_unknown="ignore"))])

        transformer=ColumnTransformer([('num',num_pipeline,num_col),
                                       ('cat',cat_pipeline,cat_col)])

        return x,y,transformer
