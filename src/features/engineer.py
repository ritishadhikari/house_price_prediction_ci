import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s -(levelname)s - %(message)s'
)
logger=logging.getLogger('feature-engineering')

def create_features(df:pd.DataFrame):
    """
    Create new features from existing data
    """
    logger.info("Creating new features")
    df_featured=df.copy()
    
    current_year=datetime.now().year
    df_featured['house_age']=current_year-df_featured['year_built']
    logger.info('Created "house_age" features')

    df_featured['price_per_sqft']=df_featured['price']/df_featured['sqft']
    logger.info('Created "price_per_sqft" feature')

    df_featured['bed_bath_ratio']=df_featured['bedrooms']/df_featured['bathrooms']
    df_featured['bed_bath_ratio']=df_featured['bed_bath_ratio'].replace(
    to_replace=[-np.inf,np.inf],value=np.nan).fillna(0)
    logger.info('Created "bed_bath_ratio" feature')

    return df_featured

def create_preprocessor():
    """
    Create a preprocessing pipeline
    """
    logger.info('Creating Preprocessor Pipeline')
    categorical_features=['location','condition']
    numerical_features=['sqft','bedrooms','bathrooms','house_age','price_per_sqft','bed_bath_ratio']
    numerical_transformer=Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='mean'))
    ])
    categorical_transformer=Pipeline(
        steps=[
            ('onehot',OneHotEncoder(handle_unknown='ignore'))
        ]
    )
    preprocessor=ColumnTransformer(
        transformers=[
            ("num", numerical_transformer ,numerical_features),
            ("cat",categorical_transformer, categorical_features)
        ]
    )
    return preprocessor
    
    
def run_feature_engineering(input_file,output_file,preprocessor_file):
    """
    Full Feature Engineering Pipeline
    """

    logger.info(f"Loading data from {input_file}")
    df=pd.read_csv(input_file)

    # Create Features
    df_featured=create_features(df=df)

    logger.info(f"Created Featured Dataset with shape: {df_featured.shape}")

    preprocessor=create_preprocessor()
    X=df_featured.drop(columns=['price'],errors='ignore')  # features only
    y=df_featured['price'] if 'price' in df_featured.columns else None  # target column if available
    x_transformed=preprocessor.fit_transform(X)
    logger.info(f"Fitted the preprocessor and transformed the features")
    joblib.dump(preprocessor,preprocessor_file)
    logger.info(f"Saved Preprocessor to {preprocessor_file}")

    df_transformed=pd.DataFrame(data=x_transformed)
    if y is not None:
        df_transformed['price']=y.values
    df_transformed.to_csv(output_file,index=False)
    logger.info(f"Saved fully preprocessed data to {output_file}")
    
    return df_transformed
    
if __name__=="__main__":
    import argparse

    parser=argparse.ArgumentParser(description='Feature Engineering for Housing Data')
    parser.add_argument('--input', required=True, help='Path to cleaned CSV file')
    parser.add_argument('--output', required=True, help='Path for output CSV file (engineered features)')
    parser.add_argument('--preprocessor', required=True, help='Path for saving the Preprocessor')
    args=parser.parse_args()

    run_feature_engineering(
        input_file=args.input,
        output_file=args.output,
        preprocessor_file=args.preprocessor
    )