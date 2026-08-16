import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, FunctionTransformer, OneHotEncoder, OrdinalEncoder

class UniversalPreprocessor():
    def __init__(self, 
                 numeric_features: list,
                 categorical_features: list,
                 numeric_config: dict = None,
                 categorical_config: dict = None,
                 default_num_strategy: list = None,
                 default_cat_strategy: list = None):

        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.numeric_config = numeric_config if numeric_config is not None else {}
        self.categorical_config = categorical_config if categorical_config is not None else {}
        self.default_num_strategy = default_num_strategy if default_num_strategy is not None else ['standard', 'median']
        self.default_cat_strategy = default_cat_strategy if default_cat_strategy is not None else ['ohe', 'most_frequent']

        self.transformers_ = {}

    def fit(self, X_train):
        scaler_mapping = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'log': FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=False)
        }

        encoder_mapping = {
            'ohe': OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
            'ordinal': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
        }
        
        for col in self.numeric_features:
            strategy = self.numeric_config.get(col, self.default_num_strategy)
            imputer = SimpleImputer(strategy=strategy[1])
            scaler = scaler_mapping[strategy[0]]

            col_pipeline = Pipeline([
                ('imputer', imputer),
                ('scaler', scaler)
            ])

            col_pipeline.fit(X_train[[col]])
            self.transformers_[col] = col_pipeline

        for col in self.categorical_features:
            strategy = self.categorical_config.get(col, self.default_cat_strategy)
            
            # Исправлено: убрано дублированное переопределение импутера
            if len(strategy) > 2 and strategy[1] == 'constant':
                imputer = SimpleImputer(strategy='constant', fill_value=strategy[2])
            else:
                imputer = SimpleImputer(strategy=strategy[1])
                
            encoder = encoder_mapping[strategy[0]]

            col_pipeline = Pipeline([
                ('imputer', imputer),
                ('encoder', encoder)
            ])

            col_pipeline.fit(X_train[[col]])
            self.transformers_[col] = col_pipeline
            
        return self

    def transform(self, X_test):
        transformed_arrays = []
        all_columns = []
            
        for col in self.numeric_features:
            pipeline = self.transformers_[col]
            transformed_arr = pipeline.transform(X_test[[col]])
            transformed_arrays.append(transformed_arr)
            all_columns.append(col)
            
        for col in self.categorical_features:
            pipeline = self.transformers_[col]
            transformed_arr = pipeline.transform(X_test[[col]])
            encoder = pipeline.named_steps['encoder']
            
            if hasattr(encoder, 'get_feature_names_out'):
                col_names = encoder.get_feature_names_out([col])
            else:
                col_names = [col]

            transformed_arrays.append(transformed_arr)
            all_columns.extend(col_names)
            
        # Оптимизация: склеиваем массивы один раз через numpy, а не DataFrame в цикле
        final_array = np.hstack(transformed_arrays)
        return pd.DataFrame(final_array, columns=all_columns, index=X_test.index)