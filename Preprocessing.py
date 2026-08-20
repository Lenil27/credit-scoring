import numpy as np
import pandas as pd
from typing import Any, Optional, List, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, FunctionTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class DataFrameImputer(BaseEstimator, TransformerMixin):
    """Класс для заполнения пропусков с сохранением формата DataFrame."""
    def __init__(self, strategy: str = 'median', fill_value: Any = None) -> None:
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputer_: Optional[SimpleImputer] = None
        self.columns_: Optional[pd.Index] = None
    
    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> 'DataFrameImputer':
        self.columns_ = X.columns
        if self.strategy == 'constant':
            self.imputer_ = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)
        else:
            self.imputer_ = SimpleImputer(strategy=self.strategy)
        self.imputer_.fit(X) 
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.imputer_ is None:
            raise RuntimeError("Imputer has not been fitted yet.")
        X_imputed = self.imputer_.transform(X)
        return pd.DataFrame(X_imputed, columns=self.columns_, index=X.index)
    
class DataFrameOutlierClipper(BaseEstimator, TransformerMixin):
    """Класс для обрезки выбросов (поддерживает квантили и IQR)."""
    def __init__(self, method: str = 'none', lower_q: float = 0.01, upper_q: float = 0.99, iqr_factor: float = 1.5) -> None:
        self.method = method
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.iqr_factor = iqr_factor
        self.lower_bounds_: Optional[pd.Series] = None
        self.upper_bounds_: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> 'DataFrameOutlierClipper':
        if self.method == 'clip_quantile':
            self.lower_bounds_ = X.quantile(self.lower_q)
            self.upper_bounds_ = X.quantile(self.upper_q)
        elif self.method == 'clip_iqr':
            q25 = X.quantile(0.25)
            q75 = X.quantile(0.75)
            iqr = q75 - q25
            self.lower_bounds_ = q25 - self.iqr_factor * iqr
            self.upper_bounds_ = q75 + self.iqr_factor * iqr
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.method == 'none':
            return X.copy()
            
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("Clipper has not been fitted yet.")
            
        X_clipped = X.copy()
        for col in X.columns:
            if col in self.lower_bounds_ and col in self.upper_bounds_:
                X_clipped[col] = X_clipped[col].clip(
                    lower=self.lower_bounds_[col], 
                    upper=self.upper_bounds_[col]
                )
        return X_clipped

class DataFrameScaler(BaseEstimator, TransformerMixin):
    """Класс для масштабирования данных на выбор."""
    def __init__(self, scaler_type: str = 'standard') -> None:
        self.scaler_type = scaler_type
        self.scaler_: Optional[Any] = None
        self.columns_: Optional[pd.Index] = None

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> 'DataFrameScaler':
        self.columns_ = X.columns
        
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'log': FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=False),
            'none': FunctionTransformer(validate=False)
        }
        
        self.scaler_ = scalers.get(self.scaler_type, StandardScaler())
        self.scaler_.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        X_scaled = self.scaler_.transform(X)
        return pd.DataFrame(X_scaled, columns=self.columns_, index=X.index)
    
class UniversalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 numeric_features: List[str],
                 categorical_features: List[str],
                 numeric_config: Optional[Dict[str, list]] = None,
                 categorical_config: Optional[Dict[str, list]] = None,
                 default_num_strategy: Optional[list] = None,
                 default_cat_strategy: Optional[list] = None,
                 ignore_scaler: bool = False):
        
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.numeric_config = numeric_config if numeric_config is not None else {}
        self.categorical_config = categorical_config if categorical_config is not None else {}
        self.default_num_strategy = default_num_strategy if default_num_strategy is not None else ['standard', 'median', 'clip_quantile']
        self.default_cat_strategy = default_cat_strategy if default_cat_strategy is not None else ['ohe', 'most_frequent']
        self.ignore_scaler = ignore_scaler

        self.transformers_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in self.numeric_features:
            strategy = self.numeric_config.get(col, self.default_num_strategy)

            scaler_type = 'none' if self.ignore_scaler else strategy[0]
            imputer_strategy = strategy[1]
            outlier_method = strategy[2] if len(strategy) > 2 else 'none'

            col_pipeline = Pipeline([
                ('imputer', DataFrameImputer(strategy=imputer_strategy)),
                ('outliers', DataFrameOutlierClipper(method=outlier_method)),
                ('scaler', DataFrameScaler(scaler_type=scaler_type))
            ])
            col_pipeline.fit(X[[col]])
            self.transformers_[col] = col_pipeline

        encoder_mapping = {
            'ohe': OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
            'ordinal': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
        }

        for col in self.categorical_features:
            strategy = self.categorical_config.get(col, self.default_cat_strategy)
            enc_type = strategy[0]
            impute_strat = strategy[1]
            fill_val = strategy[2] if len(strategy) > 2 else None

            encoder = encoder_mapping[enc_type]

            col_pipeline = Pipeline([
                ('imputer', DataFrameImputer(strategy=impute_strat, fill_value=fill_val)),
                ('encoder', encoder)
            ])
            col_pipeline.fit(X[[col]])
            self.transformers_[col] = col_pipeline

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed_arrays = []
        all_columns = []

        # Обработка числовых признаков
        for col in self.numeric_features:
            pipeline = self.transformers_[col]
            transformed_df = pipeline.transform(X[[col]])
            
            transformed_arrays.append(transformed_df.to_numpy())
            all_columns.extend(transformed_df.columns.tolist())

        # Обработка категориальных признаков
        for col in self.categorical_features:
            pipeline = self.transformers_[col]
            transformed_arr = pipeline.transform(X[[col]])
            encoder = pipeline.named_steps['encoder']
            
            if hasattr(encoder, 'get_feature_names_out'):
                col_names = encoder.get_feature_names_out([col])
            else:
                col_names = [col]

            if isinstance(transformed_arr, pd.DataFrame):
                transformed_arrays.append(transformed_arr.to_numpy())
            else:
                transformed_arrays.append(transformed_arr)
                
            all_columns.extend(list(col_names))
            
        if not transformed_arrays:
            return pd.DataFrame(index=X.index)
            
        final_array = np.hstack(transformed_arrays)
        return pd.DataFrame(final_array, columns=all_columns, index=X.index)
    
    @classmethod
    def from_yaml(cls, path: str, ignore_scaler: bool = False):
        """Создает препроцессор напрямую из YAML-файла конфигурации."""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        return cls(
            numeric_features=config.get("numeric_features", []),
            categorical_features=config.get("categorical_features", []),
            numeric_config=config.get("numeric_config"),
            categorical_config=config.get("categorical_config"),
            default_num_strategy=config.get("default_num_strategy"),
            default_cat_strategy=config.get("default_cat_strategy"),
            ignore_scaler=ignore_scaler
        )