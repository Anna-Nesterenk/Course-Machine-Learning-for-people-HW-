import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Dict, Any

def columns_to_drop(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Видаляємо стовпці, що не несуть інформаційного значення для подальшої роботи
    """
    return df.drop(subset=columns, axis=1)

def split_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
   Розділяємо дані на трейнові і валідаційні
    """
    train_df, val_df = train_test_split(df, test_size=0.25, random_state=42)
    return {'train': train_df, 'val': val_df}

def create_inputs_targets(df_dict: Dict[str, pd.DataFrame], input_cols: list, target_col: str) -> Dict[str, Any]:
    """
    Створюємо цільові і не цільові колонки
    """
    data = {}
    for split in df_dict:
        data[f'{split}_inputs'] = df_dict[split][input_cols].copy()
        data[f'{split}_targets'] = df_dict[split][target_col].copy()
    return data


def scale_numeric_features(data: Dict[str, Any], numeric_cols: list) -> None:
    """
    Маштабуємо числові ознаки
    """
    scaler = MinMaxScaler().fit(data['train_inputs'][numeric_cols])
    for split in ['train', 'val', 'test']:
        data[f'{split}_inputs'][numeric_cols] = scaler.transform(data[f'{split}_inputs'][numeric_cols])


def encode_categorical_features(data: Dict[str, Any], categorical_cols: list) -> None:
    """
   Кодуємо категоріальні дані
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(data['train_inputs'][categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    for split in ['train', 'val', 'test']:
        encoded = encoder.transform(data[f'{split}_inputs'][categorical_cols])
        data[f'{split}_inputs'] = pd.concat([data[f'{split}_inputs'], pd.DataFrame(encoded, columns=encoded_cols, index=data[f'{split}_inputs'].index)], axis=1)
        data[f'{split}_inputs'].drop(columns=categorical_cols, inplace=True)
    data['encoded_cols'] = encoded_cols

def preprocess_data(raw_df: pd.DataFrame, scaler_numeric = False) -> Dict[str, Any]:
    """
    Обробляємо "сирі" дані
    """
    raw_df = columns_to_drop(raw_df, ['Surname', 'CustomerId'])
    split_dfs = split_data(raw_df)
    input_cols = list(raw_df.columns)[1:-1]
    target_col = 'Exited'
    data = create_inputs_targets(split_dfs, input_cols, target_col) 
    
    numeric_cols = data['train_inputs'].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data['train_inputs'].select_dtypes('object').columns.tolist()

    if scaler_numeric:
        scale_numeric_features(data, numeric_cols)
    else:
        pass
        
    encode_categorical_features(data, categorical_cols)

    # Extract X_train, X_val
    X_train = data['train_inputs'][numeric_cols + data['encoded_cols']]
    X_val = data['val_inputs'][numeric_cols + data['encoded_cols']]

    return {
        'X_train': X_train,
        'train_targets': data['train_targets'],
        'X_val': X_val,
        'val_targets': data['val_targets'],
        'input_cols': input_cols,
        'scaler': scaler,
        'encoder': encoder
    }