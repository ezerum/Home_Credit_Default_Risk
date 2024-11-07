from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. TODO Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.

    ### Ordinal Encoder ###
    # Create encoder
    enc_ord = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)

    # fit the encoder using data train
    ordinal_feat = working_train_df.columns[
        (working_train_df.dtypes == "object") & (working_train_df.nunique() <= 2)
    ]
    enc_ord.fit(working_train_df[ordinal_feat])

    # transform dataframes using the ordinal encoder
    working_train_df[ordinal_feat] = enc_ord.transform(working_train_df[ordinal_feat])
    working_val_df[ordinal_feat] = enc_ord.transform(working_val_df[ordinal_feat])
    working_test_df[ordinal_feat] = enc_ord.transform(working_test_df[ordinal_feat])

    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    # Take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the OrdinalEncoder and
    #     OneHotEncoder classes, then use the fitted models to transform all the
    #     datasets.

    ### CATEGORICAL ENCODER ###
    # Function to apply the transformation
    def cat_transform_enc(enc, DF, new_name_col, cat_col):
        # transform
        encoded_cols = enc.transform(DF[cat_col])
        df_enc = pd.DataFrame(encoded_cols, columns=new_name_col)
        df_oh = DF.join(df_enc)
        df_oh.drop(columns=cat_col, inplace=True)
        return df_oh

    # encoder
    enc_cat = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    # select categorical columns to encoder
    categorical_feat = working_train_df.columns[
        (working_train_df.dtypes == "object") & (working_train_df.nunique() > 2)
    ]

    # create a list of column name for each category
    cat_cols_encoded = []
    for col in categorical_feat:
        cat_cols_encoded += [
            f"{col[0]}_{cat}" for cat in list(working_train_df[col].unique())
        ]

    # fit using data train
    enc_cat.fit(working_train_df[categorical_feat])

    # transform
    working_train_df = cat_transform_enc(
        enc_cat, working_train_df, cat_cols_encoded, categorical_feat
    )
    working_val_df = cat_transform_enc(
        enc_cat, working_val_df, cat_cols_encoded, categorical_feat
    )
    working_test_df = cat_transform_enc(
        enc_cat, working_test_df, cat_cols_encoded, categorical_feat
    )

    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.

    ### SIMPLE IMPUTER ###
    # create simple imputer
    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")

    # fit using data train
    imp_mean.fit(working_train_df)

    # apply transformation to the train, val and test data
    working_train_df = imp_mean.transform(working_train_df)
    working_val_df = imp_mean.transform(working_val_df)
    working_test_df = imp_mean.transform(working_test_df)

    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.

    ### SCALER USING MINMAX ###
    # create sacaler model
    scaler_min_max = MinMaxScaler()
    # fit using data train
    scaler_min_max.fit(working_train_df)
    # transform all data
    working_train_df = scaler_min_max.transform(working_train_df)
    working_val_df = scaler_min_max.transform(working_val_df)
    working_test_df = scaler_min_max.transform(working_test_df)

    return working_train_df, working_val_df, working_test_df
