import dill
import datetime
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, cross_val_score


TRAIN_DATA_PATH = '../../data/train.csv'
CONT_FEATS_TO_EXCLUDE = ['HasAlley', 'HasMasVnr', 'HasBsmt', 'HasFireplace', 'HasGarage', 'HasPool', 'HasFence',
                         'HasPorch', 'IsRichNeighborhood', 'SalePrice']

params_grid = {
    'Ridge': {
        'regressor__alpha': [0.01, 0.1, 1.0, 10, 100],
        'regressor__max_iter': [10000, 50000, 100000],
        'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    },
    'Lasso': {
        'regressor__alpha': [0.01, 0.1, 1.0, 10, 100],
        'regressor__max_iter': [10000, 50000, 100000],
        'regressor__tol': [1e-4, 1e-3, 1e-2]
    },
    'GradientBoostingRegressor': {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__learning_rate': [0.01, 0.1, 0.2, 0.3],
        'regressor__max_depth': [3, 5, 7],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__subsample': [0.8, 1.0],
        'regressor__max_features': ['sqrt', 'log2']
    }
}


gb_param_grid = {
    'n_estimators': [50, 100, 200],                # Number of boosting stages
    'learning_rate': [0.01, 0.1, 0.2, 0.3],        # Learning rate
    'max_depth': [3, 5, 7],                        # Maximum depth of the individual estimators
    'min_samples_split': [2, 5, 10],               # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],                 # Minimum samples required to be a leaf node
    'subsample': [0.8, 1.0],                       # Fraction of samples to use for training each tree
    'max_features': ['auto', 'sqrt', 'log2']       # Number of features to consider when looking for best split
}

# Dictionary for filling NaN values
fill_na_strings = {
    'PoolQC': 'No Pool', 'MiscFeature': 'No feature', 'Alley': 'No Alley', 'Fence': 'No Fence',
    'MasVnrType': 'No masonry veneer', 'FireplaceQu': 'No fireplace', 'GarageType': 'No Garage',
    'GarageQual': 'No Garage', 'GarageFinish': 'No Garage', 'GarageCond': 'No Garage',
    'BsmtCond': 'No Basement', 'BsmtQual': 'No Basement', 'BsmtFinType1': 'No Basement',
    'BsmtFinType2': 'No Basement', 'BsmtExposure': 'No Basement'
}


# Fill missing values
def fill_na(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    df_upd = df_upd.fillna(fill_na_strings)
    df_upd['GarageYrBlt'] = df_upd['GarageYrBlt'].fillna(0)
    df_upd['MasVnrArea'] = df_upd['MasVnrArea'].fillna(0)
    df_upd['Electrical'] = df_upd['Electrical'].fillna(df_upd['Electrical'].mode()[0])
    df_upd['LotFrontage'] = df_upd.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    return df_upd


# Create new age and date-related features
def create_new_age_features(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    df_upd['HouseAge'] = df_upd['YrSold'] - df_upd['YearBuilt']
    df_upd['YearsSinceRemodel'] = df_upd['YrSold'] - df_upd['YearRemodAdd']
    df_upd['WasRemodeled'] = (df_upd['YearRemodAdd'] > df_upd['YearBuilt']).astype(int)
    df_upd['GarageAge'] = df_upd['YrSold'] - df_upd['GarageYrBlt']
    df_upd['HouseAgeCategory'] = pd.cut(df_upd['HouseAge'], bins=[0, 10, 20, 50, 100, 200],
                                        labels=['New', 'Recent', 'Modern', 'Old', 'Very Old'])
    season_map = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer',
                  7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'}
    df_upd['SeasonSold'] = df_upd['MoSold'].map(season_map)
    return df_upd


# Create new quality/condition-related features
def create_new_qual_features(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    df_upd['QualCondMult'] = df_upd['OverallQual'] * df_upd['OverallCond']
    df_upd['QualCondRatio'] = df_upd['OverallQual'] / df_upd['OverallCond']

    qc_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    qc_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual',
                   'GarageQual', 'GarageCond', 'PoolQC', 'FireplaceQu']

    for feature in qc_features:
        df_upd[feature] = df_upd[feature].map(qc_mapping).fillna(0)
    df_upd['OverallQualScore'] = df_upd[qc_features].sum(axis=1)
    return df_upd


# Create "Has..." features
def create_new_has_features(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    df_upd['HasAlley'] = (df_upd['Alley'] != 'No Alley').astype(int)
    df_upd['HasMasVnr'] = (df_upd['MasVnrType'] != 'No masonry veneer').astype(int)
    df_upd['HasBsmt'] = (df_upd['BsmtQual'] > 0).astype(int)
    df_upd['HasFireplace'] = (df_upd['FireplaceQu'] > 0).astype(int)
    df_upd['HasGarage'] = (df_upd['GarageQual'] > 0).astype(int)
    df_upd['HasPool'] = (df_upd['PoolQC'] > 0).astype(int)
    df_upd['HasFence'] = (df_upd['Fence'] != 'No Fence').astype(int)
    df_upd['HasPorch'] = df_upd[['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']].sum(axis=1).gt(0).astype(
        int)
    return df_upd


# Create new footage/area-related features
def create_new_footage_features(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    df_upd['TotalSF'] = df_upd[['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']].sum(axis=1)
    df_upd['TotalPorchSF'] = df_upd[['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']].sum(axis=1)
    df_upd['TotalBsmtFinSF'] = df_upd[['BsmtFinSF1', 'BsmtFinSF2']].sum(axis=1)
    df_upd['HouseToLotRatio'] = df_upd['GrLivArea'] / df_upd['LotArea']
    return df_upd


# Create neighborhood-related feature
def create_new_neighb_features(df: pd.DataFrame, sale_price: pd.Series) -> pd.DataFrame:
    df_upd = df.copy()
    rich_neighborhoods = sale_price.groupby(df_upd['Neighborhood']).mean().nlargest(10).index
    df_upd['IsRichNeighborhood'] = df_upd['Neighborhood'].isin(rich_neighborhoods).astype(int)
    return df_upd


# Change data type of specific columns
def change_type(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    df_upd['MSSubClass'] = df_upd['MSSubClass'].astype('object')
    return df_upd


# Select relevant continuous features
def custom_column_selector(dtype_include=None, dtype_exclude=None):
    selector = make_column_selector(dtype_include=dtype_include, dtype_exclude=dtype_exclude)
    return lambda df: [col for col in selector(df) if col not in CONT_FEATS_TO_EXCLUDE]


# Define main pipeline logic
def main():
    # Load dataset
    df = pd.read_csv(TRAIN_DATA_PATH)
    x = df.drop(columns=['Id', 'SalePrice'], axis=1)
    y = df['SalePrice']

    # Create preprocessor pipeline for filling NaN values and creating new features
    preprocessor = Pipeline(steps=[
        ('fill_na', FunctionTransformer(fill_na)),
        ('create_age_features', FunctionTransformer(create_new_age_features)),
        ('create_new_qual_features', FunctionTransformer(create_new_qual_features)),
        ('create_has_features', FunctionTransformer(create_new_has_features)),
        ('create_new_footage_features', FunctionTransformer(create_new_footage_features)),
        ('create_new_neighb_features', FunctionTransformer(lambda df: create_new_neighb_features(df, y))),
        ('change_type', FunctionTransformer(change_type))
    ])

    # Define transformation logic for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Define transformation logic for continuous features
    continuous_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('std_scaler', StandardScaler())
    ])

    # Create ColumnTransformer
    column_transformer = ColumnTransformer(transformers=[
        ('categorical_transformer', categorical_transformer, make_column_selector(dtype_include=['object', 'category'])),
        ('continuous_transformer', continuous_transformer, custom_column_selector(dtype_include=['float64', 'int64']))
    ])

    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'GradientBoostingRegressor': GradientBoostingRegressor()
    }

    best_rmse = float('inf')
    best_model = None
    for model_name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('column_transformer', column_transformer),
            ('regressor', model)
        ])

        grid_search = GridSearchCV(pipeline, params_grid[model_name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(x, y)

        best_pipeline = grid_search.best_estimator_

        cv_scores = cross_val_score(best_pipeline, x, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        rmse_score = np.sqrt(-cv_scores.mean())

        if rmse_score < best_rmse:
            best_rmse = rmse_score
            best_model = best_pipeline

    print(f'Best model: {type(best_model.named_steps["regressor"]).__name__} with RMSE: {best_rmse:.3f}')

    with open('house_prices_regressor.pkl', 'wb') as f:
        dill.dump({
            'model': best_model,
            'metadata': {
                'name': 'House Prices Regressor',
                'author': 'Vasilii Tokarev',
                'version': '1.0',
                'date': datetime.datetime.now().strftime('%Y-%m-%d'),
                'type': type(best_model.named_steps['regressor']).__name__,
                'RMSE': best_rmse
            }
        }, file=f, recurse=True)


if __name__ == '__main__':
    main()
