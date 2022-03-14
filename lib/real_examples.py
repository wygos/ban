import shap
import xgboost as xgb
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sklearn
import matplotlib.pyplot as pl
import numpy as np
from tqdm import tqdm
import pandas as pd
import lib.loadnhanes as loadnhanes
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def get_boston(model_type):
    assert(model_type in ['GBDT','DT'])

    NAME=f'boston_{model_type}'
    X,y = shap.datasets.boston()
    if model_type == 'DT':
        max_depth=10
        NAME=f'{NAME}_{max_depth}'
        model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=1)
        model.fit(X, y)
    else:
        model = xgb.train({"learning_rate": 0.01, 'seed': 42}, xgb.DMatrix(X, label=y), 100)
    return NAME, X, model



def get_health_insurance(model_type):
    assert(model_type in ['GBDT','DT'])
    NAME=f'health_insurance_{model_type}'

    health = pd.read_csv('../data/health_insurance/train.csv')
    X = health.drop(columns=['Response', 'id'])
    y = health['Response']

    for cat in ['Gender', 'Vehicle_Age', 'Vehicle_Damage']:
        X = pd.concat([X.drop(columns=[cat]), X[cat].str.get_dummies(sep=',')], axis=1, sort=False)
    X.columns = [c.replace("<", " less ") for c in X.columns]
    X.columns = [c.replace(">", " more ") for c in X.columns]
    X.columns = [c.replace(" ", "_") for c in X.columns]

    param = {
        'eta': 0.2,
        'max_depth': 4,
        'objective' : 'binary:logistic',
        'seed': 42,
    }
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'GBDT':
        D_train = xgb.DMatrix(X_train, label=Y_train)
        D_test = xgb.DMatrix(X_test, label=Y_test)
        steps=250
        model = xgb.train(param, D_train, steps, [(D_test,'eval'), (D_train, 'train')], verbose_eval=50)
    else:
        max_depth=60
        NAME  = f'{NAME}_{max_depth}'
        model = RandomForestClassifier(max_depth=max_depth, random_state=0, n_estimators=1)
        model.fit(X_train, Y_train)
        # Use the forest's predict method on the test data
        predictions = model.predict_proba(X_test)# Calculate the absolute errors
        print('Loss:', sklearn.metrics.log_loss(Y_test, predictions))


    return NAME, X_train, model


def get_flights(model_type):
    assert(model_type in ['GBDT','DT'])
    NAME=f'flights_{model_type}'
    flights=pd.read_csv('../data/flights/DelayedFlights.csv')
    flights.drop(columns=['Unnamed: 0', 'TaxiOut', 'ActualElapsedTime', 'TailNum', 'ArrDelay', 'NASDelay', 'LateAircraftDelay', 'CarrierDelay', 'WeatherDelay', 'SecurityDelay', 'Diverted', 'Cancelled'], inplace=True)

    flights['target'] = flights.ArrTime- flights.CRSArrTime

    flights.drop(columns=['ArrTime'], inplace=True)
    flights = flights[~flights['target'].isnull()]

    x_all = flights.drop(columns=['target'])
    y = flights['target']

    for cat in ['UniqueCarrier', 'Origin', 'Dest', 'CancellationCode']:
        x_all = pd.concat([x_all.drop(columns=[cat]),
                       x_all[cat].str.get_dummies(sep=',').rename(columns=lambda c:cat+"_"+c)
                      ], axis=1, sort=False)

    param = {
        'eta': 0.2,
        'max_depth': 10,
        'objective' : 'reg:squarederror',
        'seed': 42,
    }
    X_train, X_test, Y_train, Y_test = train_test_split(x_all, y, test_size=0.2)
    if model_type == 'GBDT':
        D_train = xgb.DMatrix(X_train, label=Y_train)
        D_test = xgb.DMatrix(X_test, label=Y_test)
        steps=250
        model = xgb.train(param, D_train, steps, [(D_test,'eval'), (D_train, 'train')], verbose_eval=50)
    else:
        max_depth=100
        NAME  = f'{NAME}_{max_depth}'
        X_train.fillna(-1, inplace=True)
        X_test.fillna(-1, inplace=True)
        model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=1)
        model.fit(X_train, Y_train)

    return NAME, X_train.sample(n=10000, random_state=1), model



def get_nhanes(model_type):
    assert(model_type in ['GBDT','DT'])
    NAME=f'nhanes_{model_type}'
    X,y = loadnhanes._load()

    # clean up a bit
    for c in X.columns:
        if c.endswith("_isBlank"):
            del X[c]
    X["bmi"] = 10000 * X["weight"].values.copy() / (X["height"].values.copy() * X["height"].values.copy())
    del X["weight"]
    del X["height"]
    del X["urine_hematest_isTrace"] # would have no variance in the strain set
    del X["SGOT_isBlankbutapplicable"] # would have no variance in the strain set
    del X["calcium_isBlankbutapplicable"] # would have no variance in the strain set
    del X["uric_acid_isBlankbutapplicable"] # would only have one true value in the train set
    del X["urine_hematest_isVerylarge"] # would only have one true value in the train set
    del X["total_bilirubin_isBlankbutapplicable"] # would only have one true value in the train set
    del X["alkaline_phosphatase_isBlankbutapplicable"] # would only have one true value in the train set
    del X["hemoglobin_isUnacceptable"] # redundant with hematocrit_isUnacceptable
    rows = np.where(np.invert(np.isnan(X["systolic_blood_pressure"]) | np.isnan(X["bmi"])))[0]
    X = X.iloc[rows,:]
    y = y[rows]

    name_map = {
        "sex_isFemale": "Sex",
        "age": "Age",
        "systolic_blood_pressure": "Systolic blood pressure",
        "bmi": "BMI",
        "white_blood_cells": "White blood cells", # (mg/dL)
        "sedimentation_rate": "Sedimentation rate",
        "serum_albumin": "Blood albumin",
        "alkaline_phosphatase": "Alkaline phosphatase",
        "cholesterol": "Total cholesterol",
        "physical_activity": "Physical activity",
        "hematocrit": "Hematocrit",
        "uric_acid": "Uric acid",
        "red_blood_cells": "Red blood cells",
        "urine_albumin_isNegative": "Albumin present in urine",
        "serum_protein": "Blood protein"
    }
    mapped_feature_names = list(map(lambda x: name_map.get(x, x), X.columns))

    # split by patient id
    pids = np.unique(X.index.values)
    train_pids,test_pids = train_test_split(pids, random_state=0)
    strain_pids,valid_pids = train_test_split(train_pids, random_state=0)

    # find the indexes of the samples from the patient ids
    train_inds = np.where([p in train_pids for p in X.index.values])[0]
    strain_inds = np.where([p in strain_pids for p in X.index.values])[0]
    valid_inds = np.where([p in valid_pids for p in X.index.values])[0]
    test_inds = np.where([p in test_pids for p in X.index.values])[0]

    # create the split datasets
    X_train = X.iloc[train_inds,:]
    X_strain = X.iloc[strain_inds,:]
    X_valid = X.iloc[valid_inds,:]
    X_test = X.iloc[test_inds,:]
    y_train = y[train_inds]
    y_strain = y[strain_inds]
    y_valid = y[valid_inds]
    y_test = y[test_inds]

    # mean impute for linear and deep models
    imp = SimpleImputer()
    imp.fit(X_strain)
    X_strain_imp = imp.transform(X_strain)
    X_train_imp = imp.transform(X_train)
    X_valid_imp = imp.transform(X_valid)
    X_test_imp = imp.transform(X_test)
    X_imp = imp.transform(X)

    # standardize
    scaler = StandardScaler()
    scaler.fit(X_strain_imp)
    X_strain_imp = scaler.transform(X_strain_imp)
    X_train_imp = scaler.transform(X_train_imp)
    X_valid_imp = scaler.transform(X_valid_imp)
    X_test_imp = scaler.transform(X_test_imp)
    X_imp = scaler.transform(X_imp)



    # these parameters were found using the Tune XGboost on NHANES notebook (coordinate decent)
    params = {
        "learning_rate": 0.001,
        "n_estimators": 6765,
        "max_depth": 4,
        "subsample": 0.5,
        "reg_lambda": 5.5,
        "reg_alpha": 0,
        "colsample_bytree": 1
    }
    for col in X_train.columns:
        if X_train[col].dtype == np.dtype('bool'):
            print("casting ", col)
            X_strain[col] = X_strain[col].astype('int32')
            X_test[col] = X_test[col].astype('int32')
            X_valid[col] = X_valid[col].astype('int32')

    X_strain =  X_strain.fillna(-1)

    if model_type == 'DT':
        max_depth=40
        NAME  = f'{NAME}_{max_depth}'
        model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=1)
        model.fit(X_strain, y_strain)

    else:
        xgb_model = xgb.XGBRegressor(
        max_depth=params["max_depth"],
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],#math.pow(10, params["learning_rate"]),
        subsample=params["subsample"],
        reg_lambda=params["reg_lambda"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        n_jobs=16,
        random_state=1,
        objective="survival:cox",
        base_score=1
        )
        xgb_model.fit(
            X_strain, y_strain, verbose=500,
            eval_set=[(X_valid, y_valid)],
           #eval_metric="logloss",
           early_stopping_rounds=10000
        )
        model = xgb_model.get_booster()

    return NAME, X_strain, model

