
import yaml
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

# sklearn related imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, ConfusionMatrixDisplay
import xgboost as xgb
import joblib

# Import helper functions from other modules
from db import read_data_from_sqlite
from data_cleaning import clean_df


def get_feature_list(config):
    """Extract the Feature list from the config file

    Args:
        config (dictionary): Dictionary containing the configuration file

    Returns:
        tuple : A tuple containing numeric_features,categorical_features,ordinal_features,
        ordinal_features_maps, target
    """
    # Getting the features list from config file
    numeric_features = config['features']['numeric_features']
    categorical_features = config['features']['categorical_features']
    ordinal_features_dict = config['features']['ordinal_features']

    ordinal_features = []
    ordinal_features_maps = []

    if ordinal_features_dict is not None:
        for keys,values in ordinal_features_dict.items():
            ordinal_features.append(keys)
            ordinal_features_maps.append(values)
    
    target = config['features']['target']

    return numeric_features,categorical_features,ordinal_features,ordinal_features_maps,target

def make_pipeline(numeric_features,categorical_features,ordinal_features,ordinal_features_maps,algorithms):
    """Function to construct the pipeline for the algorithms

    Args:
        numeric_features (list): List of numeric features
        categorical_features (list): List of categorical features
        ordinal_features (list): List of ordinal features
        ordinal_features_maps (list): Maps of ordinal features
        algorithms (dict): Dict of algorithms from config file

    Returns:
        list: A list containing model name, model pipeline and param_grid for each model to run
    """
    
    # Set up the preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=ordinal_features_maps))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('ord', ordinal_transformer, ordinal_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Set up the models to train
    models = []

    for algorithm_name, algorithm_values in algorithms.items():
        if algorithm_values['to_run']:
            if ('param_grid' in algorithm_values) and (algorithm_values['param_grid'] is not None):
                param_grid = algorithm_values['param_grid']
                for param_grid_key in list(param_grid.keys()):
                    param_grid['clf__'+param_grid_key] = param_grid.pop(param_grid_key)
            else:
                param_grid = {}
        
            if ('init_parameters' in algorithm_values) and (algorithm_values['init_parameters'] is not None):
                init_parameters = algorithm_values['init_parameters']
            else:
                init_parameters = {}
            if algorithm_name == 'LogisticRegression':
                model = LogisticRegression(**init_parameters)
            elif algorithm_name == 'RandomForest':
                model = RandomForestClassifier(**init_parameters)
            elif algorithm_name == 'XGBoost':
                model = xgb.XGBClassifier(**init_parameters)
            elif algorithm_name =='KNNeighbors':
                model = KNeighborsClassifier(**init_parameters)
            else:
                print(f'Unknown model name {algorithm_name} specified in config file')


            models.append(( algorithm_name, Pipeline([
                ('preprocessor', preprocessor),
                ('clf', model)
            ]), param_grid))

    return models

def plot_characteristic_curves(model, X, y, output_path, model_name):
    """Plot and save AUROC and PR curvers

    Args:
        model (scikit learn model): Scikit model classifier to predict
        X (pandas DataFrame): Input Features
        y (1d array): Output label
        output_path (Path): Output path to store the image
        model_name (string): Model name to append to image
    """
    y_pred_prob = model.predict_proba(X)[:,1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds_auc = roc_curve(y, y_pred_prob)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(output_path.joinpath(f'ROC_curve_{model_name}.png'))
    plt.close() 
    # Calculate AUC score
    print('Test Scores')
    auc_score = roc_auc_score(y, y_pred_prob)
    print('AUC score:', auc_score)


    precision, recall, thresholds_pr = precision_recall_curve(y, y_pred_prob)
    f1score = (2 * precision * recall) / (precision + recall)
    index = np.argmax(f1score)

    # Plot PR curve
    plt.figure()
    plt.plot(precision, recall)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('PR Curve')
    plt.savefig(output_path.joinpath(f'PR_curve_{model_name}.png'), dpi=200)
    plt.close()
    print(f'F-Score={f1score[index]}')
    best_threshold = thresholds_pr[index]
    print(f'Best Threshold based on F1-score={best_threshold}')

    # Get output predicitons
    y_pred = (y_pred_prob>=best_threshold).astype(int)

    print('Confusion matrix')
    cm = confusion_matrix(y, y_pred)
    print(cm)

    plt.figure()
    disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, normalize= 'all')
    disp.plot()
    plt.savefig(output_path.joinpath(f'Confusion_Matrix_{model_name}.png'))
    
    return(best_threshold)

def train_and_evaluate(models, X_train, y_train, X_test, y_test, output_folder):
    """Trains and evaluates the list of model files

    Args:
        models (list): List of models to run outputted by the make_pipeline function
        X_train (pandas DataFrame): Training Features
        y_train (pandas DataFrame): Training target label
        X_test (pandas DataFrame): Test Features
        y_test (pandas DataFrame): Test target label
        output_folder (str): Path to output folder

    Returns:
        tuple: tuple containing trained_models, best_model, best_model_name, best_threshold
    """
    
    # Create Output directories
    folder_name = Path(output_folder)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = folder_name.joinpath(f'results_{timestamp}')
    os.makedirs(output_dir)

    trained_models = {}
    best_scores = {}

    # Train each algorithm to choose the best algorithm
    for name, model, param_grid in models:

        # model.fit(X_train, y_train)
        grid = GridSearchCV(model, param_grid, scoring='roc_auc', verbose=1.1)
        
        # Fit the pipeline
        grid.fit(X_train, y_train)
        
        # Print the CV score
        print(f"{name} Best CV score: {grid.best_score_}")
        
        # For choosing the best model
        trained_models[name] = grid 
        best_scores[name] = grid.best_score_ 

        # Store CV results in a dataframe and dump to disk
        score_df = pd.concat((pd.DataFrame(grid.cv_results_['params']), pd.DataFrame(grid.cv_results_['mean_test_score'],columns=['Average CV score'])), axis=1)
        filename = output_dir.joinpath(Path(f"cv_results_{name}.csv"))
        score_df.to_csv(filename, index=False)

        # Save the trained model to disk
        filename = output_dir.joinpath(Path(f"model_{name}.pkl"))
        joblib.dump(model, filename)

    best_model_name = pd.Series(best_scores).idxmax()
    best_model = trained_models[best_model_name]
    
    print(f'Best model: {best_model_name}. CV score: {best_model.best_score_}')

    print('Plot Characteristic curves')
    best_threshold = plot_characteristic_curves(best_model, X_test, y_test, output_dir, best_model_name)
    

    return trained_models, best_model, best_model_name, best_threshold

def main():
    # Read the configuration file
    print('Reading Configuration File')
    with open('src/config.yaml','r') as f:
        config = yaml.safe_load(f)

    print('Reading data from Database')
    # Read the data from database
    df = read_data_from_sqlite(config['input_data']['database_path'],
                            config['input_data']['table_name'])


    print('Cleaning data')
    # Data Cleaning
    df = clean_df(df)

    print("Getting the feature list")
    numeric_features,categorical_features,ordinal_features,ordinal_features_maps,target = get_feature_list(config)
    features = numeric_features + categorical_features + ordinal_features

    print('Performing the train test split')
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target],
                                                        test_size=config['data_preparation']['test_size'],
                                                        random_state=config['data_preparation']['random_state'],
                                                        stratify = df[target])


    algorithms = config['algorithms']
    print('Construct the pipeline')
    models = make_pipeline(numeric_features,categorical_features,ordinal_features,ordinal_features_maps,algorithms)

    output_folder = config['output']['folder_name']
    print('Training the models')
    trained_models, best_model, best_model_name, best_threshold = train_and_evaluate(models, X_train, y_train, X_test, y_test, output_folder)


if __name__== '__main__':
    main()