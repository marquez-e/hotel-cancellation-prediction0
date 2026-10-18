import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix

# Connection to Data Warehouse
url = 'postgresql://postgres:password@localhost:5432/hotel_bi'
engine = create_engine(url)

# Model training function
def run_training():
    print("Extraction of data from the Data Warehouse")

    # Get all data (past data) together in one dataframe
    df = pd.read_sql('''
                    SELECT 
                        f.*,
                        c.arrival_date, c.arrival_date_year, c.arrival_date_month, c.arrival_date_week_number,
                        cl.country, cl.customer_type, cl.is_repeated_guest, cl.previous_cancellations, cl.previous_bookings_not_canceled,
                        ch.market_segment, ch.distribution_channel, ch.agent, ch.company, ch.deposit_type,
                        r.hotel, r.reserved_room_type, r.assigned_room_type, r.meal
                    FROM fact_reservations f
                    LEFT JOIN dim_calendar c ON f.id_calendar = c.id_calendar
                    LEFT JOIN dim_client cl ON f.id_client = cl.id_client
                    LEFT JOIN dim_channel ch ON f.id_channel = ch.id_channel
                    LEFT JOIN dim_room r ON f.id_room = r.id_room
                    ORDER BY c.arrival_date, f.lead_time
                    ''', engine)
       
    ### Training the dataset
    # Variable to predict
    y = df['is_canceled']
    
    # Predictor variables (without target, date/status info and ids)
    X = df.drop(columns=['is_canceled', 'arrival_date', 'reservation_status', 'reservation_status_date', 'assigned_room_type',
                              'id_reservation', 'id_client', 'id_calendar', 'id_room', 'id_channel'], errors='ignore')
    
    # Text variables will be encoded
    X = pd.get_dummies(X, drop_first=True)
    
    # Division of training and test sets (80/20)
    # X = independent variables, y= dependent variable. Train = trainging data, test = test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # Avaluation of unbalanced dataset
    print(f"\n--Dataset analysis (past data)--")
    print(f"\n1. Analysis of balance of the dataset")
    percentage_canceled = y_train.mean()
    is_unbalanced = percentage_canceled < 0.4 or percentage_canceled > 0.6
    print(f"Percentage of canceled reservations in training set: {percentage_canceled:.2%}")
    if is_unbalanced:
        print("The dataset is unbalanced. Main metrics to evaluate will be AUC-PR.")
    else:
        print("The dataset is balanced. Best balance of global metrics will be looked for.")

    # Models definition
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced')),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=5, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', max_depth = 5, scale_pos_weight=1.5, random_state=42)
    }
 
    # Training and evaluation of models (feedback loop CRISP-DM)
    print("\n2. Training and evaluation of models:")
    #results = {}
    baseline_f1 = 0
    best_mark = -1
    best_model = ""
    best_model_obj = None

    # For each model, training, evaluation and feedback loops are done
    for name, model in models.items():
        print(f"· Training {name}")
        model.fit(X_train, y_train)
        
        # Predictions and probabilities for train (overfitting) and test (real)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_prob_train = model.predict_proba(X_train)[:, 1]
        y_prob_test = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        auc_roc_train = roc_auc_score(y_train, y_prob_train)
        auc_roc_test = roc_auc_score(y_test, y_prob_test)
        f1_train = f1_score(y_train, y_pred_train)
        f1_test = f1_score(y_test, y_pred_test)
        auc_pr_test = average_precision_score(y_test, y_prob_test)
        mcc_test = matthews_corrcoef(y_test, y_pred_test)
        print(f"Results for {name}: AUC-ROC: {auc_roc_test:.3f} | AUC-PR: {auc_pr_test:.3f} | F1: {f1_test:.3f} | MCC: {mcc_test:.3f}")
        
        # If regression logistics, seet baseline F1
        if name == "Logistic Regression":
            baseline_f1 = f1_test

        # Set feedback loop
        auc_roc_diff = (auc_roc_train - auc_roc_test) / auc_roc_train
        f1_diff = (f1_train - f1_test) / f1_train

        loop_1 = False
        loop_2 = False

        # Check loop 1 (data preparation)
        if mcc_test < 0.5 or (name != "Logistic Regression" and f1_test < baseline_f1):
            print(f"{name} has a low performance (low MCC or worse than baseline). More cleaning is needed.\n")
            loop_1 = True
        if auc_roc_diff > 0.05 or f1_diff > 0.05:
            print(f"{name} has overfitting (lowering test vs train > 5%). Regularization, more data or simpler model needed.\n")
            loop_2 = True
        
        # Check loop 2 (threhold)
        if auc_pr_test < 0.5:
            print(f"{name} has a low AUC-PR. Adjusting threshold, resampling techniques or different algorithms needed.\n")
            loop_2 = True

        if not loop_1 and not loop_2:
            print(f"{name} has a good performance. Quality criteria passed.\n")

        # Best model selection
        # If dataset is unbalanced, AUC-PR will be the main metric. Else, average of all metrics will be done
        if is_unbalanced:
            mark = auc_pr_test
        else:
            mark = (auc_roc_test + auc_pr_test + f1_test + mcc_test) / 4

        if mark > best_mark and not loop_1:
            best_mark = mark
            best_model = name
            best_model_obj = model

    # Export the best model
    if best_model_obj is not None:
        print(f"The best model is: {best_model}")
        print(f"Selection criteria: {'AUC-PR' if is_unbalanced else 'Average of AUC-ROC, AUC-PR, F1 and MCC'} with a score of {best_mark:.3f}")
        train_columns = list(X.columns)
        joblib.dump({'model': best_model_obj, 'columns': train_columns}, '../api/final_model.pkl')
        print("\nFile 'final_model.pkl' generated successfully.")
    else:
        print("No model passed the quality criteria. Execute the feedback loops to improve data preparation, model selection or threshold adjustment.")

        
    # Generation of graphics for the best model
    print("\nGenerating graphs for the best model.")
    plt.style.use('seaborn-v0_8-whitegrid')

    # Confusion matrix for best model
    y_pred_best = best_model_obj.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Not cancelled', 'Cancelled'], yticklabels=['Not cancelled', 'Cancelled'])
    plt.title(f'Confusion Matrix - {best_model}')
    plt.ylabel('Reality')
    plt.xlabel('Prediction')
    plt.savefig('../graphics/confusion_matrix.png')
    plt.close()

    # ROC and PR curves
    figure_roc, ax_roc = plt.subplots(figsize=(8, 6))
    figure_pr, ax_pr = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc, name=name)
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax_pr, name=name)
    ax_roc.set_title('Comparison of ROC curves')
    figure_roc.savefig('../graphics/roc_curves.png')
    ax_pr.set_title('Comparison of Precision-Recall curves')
    figure_pr.savefig('../graphics/pr_curves.png')

if __name__ == "__main__":
    run_training()