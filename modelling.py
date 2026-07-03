import os
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

def train_model(n_estimators, max_depth, min_samples_split):
    print(f"Menjalankan retraining dengan parameter: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}")
    
    # 1. Cek apakah berjalan di dalam konteks "mlflow run"
    env_run_id = os.environ.get("MLFLOW_RUN_ID")
    is_in_mlflow_run = env_run_id is not None

    # Selalu set tracking URI ke path relatif terlebih dahulu agar artifact
    # disimpan di direktori kerja runner (bukan path absolut mesin lokal).
    # Ini penting ketika berjalan di GitHub Actions via "mlflow run .".
    mlflow.set_tracking_uri("file:./mlruns")

    if not is_in_mlflow_run:
        # Konfigurasi DagsHub MLflow Tracking jika kredensial tersedia
        dagshub_username = os.environ.get("DAGSHUB_USERNAME")
        dagshub_token = os.environ.get("DAGSHUB_TOKEN")
        
        if dagshub_username and dagshub_token:
            print("DagsHub credentials found. Logging to DagsHub online...")
            tracking_uri = f"https://dagshub.com/{dagshub_username}/mlops-wine-quality.mlflow"
            mlflow.set_tracking_uri(tracking_uri)
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
            mlflow.set_experiment("Wine-Quality-Retraining-CI")
        else:
            print("No DagsHub credentials found. Logging to local mlruns...")
            mlflow.set_experiment("Wine-Quality-Retraining-Local")
        
    # 2. Memuat dataset preprocessing
    train_path = 'winequality_preprocessing/train_processed.csv'
    test_path = 'winequality_preprocessing/test_processed.csv'
    
    if not os.path.exists(train_path):
        raise FileNotFoundError("Data preprocessing tidak ditemukan di folder winequality_preprocessing.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop('good_quality', axis=1)
    y_train = train_df['good_quality']
    X_test = test_df.drop('good_quality', axis=1)
    y_test = test_df['good_quality']
    
    # 3. Training & Logging
    if is_in_mlflow_run:
        print("Running inside an active MLflow project run. Logging parameters & metrics directly.")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else None,
            min_samples_split=min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Hasil Evaluasi - Akurasi: {acc:.4f}, F1-Score: {f1:.4f}")
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")
        
        run_id = env_run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        print(f"Retraining selesai. Run ID logged: {run_id}")
    else:
        with mlflow.start_run(run_name="RandomForest-CI-Retrained") as run:
            print(f"Run ID: {run.info.run_id}")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("min_samples_split", min_samples_split)
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth if max_depth > 0 else None,
                min_samples_split=min_samples_split,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            print(f"Hasil Evaluasi - Akurasi: {acc:.4f}, F1-Score: {f1:.4f}")
            
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(model, "model")
            
            with open("run_id.txt", "w") as f:
                f.write(run.info.run_id)
            print("Model retraining berhasil di-log ke MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=5)
    args = parser.parse_args()
    
    train_model(args.n_estimators, args.max_depth, args.min_samples_split)
