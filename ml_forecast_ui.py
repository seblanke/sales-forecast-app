
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.title("📊 ML Forecast & Sales Target Tool")

# Step 1: Daten-Upload
st.header("1. Daten-Upload")
uploaded_file = st.file_uploader("Lade deine CSV-Datei hoch", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Vorschau der Daten:", df.head())

    # Step 2: Feature-Auswahl
    st.header("2. Feature-Auswahl")
    target_col = st.selectbox("Zielvariable auswählen", df.columns)
    feature_cols = st.multiselect("Eingabefeatures auswählen", [col for col in df.columns if col != target_col])

    if len(feature_cols) > 0:
        # Step 3: Modelltraining
        st.header("3. Modelltraining")
        test_size = st.slider("Testdaten-Anteil", 0.1, 0.5, 0.2)

        # Robust: Nur numerische Features zulassen
        df_filtered = df[feature_cols].select_dtypes(include=[np.number])
        if df_filtered.shape[1] == 0:
            st.error("Keine numerischen Features gefunden. Bitte wähle passende Spalten aus.")
        else:
            X = df_filtered.dropna()
            y = df.loc[X.index, target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Fix hier
            st.success(f"RMSE des Modells: {rmse:.2f}")

            # Feature Importance
            st.subheader("Feature-Wichtigkeit")
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Wichtigkeit': model.feature_importances_
            }).sort_values(by="Wichtigkeit", ascending=False)
            st.write(importance_df)

            # Step 4: Vorhersage
            st.header("4. Forecast und Zielsetzung")
            forecast_input = X
            df_result = df.loc[forecast_input.index].copy()
            df_result["Forecast"] = model.predict(forecast_input)
            stretch = st.slider("Stretch-Faktor in %", 0, 50, 10)
            df_result["Zielwert"] = df_result["Forecast"] * (1 + stretch / 100)

            st.write(df_result[["Forecast", "Zielwert"] + [target_col]].head())

            csv_export = df_result.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Ergebnisse als CSV herunterladen", csv_export, "forecast_output.csv", "text/csv")
