
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px

# Title and description
st.title("Sales Forecasting and Analysis")
st.write("Perform revenue forecasting, analyze sales, and visualize data.")

# Sidebar - Upload CSV file
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Main content
if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Display the uploaded data
    st.subheader("Uploaded Data")
    st.dataframe(df)
    missing_values = df.isnull().sum()
    st.write("Missing Values:")
    st.write(missing_values)
    # Forecasting revenue
    st.subheader("Revenue Forecasting")
        st.subheader("Model Evaluation")

    # Train-test split
    train_size = 0.8
    train_index = int(train_size * len(df))
    train_data = df[:train_index]
    test_data = df[train_index:]

    # Train the model
    X_train = pd.to_datetime(train_data['Month']).values.astype(np.int64)[:, np.newaxis]
    y_train = train_data['Revenue'].values

    X_test = pd.to_datetime(test_data['Month']).values.astype(np.int64)[:, np.newaxis]
    y_test = test_data['Revenue'].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    st.write("Train MAE:", train_mae)
    st.write("Test MAE:", test_mae)

    st.write("Train MSE:", train_mse)
    st.write("Test MSE:", test_mse)

    st.write("Train R^2 Score:", train_r2)
    st.write("Test R^2 Score:", test_r2)



    # Predict the revenue for future months
    future_dates = pd.date_range(start=df['Month'].iloc[-1], periods=12, freq='M')
    future_dates_int = future_dates.values.astype(np.int64)[:, np.newaxis]
    future_revenue = model.predict(future_dates_int)

    forecast_df = pd.DataFrame({'Month': future_dates, 'Forecasted Revenue': future_revenue})

    st.subheader("Revenue Forecast for Next 12 Months")
    st.dataframe(forecast_df)

    # Forecasting revenue by location
    st.subheader("Revenue Forecast by Location")

    # Group the data by location and perform revenue forecasting for each location
    location_forecast_df = pd.DataFrame()
    for location in df['Location'].unique():
        location_data = df[df['Location'] == location]
        X_loc = pd.to_datetime(location_data['Month']).values.astype(np.int64)[:, np.newaxis]
        y_loc = location_data['Revenue'].values

        model_loc = LinearRegression()
        model_loc.fit(X_loc, y_loc)

        future_revenue_loc = model_loc.predict(future_dates_int)

        forecast_loc = pd.DataFrame({'Month': future_dates, 'Location': location, 'Forecasted Revenue': future_revenue_loc})
        location_forecast_df = location_forecast_df.append(forecast_loc)

    st.dataframe(location_forecast_df)

    # Forecasting revenue by product
    st.subheader("Revenue Forecast by Product")

    # Group the data by product and perform revenue forecasting for each product
    product_forecast_df = pd.DataFrame()
    for product in df['Product'].unique():
        product_data = df[df['Product'] == product]
        X_prod = pd.to_datetime(product_data['Month']).values.astype(np.int64)[:, np.newaxis]
        y_prod = product_data['Revenue'].values

        model_prod = LinearRegression()
        model_prod.fit(X_prod, y_prod)

        future_revenue_prod = model_prod.predict(future_dates_int)

        forecast_prod = pd.DataFrame({'Month': future_dates, 'Product': product, 'Forecasted Revenue': future_revenue_prod})
        product_forecast_df = product_forecast_df.append(forecast_prod)

    st.dataframe(product_forecast_df)

    # Sales analysis
    st.subheader("Sales Analysis")

    # Group the data by product and calculate total revenue
    product_analysis_df = df.groupby('Product')['Revenue'].sum().reset_index()

    st.dataframe(product_analysis_df)

    # Bar chart for product sales analysis
    st.subheader("Product Sales Analysis")
    fig_product = plt.figure(figsize=(10, 6))
    sns.barplot(x='Product', y='Revenue', data=product_analysis_df)
    plt.xlabel('Product')
    plt.ylabel('Revenue')
    plt.title('Product Sales Analysis')
    st.pyplot(fig_product)

    # Compare two products' past sales
    st.subheader("Comparison of Two Products' Past Sales")

    product1 = st.selectbox("Select Product 1", df['Product'].unique())
    product2 = st.selectbox("Select Product 2", df['Product'].unique())

    product_comparison_df = df[df['Product'].isin([product1, product2])]
    fig_comparison = plt.figure(figsize=(10, 6))
    sns.lineplot(x='Month', y='Revenue', hue='Product', data=product_comparison_df)
    plt.xlabel('Month')
    plt.ylabel('Revenue')
    plt.title('Comparison of Two Products')
    st.pyplot(fig_comparison)
