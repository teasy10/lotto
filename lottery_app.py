
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.preprocessing import StandardScaler

# Load data and models
@st.cache_data
def load_data():
    df = pd.read_csv('lottery_data_with_factors.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_resource
def load_models():
    best_model = joblib.load('best_lottery_model.joblib')
    scaler = joblib.load('feature_scaler.joblib')
    return best_model, scaler

df = load_data()
best_model, scaler = load_models()

st.title('Enhanced Lottery Number Analysis App')

# Sidebar for date range selection
st.sidebar.header('Date Range Selection')
start_date = st.sidebar.date_input('Start Date', df['Date'].min())
end_date = st.sidebar.date_input('End Date', df['Date'].max())

# Filter data based on date range
filtered_df = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]

# 1. Lottery Number Statistics
st.header('1. Lottery Number Statistics')
st.write(filtered_df[['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6']].describe())

# 2. Number Frequency Analysis
st.header('2. Number Frequency Analysis')
all_numbers = filtered_df[['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6']].values.ravel()
number_freq = pd.Series(all_numbers).value_counts().sort_index()
fig = px.bar(x=number_freq.index, y=number_freq.values, labels={'x': 'Number', 'y': 'Frequency'})
fig.update_layout(title='Frequency of Each Number')
st.plotly_chart(fig)

# 3. Correlation Heatmap
st.header('3. Correlation Heatmap')
correlation_matrix = filtered_df[['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6']].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# 4. Time Series Analysis
st.header('4. Time Series Analysis')
filtered_df['Total'] = filtered_df[['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6']].sum(axis=1)
fig = px.line(filtered_df, x='Date', y='Total', title='Sum of Winning Numbers Over Time')
st.plotly_chart(fig)

# 5. External Factors Analysis
st.header('5. External Factors Analysis')
factors = ['DayOfWeek', 'MonthOfYear', 'IsHoliday', 'EconomicIndex']
for factor in factors:
    fig = px.box(filtered_df, x=factor, y='Total', title=f'Total vs {factor}')
    st.plotly_chart(fig)

# 6. Number Prediction
st.header('6. Number Prediction')
st.subheader('Predict if the total will be above or below median')
user_input = {}
for col in ['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6', 'DayOfWeek', 'MonthOfYear', 'IsHoliday', 'EconomicIndex']:
    user_input[col] = st.number_input(f'Enter {col}', value=filtered_df[col].median())

if st.button('Predict'):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    prediction = best_model.predict(input_scaled)
    st.write('Prediction:', 'Above Median' if prediction[0] else 'Below Median')

# 7. Custom Visualization
st.header('7. Custom Visualization')
filter_column = st.selectbox('Select column to filter', df.columns)
if filter_column:
    if df[filter_column].dtype in ['int64', 'float64']:
        min_value = st.number_input('Min value', value=float(df[filter_column].min()))
        max_value = st.number_input('Max value', value=float(df[filter_column].max()))
        filtered_df = filtered_df[(filtered_df[filter_column] >= min_value) & (filtered_df[filter_column] <= max_value)]
    else:
        specific_value = st.selectbox('Select specific value', df[filter_column].unique())
        filtered_df = filtered_df[filtered_df[filter_column] == specific_value]
    
    # Custom Visualization
    st.write('Custom Visualization')
    chart_type = st.selectbox('Select chart type', ['Scatter Plot', 'Bar Chart', 'Box Plot'])
    
    if chart_type == 'Scatter Plot':
        x_axis = st.selectbox('Select X-axis', df.columns)
        y_axis = st.selectbox('Select Y-axis', df.columns)
        fig = px.scatter(filtered_df, x=x_axis, y=y_axis, title=f'{y_axis} vs {x_axis}')
        st.plotly_chart(fig)
    elif chart_type == 'Bar Chart':
        x_axis = st.selectbox('Select X-axis', df.columns)
        y_axis = st.selectbox('Select Y-axis', df.columns)
        fig = px.bar(filtered_df, x=x_axis, y=y_axis, title=f'{y_axis} by {x_axis}')
        st.plotly_chart(fig)
    elif chart_type == 'Box Plot':
        x_axis = st.selectbox('Select X-axis (categorical)', df.select_dtypes(include=['object', 'category']).columns)
        y_axis = st.selectbox('Select Y-axis (numerical)', df.select_dtypes(include=['int64', 'float64']).columns)
        fig = px.box(filtered_df, x=x_axis, y=y_axis, title=f'Distribution of {y_axis} by {x_axis}')
        st.plotly_chart(fig)

# Export Data
if st.button('Export Filtered Data'):
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="filtered_lottery_data.csv",
        mime="text/csv"
    )

st.header('8. Custom Analysis')
st.write("Select columns for custom analysis:")
selected_columns = st.multiselect('Select columns', df.columns)
if selected_columns:
    st.write(filtered_df[selected_columns].describe())
    if len(selected_columns) == 2:
        fig = px.scatter(filtered_df, x=selected_columns[0], y=selected_columns[1])
        st.plotly_chart(fig)
    elif len(selected_columns) > 2:
        correlation = filtered_df[selected_columns].corr()
        fig = px.imshow(correlation, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig)

# Real-time data update
import threading
import schedule
import time

def update_dataset():
    global df
    new_data = generate_new_data()
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv('lottery_data_with_factors.csv', index=False)
    st.write(f"Dataset updated with new entry for {new_data['Date'].values[0]}")

def run_scheduled_tasks():
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

# Start the scheduled tasks in a separate thread
update_thread = threading.Thread(target=run_scheduled_tasks)
update_thread.start()

# Add a button to manually trigger an update
if st.button('Update Dataset'):
    update_dataset()
    st.write("Dataset updated manually.")

