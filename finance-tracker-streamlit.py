import requests
import pandas as pd
import numpy as np
import streamlit as st
import datetime, time
import os


API_KEY = os.environ['API_KEY']
SHEET_ID = os.environ['SHEET_ID']

sheet_name = 'Факт'
range = 'A1:E1000'

url_data = requests.get(f'https://sheets.googleapis.com/v4/spreadsheets/{SHEET_ID}/values/{sheet_name}!{range}?key={API_KEY}')
df = pd.DataFrame(url_data.json()['values'][1:], columns=url_data.json()['values'][0])
df['Дата'] = pd.to_datetime(df['Дата'], format='%d.%m.%Y')
df = df[df['Дата'].notna()]
for col in ['На карте + нал', 'На вкладах', 'Инвестиции', 'Всего']:
    df[col] = df[col].str.split().str.join('').astype(int)

st.title('Finance Tracker App')

year = st.selectbox(
    'Select a year',
    (2023, 2024),  index=None)

if year is not None:
    uniq_months = df[df['Дата'].dt.year == year]['Дата'].dt.month.unique()
else:
    uniq_months = df['Дата'].dt.month.unique()

months = st.multiselect(
    'Select the month(s)',
    tuple(uniq_months))


if year is not None and months != []:
    part = df[(df['Дата'].dt.year == year) & (df['Дата'].dt.month.isin(months))]
elif year is not None:
    part = df[(df['Дата'].dt.year == year)]
elif months != []:
    part = df[(df['Дата'].dt.month.isin(months))]
else:
    part = df
st.table(part.reset_index(drop=True))
st.line_chart(data=part, x='Дата', y='Всего')

last_change = str(max(df['Дата']))[:10]
st.text(f'Дата последнего внесения данных: {last_change}')


sheet_name = 'План'
range = 'A1:C20'
url_data = requests.get(f'https://sheets.googleapis.com/v4/spreadsheets/{SHEET_ID}/values/{sheet_name}!{range}?key={API_KEY}')
plan = pd.DataFrame(url_data.json()['values'][1:], columns=url_data.json()['values'][0])
plan = plan.fillna('')
with st.expander("ПЛАН"):
    st.table(plan)


st.title('Линейная регрессия')
from sklearn.linear_model import LinearRegression
from datetime import timedelta

periods = st.number_input("Введите прогнозируемое число дней", value=30)

date_range = pd.DataFrame(pd.date_range(min(df['Дата']), max(df['Дата'])), columns=['Дата'])
train_df = date_range.merge(df[['Дата', 'Всего']], how='left')
train_df = train_df.fillna(method='ffill')
train_df.index = train_df.index + 1


X = np.reshape(train_df.index, (-1, 1))
y = train_df['Всего']
reg = LinearRegression(fit_intercept=True).fit(X, y)
train_df['Прогноз'] = reg.intercept_ + train_df.index * reg.coef_

test_df = pd.DataFrame(pd.date_range(max(train_df['Дата']) + timedelta(days=1), periods=periods), columns=['Дата'])
test_df.index = np.arange(max(train_df.index) + 1, max(train_df.index) + periods + 1)
test_df['Прогноз'] = reg.intercept_ + test_df.index * reg.coef_

total_df = pd.concat([train_df, test_df], axis=0)
st.line_chart(data=total_df, x='Дата', y=['Всего', 'Прогноз'], color=[(0, 0, 255), (255, 0, 0)])


