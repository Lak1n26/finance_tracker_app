import requests
import pandas as pd
import numpy as np
import streamlit as st
import datetime, time
import os


API_KEY = os.environ['API_KEY']
SHEET_ID = os.environ['SHEET_ID']


st.title('Finance Tracker App')

sheet_name = 'Факт'
range = 'A1:E1000'

# подтягиваем фактические данные из Google Sheets
url_data = requests.get(f'https://sheets.googleapis.com/v4/spreadsheets/{SHEET_ID}/values/{sheet_name}!{range}?key={API_KEY}')
df = pd.DataFrame(url_data.json()['values'][1:], columns=url_data.json()['values'][0])
df['Дата'] = pd.to_datetime(df['Дата'], format='%d.%m.%Y')
df = df[df['Дата'].notna()]
for col in ['На карте + нал', 'На вкладах', 'Инвестиции', 'Всего']:
    df[col] = df[col].str.split().str.join('').astype(int)

year = st.selectbox(
    'Выберите год',
    (2023, 2024),  index=None, placeholder='Выберите значение')

if year is not None:
    uniq_months = df[df['Дата'].dt.year == year]['Дата'].dt.month.unique()
else:
    uniq_months = df['Дата'].dt.month.unique()

months = st.multiselect(
    'Выберите месяц(ы)',
    tuple(uniq_months), placeholder='Выберите значения')


if year is not None and months != []:
    part = df[(df['Дата'].dt.year == year) & (df['Дата'].dt.month.isin(months))]
elif year is not None:
    part = df[(df['Дата'].dt.year == year)]
elif months != []:
    part = df[(df['Дата'].dt.month.isin(months))]
else:
    part = df
st.dataframe(part.reset_index(drop=True))
st.line_chart(data=part, x='Дата', y='Всего')

last_change = str(max(df['Дата']))[:10]
st.text(f'Дата последнего внесения данных: {last_change}')



st.title('Линейная регрессия')
from sklearn.linear_model import LinearRegression
from datetime import timedelta
df = df[df['Дата'] >= '2023-10-01']
periods = st.number_input("Введите прогнозируемое число дней", value=30)

date_range = pd.DataFrame(pd.date_range(min(df['Дата']), max(df['Дата'])), columns=['Дата'])
train_df = date_range.merge(df[['Дата', 'Всего']], how='left')
train_df = train_df.fillna(method='ffill')
train_df.index = train_df.index + 1


X = np.reshape(train_df.index, (-1, 1))
y = train_df['Всего']
reg = LinearRegression(fit_intercept=True).fit(X, y)

st.write(f'Прогнозируемые сбережения за 1 день: {int(reg.coef_[0])} рублей')
economy_per_day = st.slider('Величина сбережений за 1 день', int(reg.coef_[0]) // 2, int(reg.coef_[0]) * 2, int(reg.coef_[0]))
st.write(f'Прогнозируемые сбережения за месяц: {int(economy_per_day * 30)} руб.')
train_df['Прогноз'] = reg.intercept_ + train_df.index * economy_per_day

# прогнозируем на N дней вперед
test_df = pd.DataFrame(pd.date_range(max(train_df['Дата']) + timedelta(days=1), periods=periods), columns=['Дата'])
test_df.index = np.arange(max(train_df.index) + 1, max(train_df.index) + periods + 1)
test_df['Прогноз'] = reg.intercept_ + test_df.index * economy_per_day

# объединяем фактические данные и прогнозируемые
total_df = pd.concat([train_df, test_df], axis=0)
st.line_chart(data=total_df, x='Дата', y=['Всего', 'Прогноз'], color=[(0, 0, 255), (255, 0, 0)])


# подтягиваем План из Google Sheets
sheet_name = 'План'
range = 'A1:C20'
url_data = requests.get(f'https://sheets.googleapis.com/v4/spreadsheets/{SHEET_ID}/values/{sheet_name}!{range}?key={API_KEY}')
plan_completed = pd.DataFrame(url_data.json()['values'][1:], columns=url_data.json()['values'][0])
plan_completed['Дата'] = pd.to_datetime(plan_completed['Дата'])
plan_completed['План, тыс. руб.'] = plan_completed['План, тыс. руб.'].astype(np.int64)
# прогноз сбережений на год вперед
test_df = pd.DataFrame(pd.date_range(max(train_df['Дата']) + timedelta(days=1), periods=365), columns=['Дата'])
test_df.index = np.arange(max(train_df.index) + 1, max(train_df.index) + 365 + 1)
test_df['Прогноз'] = reg.intercept_ + test_df.index * economy_per_day
total_df = pd.concat([train_df, test_df], axis=0)

# # оставляем прогнозируемые значения на отчетные даты
plan_completed = plan_completed.merge(total_df)
plan_completed['Факт, тыс. руб.'] = plan_completed['Всего'].fillna(plan_completed['Прогноз'])
plan_completed['Факт, тыс. руб.'] = (plan_completed['Факт, тыс. руб.'] // 1000).astype(np.int64)
plan_completed = plan_completed[['Период', 'План, тыс. руб.', 'Факт, тыс. руб.']]
plan_completed['Выполнен план'] = (plan_completed['План, тыс. руб.'] <= plan_completed['Факт, тыс. руб.'])
plan_completed['Выполнен план'] = plan_completed['Выполнен план'].apply(lambda x: 'Да' if x == True else 'Нет')


with st.expander("ПЛАН"):
    st.table(plan_completed)



