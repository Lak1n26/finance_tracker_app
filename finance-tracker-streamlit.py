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

st.write(f'Прогнозируемые сбережения за 1 день: {int(reg.coef_[0])} рублей')

economy_per_day = st.slider('Величина сбережений за 1 день', int(reg.coef_[0]) // 2, int(reg.coef_[0]) * 2, int(reg.coef_[0]))

train_df['Прогноз'] = reg.intercept_ + train_df.index * economy_per_day

# прогнозируем на N дней вперед
test_df = pd.DataFrame(pd.date_range(max(train_df['Дата']) + timedelta(days=1), periods=periods), columns=['Дата'])
test_df.index = np.arange(max(train_df.index) + 1, max(train_df.index) + periods + 1)
test_df['Прогноз'] = reg.intercept_ + test_df.index * economy_per_day

# объединяем фактические данные и прогнозируемые
total_df = pd.concat([train_df, test_df], axis=0)
st.line_chart(data=total_df, x='Дата', y=['Всего', 'Прогноз'], color=[(0, 0, 255), (255, 0, 0)])


# создадим ДФ с отчетными датами
plan_completed = pd.DataFrame({'Дата': ['2023-12-31', '2024-01-31', '2024-02-29', '2024-03-31', '2024-04-30', '2024-05-31', '2024-06-30', '2024-07-31', '2024-08-31'],
                               'Период': ['На конец декабря \'23', 'На конец января \'24', 'На конец февраля \'24', 'На конец марта \'24', 'На конец апреля \'24', 
                                'На конец мая \'24', 'На конец июня \'24', 'На конец июля \'24', 'На конец августа \'24'],
                                'План, тыс.руб.': [150, 220, 410, 470, 540, 620, 700, 800, 850]})
plan_completed['Дата'] = pd.to_datetime(plan_completed['Дата'])

# прогноз сбережений на год вперед
test_df = pd.DataFrame(pd.date_range(max(train_df['Дата']) + timedelta(days=1), periods=365), columns=['Дата'])
test_df.index = np.arange(max(train_df.index) + 1, max(train_df.index) + 365 + 1)
test_df['Прогноз'] = reg.intercept_ + test_df.index * economy_per_day
total_df = pd.concat([train_df, test_df], axis=0)

# оставляем прогнозируемые значения на отчетные даты
plan_completed = plan_completed.merge(total_df)
plan_completed['Факт, тыс.руб.'] = plan_completed['Всего'].fillna(plan_completed['Прогноз'])
plan_completed['Факт, тыс.руб.'] = (plan_completed['Факт, тыс.руб.'] // 1000).astype(np.int64)
plan_completed = plan_completed[['Период', 'План, тыс.руб.', 'Факт, тыс.руб.']]
plan_completed['Выполнен план'] = (plan_completed['План, тыс.руб.'] <= plan_completed['Факт, тыс.руб.'])
plan_completed['Выполнен план'] = plan_completed['Выполнен план'].apply(lambda x: 'Да' if x == True else 'Нет')


with st.expander("ПЛАН"):
    st.table(plan_completed)



