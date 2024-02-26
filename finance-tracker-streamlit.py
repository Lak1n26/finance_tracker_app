import requests
import pandas as pd
import numpy as np
import streamlit as st
import datetime, time
import os


# with open('config.txt') as file:
#     API_KEY = file.readline()
#     SHEET_ID = file.readline()

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

st.write()
st.write()
sheet_name = 'План'
range = 'A1:C20'
url_data = requests.get(f'https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}/values/{sheet_name}!{range}?key={api_key}')
plan = pd.DataFrame(url_data.json()['values'][1:], columns=url_data.json()['values'][0])
plan = plan.fillna('')
with st.expander("ПЛАН"):
    st.table(plan)



# st.write('Построим линейную регрессию')
# import statsmodels.api as sm
# X = df.index
# y = df['Всего']
# X = sm.add_constant(X)

# model = sm.OLS(y, X)
# results = model.fit()
# st.write(results.summary())


# st.date_input(
#     "Select a date range",
#     (pd.to_datetime('2023-10-01'), pd.to_datetime('2024-02-25'))
# )



# sidebar
# with st.sidebar:
#     with st.spinner("Loading..."):
#         time.sleep(5)
#     st.success("Done!")


# 2 вкладки
# tab1, tab2 = st.tabs(["Факт", "Плн"])
# tab1.write("this is tab 1")
# tab2.write("this is tab 2")
# with tab1:
#     st.table(df)
# with tab2:
#     st.table(plan)


# https://arnaudmiribel.github.io/streamlit-extras/extras/metric_cards/


