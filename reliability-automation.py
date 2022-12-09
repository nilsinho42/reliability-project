# import pymysql
import pandas as pd
import numpy as np
from datetime import timedelta
import sqlalchemy as db

# DASH and PLOTLY library
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Reliability library
from reliability.Repairable_systems import MCF_nonparametric
from reliability.Repairable_systems import MCF_parametric

# All temperatures in Celsius Degrees
UPPER_RANGE_T_MOT = 100 
LOWER_RANGE_T_MOT = 20

UPPER_RANGE_T_OIL = 100
LOWER_RANGE_T_OIL = 30

LOWER_RANGE_FREQ = 27

AIR_COMPRESSOR_ON_STATUS = 0
AIR_COMPRESSOR_OFF_STATUS = 1

UPPER_LIMIT_T_MOT = 50
UPPER_LIMIT_T_OIL = 80
UPPER_LIMIT_T_AIR = 0
UPPER_LIMIT_POWER = 8625
UPPER_LIMIT_VIBR = 3.2
UPPER_LIMIT_PRES = 10


class MySQLConnection:
    def __init__(self):
        self.db_user = "root"
        self.db_password = ""
        self.host = "127.0.0.1"
        self.db_name = "SENAI_4_0"

    def save_data(self, data, table_name):
        engine = db.create_engine(
            "mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=self.host,
                                                             db=self.db_name,
                                                             user=self.db_user,
                                                             pw=self.db_password))
        conn = engine.connect()
        data.to_sql(table_name, engine, if_exists="replace", index=False)
    def get_data(self, table_name):
        engine = db.create_engine(
            "mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=self.host,
                                                             db=self.db_name,
                                                             user=self.db_user,
                                                             pw=self.db_password))
        conn = engine.connect()
        metadata = db.MetaData()
        data = db.Table('reg_comp_senai_resp', metadata, autoload=True, autoload_with=engine)
        resultproxy = conn.execute("SELECT * FROM {table}".format(table=table_name))
        resultset = resultproxy.fetchall()

        return resultset



# Variable Selection and Renaming Columns for better undestanding 
def variable_selection(data):
    data['datetime'] = data['t_data'].apply(lambda x: pd.Timestamp(x))
    data['datetime'] = data['datetime'] + data['t_hora']
    data = data.sort_values(by='datetime')
    data.set_index(pd.DatetimeIndex(data['datetime']))

    variaveis_analisadas = ['datetime', 'p_comp', 't_oleo', 't_ar', 't_mot', 'ai1_hz', 'ai2_w', 'ai3_m_s_2', 'di_00'] 
    data = data[variaveis_analisadas]
    novos_nomes_colunas = ['datetime', 'p_comp', 't_oleo', 't_ar', 't_mot', 'freq', 'potencia', 'vibracao', 'di_00'] 
    data.columns = novos_nomes_colunas
    data.reset_index(inplace=True, drop=True)
    return data

# Resampling data in periods of 30 minutes
def resample_data(data):
    data = data.resample('30min', on='datetime').agg(
        p_comp=('p_comp', 'mean'),
        t_oleo=('t_oleo', 'mean'),
        t_mot=('t_mot', 'mean'),
        t_ar=('t_ar', 'mean'),
        freq=('freq', 'mean'),
        potencia=('potencia', 'mean'),
        vibracao=('vibracao', 'mean'),
        di_00=('di_00', 'mean')
    )
    data = data.reset_index(drop=False)
    data['di_00'] = data['di_00'].apply(lambda x:1 if x >0.9 else 0)
    return data

# Calculating sum of operation hours for the entire period
def get_oper_hours_sum(data):
    oper_hours_over_period = {}
    data['last_time_on'] = 0
    for row in data.itertuples():
        i = row.Index
        if i+1 < len(data):
            oper_hours_over_period[i+1] = {'tempo':data['datetime'].iat[i+1]-data['datetime'].iat[i], 'di_00': data['di_00'].iat[i]}
    
    
    oper_hours_over_period_df = pd.DataFrame.from_dict(oper_hours_over_period, orient='index')
    oper_hours_over_period_df = oper_hours_over_period_df.groupby('di_00').sum()
    # oper_hours_over_period_df.head() # ON = 0 / OFF = 1
    return round(oper_hours_over_period_df.reset_index()['tempo'].at[0].total_seconds()/3600,2)

# Data cleaning
def clean_data(data):
    data = data[data['t_mot']<UPPER_RANGE_T_MOT]
    data = data[data['t_mot']>LOWER_RANGE_T_MOT]
    data = data[data['t_oleo']<UPPER_RANGE_T_OIL]
    data = data[data['t_oleo']>LOWER_RANGE_T_OIL]
    data = data[data['freq']>LOWER_RANGE_FREQ]
    
    data_di00 = data
    data = data[data['di_00']!=AIR_COMPRESSOR_OFF_STATUS]

    return data, data_di00

def add_limits_to_dataframe(data):
    data['limite_t_mot'] = UPPER_LIMIT_T_MOT
    data['limite_t_ar'] = UPPER_LIMIT_T_AIR
    data['limite_t_oleo'] = UPPER_LIMIT_T_OIL
    data['limite_pot'] = UPPER_LIMIT_POWER
    data['limite_vib'] = UPPER_LIMIT_VIBR
    data['limite_p_comp'] = UPPER_LIMIT_PRES
    data['t_ar_acima_lim'] = data['limite_t_ar'] < data['t_ar']
    data['t_oleo_acima_lim'] = data['limite_t_oleo'] < data['t_oleo']
    data['t_mot_acima_lim'] = data['limite_t_mot'] < data['t_mot']
    data['acima_do_limite'] = data['t_ar_acima_lim'] | data['t_oleo_acima_lim'] | data['t_mot_acima_lim']
    data['date'] = data['datetime'].dt.strftime('%Y-%m-%d')
    data['carga'] = data['freq'].apply(lambda x: '30_Hz' if x < 35 else '60_Hz')
    data['tempo_30_hz'] = data['carga'].apply(lambda x:0.5 if x=='30_Hz' else 0 )
    data['tempo_60_hz'] = data['carga'].apply(lambda x:0.5 if x=='60_Hz' else 0 )
    return data

def create_failures_dataframe(data, data_di00):
    #data_di00['t_ar_acima_lim'] = data_di00['limite_t_ar'] < data_di00['t_ar']
    #data_di00['t_oleo_acima_lim'] = data_di00['limite_t_oleo'] < data_di00['t_oleo']
    #data_di00['t_mot_acima_lim'] = data_di00['limite_t_mot'] < data_di00['t_mot']

    dados_com_limites = data
    dados_com_limites_di00 = data_di00

    t_ar_failure_state = False
    t_oleo_failure_state = False
    t_mot_failure_state = False

    valor_max = 0
    failures_dict_ar = {}
    failures_dict_oleo = {}
    failures_dict_mot = {}

    count = 0
    for i, row in dados_com_limites.iterrows():
        if row['t_ar_acima_lim'] == True:
            if t_ar_failure_state == False: 
                count = count + 1
                t_ar_failure_state = True
                failures_dict_ar[count] = {'t_start': row['datetime'], 'var_failure':'t_ar'}
            if row['t_ar'] > valor_max: valor_max = row['t_ar'] 

        if row['t_ar_acima_lim'] == False and t_ar_failure_state == True:
            failures_dict_ar[count].update({'valor_max': valor_max, 't_stop':row['datetime']})
            t_ar_failure_state = False

    for i, row in dados_com_limites.iterrows():
        if row['t_oleo_acima_lim'] == True:
            if t_oleo_failure_state == False: 
                count = count + 1
                t_oleo_failure_state = True
                failures_dict_oleo[count] = {'t_start': row['datetime'], 'var_failure':'t_oleo'}
            if row['t_oleo'] > valor_max: valor_max = row['t_oleo'] 

        if row['t_oleo_acima_lim'] == False and t_oleo_failure_state == True:
            failures_dict_oleo[count].update({'valor_max': valor_max, 't_stop':row['datetime']})
            t_oleo_failure_state = False

    for i, row in dados_com_limites.iterrows():
        if row['t_mot_acima_lim'] == True:
            if t_mot_failure_state == False: 
                count = count + 1
                t_mot_failure_state = True
                failures_dict_mot[count] = {'t_start': row['datetime'], 'var_failure':'t_mot'}
            if row['t_mot'] > valor_max: valor_max = row['t_mot'] 

        if row['t_mot_acima_lim'] == False and t_mot_failure_state == True:
            failures_dict_mot[count].update({'valor_max': valor_max, 't_stop':row['datetime']})
            t_mot_failure_state = False

    failures_df_ar = pd.DataFrame.from_dict(failures_dict_ar, orient='index')
    failures_df_ar['tempo_em_falha'] = failures_df_ar['t_stop'] - failures_df_ar['t_start']
    failures_df_ar = failures_df_ar[['t_start', 't_stop', 'tempo_em_falha', 'var_failure', 'valor_max']].sort_values('t_start').reset_index(drop=True)
    tempo_ate_primeira_falha_ar = failures_df_ar['t_start'].iloc[0] - dados_com_limites['datetime'].iloc[0]

    failures_df_mot = pd.DataFrame.from_dict(failures_dict_mot, orient='index')
    failures_df_mot['tempo_em_falha'] = failures_df_mot['t_stop'] - failures_df_mot['t_start']
    failures_df_mot = failures_df_mot[['t_start', 't_stop', 'tempo_em_falha', 'var_failure', 'valor_max']].sort_values('t_start').reset_index(drop=True)
    tempo_ate_primeira_falha_mot = failures_df_mot['t_start'].iloc[0] - dados_com_limites['datetime'].iloc[0]

    failures_df_oleo = pd.DataFrame.from_dict(failures_dict_oleo, orient='index')
    failures_df_oleo['tempo_em_falha'] = failures_df_oleo['t_stop'] - failures_df_oleo['t_start']
    failures_df_oleo = failures_df_oleo[['t_start', 't_stop', 'tempo_em_falha', 'var_failure', 'valor_max']].sort_values('t_start').reset_index(drop=True)
    tempo_ate_primeira_falha_oleo = failures_df_oleo['t_start'].iloc[0] - dados_com_limites['datetime'].iloc[0]
    
    failures_df_oleo['tempo_sem_operacao'] = timedelta(seconds = 0)
    for row in failures_df_oleo.itertuples():
        i = row.Index

        if i + 1 < len(failures_df_oleo):
            temp_df = dados_com_limites_di00[(dados_com_limites_di00['datetime'] > failures_df_oleo['t_stop'].iat[i]) & (dados_com_limites_di00['datetime'] < failures_df_oleo['t_start'].iat[i+1])]
            temp_df = temp_df[temp_df['di_00']==1].reset_index(drop=True)

            if len(temp_df) > 0:
                for j, jrow in temp_df.iterrows():
                    if j+1 < len(temp_df):
                        failures_df_oleo['tempo_sem_operacao'].iat[i] = temp_df['datetime'].iat[j+1] - temp_df['datetime'].iat[j] + failures_df_oleo['tempo_sem_operacao'].iat[i] 

    failures_df_oleo['tempo_em_falha_real'] = failures_df_oleo['tempo_em_falha'] - failures_df_oleo['tempo_sem_operacao']
    tempo_sem_operacao_oleo = failures_df_oleo.groupby('var_failure')['tempo_sem_operacao'].sum()
    print("Tempo SEM Operação Óleo", round(tempo_sem_operacao_oleo[0].total_seconds()/3600,2), "horas")
    
    failures_df_ar['tempo_sem_operacao'] = timedelta(seconds = 0)
    for row in failures_df_ar.itertuples():
        i = row.Index

        if i + 1 < len(failures_df_ar):
            temp_df = dados_com_limites_di00[(dados_com_limites_di00['datetime'] > failures_df_ar['t_stop'].iat[i]) & (dados_com_limites_di00['datetime'] < failures_df_ar['t_start'].iat[i+1])]
            temp_df = temp_df[temp_df['di_00']==1].reset_index(drop=True)

            if len(temp_df) > 0:
                for j, jrow in temp_df.iterrows():
                    if j+1 < len(temp_df):
                        failures_df_ar['tempo_sem_operacao'].iat[i] = temp_df['datetime'].iat[j+1] - temp_df['datetime'].iat[j] + failures_df_ar['tempo_sem_operacao'].iat[i] 
    
    failures_df_ar['tempo_em_falha_real'] = failures_df_ar['tempo_em_falha'] - failures_df_ar['tempo_sem_operacao']
    tempo_sem_operacao_ar = failures_df_ar.groupby('var_failure')['tempo_sem_operacao'].sum()
    print("Tempo SEM Operação Ar", round(tempo_sem_operacao_ar[0].total_seconds()/3600,2), "horas")
    
    failures_df_mot['tempo_sem_operacao'] = timedelta(seconds = 0)
    for row in failures_df_mot.itertuples():
        i = row.Index

        if i + 1 < len(failures_df_mot):
            temp_df = dados_com_limites_di00[(dados_com_limites_di00['datetime'] > failures_df_mot['t_stop'].iat[i]) & (dados_com_limites_di00['datetime'] < failures_df_mot['t_start'].iat[i+1])]
            temp_df = temp_df[temp_df['di_00']==1].reset_index(drop=True)

            if len(temp_df) > 0:
                for j, jrow in temp_df.iterrows():
                    if j+1 < len(temp_df):
                        failures_df_mot['tempo_sem_operacao'].iat[i] = temp_df['datetime'].iat[j+1] - temp_df['datetime'].iat[j] + failures_df_mot['tempo_sem_operacao'].iat[i] 
    
    failures_df_mot['tempo_em_falha_real'] = failures_df_mot['tempo_em_falha'] - failures_df_mot['tempo_sem_operacao']
    tempo_sem_operacao_ar = failures_df_mot.groupby('var_failure')['tempo_sem_operacao'].sum()
    print("Tempo SEM Operação Motor", round(tempo_sem_operacao_ar[0].total_seconds()/3600,2), "horas")
    
    # Cálculo de Tempo Médio entre falhas
    failures_df_ar['tempo_entre_falhas'] = 0
    for row in failures_df_ar.itertuples():
        i = row.Index
        if i+1 < len(failures_df_ar):
            failures_df_ar['tempo_entre_falhas'].iat[i] = failures_df_ar['t_start'].iat[i+1] - failures_df_ar['t_stop'].iat[i]


    failures_df_oleo['tempo_entre_falhas'] = 0
    for row in failures_df_oleo.itertuples():
        i = row.Index    
        if i+1 < len(failures_df_oleo):
            failures_df_oleo['tempo_entre_falhas'].iat[i] = failures_df_oleo['t_start'].iat[i+1] - failures_df_oleo['t_stop'].iat[i]

    failures_df_mot['tempo_entre_falhas'] = 0
    for row in failures_df_mot.itertuples():
        i = row.Index
        if i+1 < len(failures_df_mot):
            failures_df_mot['tempo_entre_falhas'].iat[i] = failures_df_mot['t_start'].iat[i+1] - failures_df_mot['t_stop'].iat[i]
            
    # Cálculo do tempo de falha acumulado por Variável
    failures_df_oleo['tempo_falha_acumulado'] = timedelta(seconds=0)
    for row in failures_df_oleo.itertuples():
        i = row.Index
        if i+1 < len(failures_df_oleo):
            failures_df_oleo['tempo_falha_acumulado'].iat[i+1] = failures_df_oleo['tempo_em_falha'].iat[i+1] + failures_df_oleo['tempo_falha_acumulado'].iat[i]

    failures_df_ar['tempo_falha_acumulado'] = timedelta(seconds=0)
    for row in failures_df_ar.itertuples():
        i = row.Index
        if i+1 < len(failures_df_ar):
            failures_df_ar['tempo_falha_acumulado'].iat[i+1] = failures_df_ar['tempo_em_falha'].iat[i+1] + failures_df_ar['tempo_falha_acumulado'].iat[i]

    failures_df_mot['tempo_falha_acumulado'] = timedelta(seconds=0)
    for row in failures_df_mot.itertuples():
        i = row.Index
        if i+1 < len(failures_df_mot):
            failures_df_mot['tempo_falha_acumulado'].iat[i+1] = failures_df_mot['tempo_em_falha'].iat[i+1] + failures_df_mot['tempo_falha_acumulado'].iat[i]

    ##### AR #####
    failures_df_ar.drop(index=failures_df_ar.index[-1],axis=0,inplace=True)
    failures_df_ar['tempo_entre_falhas_acumulado'] = timedelta(seconds=0) + tempo_ate_primeira_falha_ar
    for row in failures_df_ar.itertuples():
        i = row.Index
        if i+1 < len(failures_df_ar):
            failures_df_ar['tempo_entre_falhas_acumulado'].iat[i+1] = failures_df_ar['tempo_entre_falhas'].iat[i+1] + failures_df_ar['tempo_entre_falhas_acumulado'].iat[i] - failures_df_ar['tempo_sem_operacao'].iat[i+1]

    failures_df_ar['tempo_entre_falhas_acumulado_horas'] = \
    failures_df_ar['tempo_entre_falhas_acumulado'].apply(lambda x: np.round(x.total_seconds()/3600, 2))

    ##### OLEO #####
    failures_df_oleo.drop(index=failures_df_oleo.index[-1],axis=0,inplace=True)
    failures_df_oleo['tempo_entre_falhas_acumulado'] = timedelta(seconds=0) + tempo_ate_primeira_falha_oleo
    for row in failures_df_oleo.itertuples():
        i = row.Index
        if i+1 < len(failures_df_oleo):
            failures_df_oleo['tempo_entre_falhas_acumulado'].iat[i+1] = failures_df_oleo['tempo_entre_falhas'].iat[i+1] + failures_df_oleo['tempo_entre_falhas_acumulado'].iat[i] - failures_df_oleo['tempo_sem_operacao'].iat[i+1]

    failures_df_oleo['tempo_entre_falhas_acumulado_horas'] = \
    failures_df_oleo['tempo_entre_falhas_acumulado'].apply(lambda x: np.round(x.total_seconds()/3600, 2))

    ##### MOTOR #####
    failures_df_mot.drop(index=failures_df_mot.index[-1],axis=0,inplace=True)
    failures_df_mot['tempo_entre_falhas_acumulado'] = timedelta(seconds=0) + tempo_ate_primeira_falha_mot
    for row in failures_df_mot.itertuples():
        i = row.Index
        if i+1 < len(failures_df_mot):
            failures_df_mot['tempo_entre_falhas_acumulado'].iat[i+1] = failures_df_mot['tempo_entre_falhas'].iat[i+1] + failures_df_mot['tempo_entre_falhas_acumulado'].iat[i] - failures_df_mot['tempo_sem_operacao'].iat[i+1]

    failures_df_mot['tempo_entre_falhas_acumulado_horas'] = \
    failures_df_mot['tempo_entre_falhas_acumulado'].apply(lambda x: np.round(x.total_seconds()/3600, 2))

    return failures_df_oleo, failures_df_ar, failures_df_mot

# MAIN
# if __name__ == "__main__":
print("-----------------------------------------------------------------------------")
print("----------- Starting SENAI Reliability Automation for monitoring ------------")
print("-----------------------------------------------------------------------------")
print("--- This automation extracts, treats and loads data from reg_comp_senai   ---")
print("--- table, which collects events from multiple IoT sensors installed in   ---")
print("--- an air compressor located in SENAI laboratory. This data is used to   ---")
print("--- populate two other tables (aggregated data and failures events) and   ---")
print("--- it is also used to fed a dashboard for near-real-time monitoring.     ---")
print(" ")

print("----------- 1. Extracting data from reg_comp_senai_resp table ---------------")
mysql = MySQLConnection()
data = MySQLConnection.get_data(mysql, 'reg_comp_senai_resp')
raw_data = pd.DataFrame.from_dict(data)
data = variable_selection(raw_data)
print("Data Period: From ", min(data['datetime']), " to ", max(data['datetime']), " today")
oper_hours_over_period = get_oper_hours_sum(data)
print('Operation hours for the entire period: ', oper_hours_over_period)
print("----------- COMPLETED -------------------------------------------------------")
print(" ")

print("-----------------------------------------------------------------------------")
print("----------- 2. Cleaning unwanted data ---------------------------------------")
cleaned_data, cleaned_data_di00 = clean_data(data)
print(f'Events (raw data): \t{raw_data.shape[0]}')
print(f'Events (after cleanup): \t{cleaned_data.shape[0]}')
print("----------- COMPLETED -------------------------------------------------------")
print(" ")

print("-----------------------------------------------------------------------------")
print("----------- 3. Resampling data for 30 minutes interval ----------------------")
data_30min = resample_data(data)
cleaned_data_30min, cleaned_data_30min_di00 = clean_data(data_30min)
print(f'Events (raw data): \t{raw_data.shape[0]}')
print(f'Events (after cleanup): \t{cleaned_data_30min.shape[0]}')
print("----------- COMPLETED -------------------------------------------------------")
print(" ")

print("-----------------------------------------------------------------------------")
print("----------- 4. Adding limits to dataframe -----------------------------------")
cleaned_data_30min = add_limits_to_dataframe(cleaned_data_30min)
cleaned_data_30min_di00 = add_limits_to_dataframe(cleaned_data_30min_di00)
print("----------- COMPLETED -------------------------------------------------------")
print(" ")
print(cleaned_data_30min.head())

print("-----------------------------------------------------------------------------")
print("----------- 5. Populating database and saving backup xlsx file --------------")
MySQLConnection.save_data(mysql, cleaned_data_30min, "data_agg_30min")
cleaned_data_30min.to_excel("cleaned_data_30_min.xlsx")
print("----------- COMPLETED -------------------------------------------------------")
print(" ")

print("-----------------------------------------------------------------------------")
print("----------- 6. Calculating and creating failures dataframe ------------------")
failures_df_oleo, failures_df_ar, failures_df_mot = create_failures_dataframe(cleaned_data_30min, cleaned_data_30min_di00)
MySQLConnection.save_data(mysql, failures_df_oleo, "failures_df_oleo")
MySQLConnection.save_data(mysql, failures_df_ar, "failures_df_ar")
MySQLConnection.save_data(mysql, failures_df_mot, "failures_df_mot")
failures_df_oleo.to_excel("failures_df_oleo.xlsx")
failures_df_ar.to_excel("failures_df_ar.xlsx")
failures_df_mot.to_excel("failures_df_mot.xlsx")
print("----------- COMPLETED -------------------------------------------------------")
print(" ")

tempo_entre_falhas_acumulado_motor = list(failures_df_mot[:-1].tempo_entre_falhas_acumulado.apply(lambda x: np.round(x.total_seconds()/3600, 2)))[1:]
tempo_entre_falhas_acumulado_ar = list(failures_df_ar[:-1].tempo_entre_falhas_acumulado.apply(lambda x: np.round(x.total_seconds()/3600, 2)))
tempo_entre_falhas_acumulado_oleo = list(failures_df_oleo[:-1].tempo_entre_falhas_acumulado.apply(lambda x: np.round(x.total_seconds()/3600, 2)))

# TAB 1 (Sistema Completo) - MEAN CUMULATIVE FUNCTION
times = [tempo_entre_falhas_acumulado_motor, tempo_entre_falhas_acumulado_ar, tempo_entre_falhas_acumulado_oleo]
times_concat = tempo_entre_falhas_acumulado_motor + tempo_entre_falhas_acumulado_ar + tempo_entre_falhas_acumulado_oleo

obj = MCF_parametric(data=times)

alpha = obj.results['Point Estimate'][0]
beta = obj.results['Point Estimate'][1]
x = np.linspace(0, max(times_concat), 10000, endpoint=True)
y = [pow(t/alpha, beta) for t in x]

alpha_lower_ci = obj.results['Lower CI'][0]
beta_lower_ci = obj.results['Lower CI'][1]
x_lower_ci = np.linspace(0, max(times_concat), 10000, endpoint=True)
y_lower_ci = [pow(t/alpha_lower_ci, beta_lower_ci) for t in x_lower_ci]

alpha_upper_ci = obj.results['Upper CI'][0]
beta_upper_ci = obj.results['Upper CI'][1]
x_upper_ci = np.linspace(0, max(times_concat), 10000, endpoint=True)
y_upper_ci = [pow(t/alpha_upper_ci, beta_upper_ci) for t in x_upper_ci]

trace1 = go.Scatter(x=obj.times, y=obj.MCF, name='Real', mode='markers', marker=dict(color='black', size=4))
trace2 = go.Scatter(x=x, y=y, mode='lines', name='MCF', line=dict(color='royalblue', width=1))
trace3 = go.Scatter(x=x_upper_ci, y=y_upper_ci, mode = 'lines', name='Limite Superior', line=dict(color='royalblue', width=1, dash='dash'))
trace4 = go.Scatter(x=x_lower_ci, y=y_lower_ci, mode = 'lines', name='Limite Inferior', line=dict(color='royalblue', width=1, dash='dash'))

mcf_fig = make_subplots()
mcf_fig.add_trace(trace1)
mcf_fig.add_trace(trace2)
mcf_fig.add_trace(trace3)
mcf_fig.add_trace(trace4)    

# TAB 1 (Sistema Completo) - INTENSIDADE ACUMULADA
x = np.linspace(1, max(times_concat), 10000, endpoint=True)
y = [beta/pow(alpha, beta) * pow(t, beta-1) for t in x]

trace1 = go.Scatter(x=x, y=y, mode='lines', name='Intensidade', line=dict(color='royalblue', width=1))

intens = make_subplots()
intens.add_trace(trace1)

# TAB 1 (Sistema Completo) - GRÁFICO DE CONFIABILIDADE
#TODO

# TAB 2 (Temperatura Motor) - MEAN CUMULATIVE FUNCTION
times = [tempo_entre_falhas_acumulado_motor]
obj_motor = MCF_parametric(data=times)
plt.show()

alpha_motor = obj_motor.results['Point Estimate'][0]
beta_motor = obj_motor.results['Point Estimate'][1]
x_motor = np.linspace(0, max(tempo_entre_falhas_acumulado_motor), 10000, endpoint=True)
y_motor = [pow(t/alpha_motor, beta_motor) for t in x_motor]

alpha_lower_ci_motor = obj_motor.results['Lower CI'][0]
beta_lower_ci_motor = obj_motor.results['Lower CI'][1]
x_lower_ci_motor = np.linspace(0, max(tempo_entre_falhas_acumulado_motor), 10000, endpoint=True)
y_lower_ci_motor = [pow(t/alpha_lower_ci_motor, beta_lower_ci_motor) for t in x_lower_ci_motor]

alpha_upper_ci_motor = obj_motor.results['Upper CI'][0]
beta_upper_ci_motor = obj_motor.results['Upper CI'][1]
x_upper_ci_motor = np.linspace(0, max(tempo_entre_falhas_acumulado_motor), 10000, endpoint=True)
y_upper_ci_motor = [pow(t/alpha_upper_ci_motor, beta_upper_ci_motor) for t in x_upper_ci_motor]

trace1 = go.Scatter(x=obj_motor.times, y=obj_motor.MCF, name='Real', mode='markers', marker=dict(color='black', size=4))
trace2 = go.Scatter(x=x_motor, y=y_motor, mode='lines', name='MCF', line=dict(color='royalblue', width=1))
trace3 = go.Scatter(x=x_upper_ci_motor, y=y_upper_ci_motor, mode = 'lines', name='Limite Superior', line=dict(color='royalblue', width=1, dash='dash'))
trace4 = go.Scatter(x=x_lower_ci_motor, y=y_lower_ci_motor, mode = 'lines', name='Limite Inferior', line=dict(color='royalblue', width=1, dash='dash'))

mcf_fig_motor = make_subplots()
mcf_fig_motor.add_trace(trace1)
mcf_fig_motor.add_trace(trace2)
mcf_fig_motor.add_trace(trace3)
mcf_fig_motor.add_trace(trace4)

# TAB 2 (Temperatura Motor) - INTENSIDADE ACUMULADA
x_intens_motor = np.linspace(1, max(tempo_entre_falhas_acumulado_motor), 10000, endpoint=True)
y_intens_motor = [beta/pow(alpha_motor, beta_motor) * pow(t, beta_motor-1) for t in x_intens_motor]

trace1 = go.Scatter(x=x_intens_motor, y=y_intens_motor, mode='lines', name='Intensidade', line=dict(color='royalblue', width=1))

intens_motor = make_subplots()
intens_motor.add_trace(trace1)

# TAB 2 (Temperatura Motor) - GRÁFICO DE CONFIABILIDADE
# TODO

# TAB 3 (Temperatura Óleo) - MEAN CUMULATIVE FUNCTION
times = [tempo_entre_falhas_acumulado_oleo]
obj_oil = MCF_parametric(data=times)
plt.show()

alpha_oil = obj_oil.results['Point Estimate'][0]
beta_oil = obj_oil.results['Point Estimate'][1]
x_oil = np.linspace(0, max(tempo_entre_falhas_acumulado_oleo), 10000, endpoint=True)
y_oil = [pow(t/alpha_oil, beta_oil) for t in x_oil]

alpha_lower_ci_oil = obj_oil.results['Lower CI'][0]
beta_lower_ci_oil = obj_oil.results['Lower CI'][1]
x_lower_ci_oil = np.linspace(0, max(tempo_entre_falhas_acumulado_oleo), 10000, endpoint=True)
y_lower_ci_oil = [pow(t/alpha_lower_ci_oil, beta_lower_ci_oil) for t in x_lower_ci_oil]

alpha_upper_ci_oil = obj_oil.results['Upper CI'][0]
beta_upper_ci_oil = obj_oil.results['Upper CI'][1]
x_upper_ci_oil = np.linspace(0, max(tempo_entre_falhas_acumulado_oleo), 10000, endpoint=True)
y_upper_ci_oil = [pow(t/alpha_upper_ci_oil, beta_upper_ci_oil) for t in x_upper_ci_oil]

trace1 = go.Scatter(x=obj_oil.times, y=obj_oil.MCF, name='Real', mode='markers', marker=dict(color='black', size=4))
trace2 = go.Scatter(x=x_oil, y=y_oil, mode='lines', name='MCF', line=dict(color='royalblue', width=1))
trace3 = go.Scatter(x=x_upper_ci_oil, y=y_upper_ci_oil, mode = 'lines', name='Limite Superior', line=dict(color='royalblue', width=1, dash='dash'))
trace4 = go.Scatter(x=x_lower_ci_oil, y=y_lower_ci_oil, mode = 'lines', name='Limite Inferior', line=dict(color='royalblue', width=1, dash='dash'))

mcf_fig_oil = make_subplots()
mcf_fig_oil.add_trace(trace1)
mcf_fig_oil.add_trace(trace2)
mcf_fig_oil.add_trace(trace3)
mcf_fig_oil.add_trace(trace4)

# TAB 3 (Temperatura Óleo) - INTENSIDADE ACUMULADA
x_intens_oil = np.linspace(1, max(tempo_entre_falhas_acumulado_oleo), 10000, endpoint=True)
y_intens_oil = [beta/pow(alpha_oil, beta_oil) * pow(t, beta_oil-1) for t in x_intens_oil]

trace1 = go.Scatter(x=x_intens_oil, y=y_intens_oil, mode='lines', name='Intensidade', line=dict(color='royalblue', width=1))

intens_oil = make_subplots()
intens_oil.add_trace(trace1)

# TAB 3 (Temperatura Óleo) - GRÁFICO DE CONFIABILIDADE,
# TODO

# TAB 4 (Temperatura Ar) - MEAN CUMULATIVE FUNCTION
times = [tempo_entre_falhas_acumulado_ar]
obj_air = MCF_parametric(data=times)
plt.show()

alpha_air = obj_air.results['Point Estimate'][0]
beta_air = obj_air.results['Point Estimate'][1]
x_air = np.linspace(0, max(tempo_entre_falhas_acumulado_ar), 10000, endpoint=True)
y_air = [pow(t/alpha_air, beta_air) for t in x_air]

alpha_lower_ci_air = obj_air.results['Lower CI'][0]
beta_lower_ci_air = obj_air.results['Lower CI'][1]
x_lower_ci_air = np.linspace(0, max(tempo_entre_falhas_acumulado_ar), 10000, endpoint=True)
y_lower_ci_air = [pow(t/alpha_lower_ci_air, beta_lower_ci_air) for t in x_lower_ci_air]

alpha_upper_ci_air = obj_air.results['Upper CI'][0]
beta_upper_ci_air = obj_air.results['Upper CI'][1]
x_upper_ci_air = np.linspace(0, max(tempo_entre_falhas_acumulado_ar), 10000, endpoint=True)
y_upper_ci_air = [pow(t/alpha_upper_ci_air, beta_upper_ci_air) for t in x_upper_ci_air]

trace1 = go.Scatter(x=obj_air.times, y=obj_air.MCF, name='Real', mode='markers', marker=dict(color='black', size=4))
trace2 = go.Scatter(x=x_air, y=y_air, mode='lines', name='MCF', line=dict(color='royalblue', width=1))
trace3 = go.Scatter(x=x_upper_ci_air, y=y_upper_ci_air, mode = 'lines', name='Limite Superior', line=dict(color='royalblue', width=1, dash='dash'))
trace4 = go.Scatter(x=x_lower_ci_air, y=y_lower_ci_air, mode = 'lines', name='Limite Inferior', line=dict(color='royalblue', width=1, dash='dash'))

mcf_fig_air = make_subplots()
mcf_fig_air.add_trace(trace1)
mcf_fig_air.add_trace(trace2)
mcf_fig_air.add_trace(trace3)
mcf_fig_air.add_trace(trace4)

# TAB 4 (Temperatura Ar) - INTENSIDADE ACUMULADA
x_intens_air = np.linspace(1, max(tempo_entre_falhas_acumulado_ar), 10000, endpoint=True)
y_intens_air = [beta/pow(alpha_air, beta_air) * pow(t, beta_air-1) for t in x_intens_air]

trace1 = go.Scatter(x=x_intens_air, y=y_intens_air, mode='lines', name='Intensidade', line=dict(color='royalblue', width=1))

intens_air = make_subplots()
intens_air.add_trace(trace1)

# TAB 4 (Temperatura Ar) - GRÁFICO DE CONFIABILIDADE
#TODO

# TAB 2 (Temperatura Motor) - Valores Agg 30 min
t_mot_90quartil = dados[['t_mot', 'datetime']].resample('30min', on='datetime').agg(func='quantile', **{'q':[0.9]})['t_mot'].reset_index()
t_mot_90quartil.columns = ['datetime90quartil','level','90quartil']

t_mot_10quartil = dados[['t_mot', 'datetime']].resample('30min', on='datetime').agg(func='quantile', **{'q':[0.1]})['t_mot'].reset_index()
t_mot_10quartil.columns = ['datetime10quartil','level','10quartil']

t_mot_media = dados[['t_mot', 'datetime']].resample('30min', on='datetime').agg(func='mean')['t_mot'].reset_index()
t_mot_media.columns = ['datetime', 'media']

t_mot_df = pd.concat([t_mot_media, t_mot_10quartil, t_mot_90quartil], axis=1).dropna()
t_mot_df['10quartil'] = t_mot_df['media']-t_mot_df['10quartil']
t_mot_df['90quartil'] = -t_mot_df['media']+t_mot_df['90quartil']

limite_t_mot_array = np.empty(len(t_mot_df.index))
limite_t_mot_array.fill(limite_t_mot)

t_mot_index = np.arange(1, len(t_mot_media)+1, 1)

color = 'rgba(00,100,00,0.12)' if t_mot_df['media'].iat[-1] < limite_t_mot else 'rgba(100,00,00,0.2)'

error_bar_t_mot_plotly = go.Scatter(
                                        x=t_mot_df['datetime'], 
                                        y=t_mot_df['media'], 
                                        name='30 min T Motor',
                                        mode='markers', 
                                        marker=dict(color=('rgba(00,00,00,0.8)'), size=4), 
                                        error_y=dict(
                                            type='data',
                                            symmetric=False,
                                            array=t_mot_df['90quartil'],
                                            arrayminus=t_mot_df['10quartil'],
                                            thickness=1,
                                            width=0)
                                        )
error_bar_t_mot_plotly_limite = go.Scatter(x=t_mot_df['datetime'],
                                            y=limite_t_mot_array,
                                            name='limite T Motor',
                                            mode='lines',
                                            line=dict(color='darkred', width=3, dash='dash')
                                            )

mot_fig = make_subplots()
mot_fig.add_trace(error_bar_t_mot_plotly)
mot_fig.add_trace(error_bar_t_mot_plotly_limite)

mot_fig.update_layout(height = 300,
                  width = 1200,
                  title='Temperatura do Motor - 30 minutos',
                  title_xanchor='center',
                  title_x=0.5,
                  title_y=1,
                  xaxis_title='Tempo [h]',
                  yaxis_title='Temperatura [ºC]',
                  showlegend=False,
                  margin=dict(t=30, b=30),
                  plot_bgcolor=color)

# TAB 3 (Temperatura Óleo) - Valores Agg 30 min
t_oleo_90quartil = dados[['t_oleo', 'datetime']].resample('30min', on='datetime').agg(func='quantile', **{'q':[0.9]})['t_oleo'].reset_index()
t_oleo_90quartil.columns = ['datetime90quartil','level','90quartil']

t_oleo_10quartil = dados[['t_oleo', 'datetime']].resample('30min', on='datetime').agg(func='quantile', **{'q':[0.1]})['t_oleo'].reset_index()
t_oleo_10quartil.columns = ['datetime10quartil','level','10quartil']

t_oleo_media = dados[['t_oleo', 'datetime']].resample('30min', on='datetime').agg(func='mean')['t_oleo'].reset_index()
t_oleo_media.columns = ['datetime', 'media']

t_oleo_df = pd.concat([t_oleo_media, t_oleo_10quartil, t_oleo_90quartil], axis=1).dropna()
t_oleo_df['10quartil'] = t_oleo_df['media']-t_oleo_df['10quartil']
t_oleo_df['90quartil'] = -t_oleo_df['media']+t_oleo_df['90quartil']

limite_t_oleo_array = np.empty(len(t_oleo_df.index))
limite_t_oleo_array.fill(limite_t_oleo)

t_oleo_index = np.arange(1, len(t_oleo_media)+1, 1)

color = 'rgba(00,100,00,0.12)' if t_oleo_df['media'].iat[-1] < limite_t_oleo else 'rgba(100,00,00,0.2)'

error_bar_t_oleo_plotly = go.Scatter(
                                        x=t_oleo_df['datetime'], 
                                        y=t_oleo_df['media'], 
                                        name='30 min T Óleo',
                                        mode='markers', 
                                        marker=dict(color=('rgba(00,00,00,0.6)'), size=4), 
                                        error_y=dict(
                                            type='data',
                                            symmetric=False,
                                            array=t_oleo_df['10quartil'],
                                            arrayminus=t_oleo_df['90quartil'],
                                            thickness=1,
                                            width=0)
                                        )

error_bar_t_oleo_plotly_limite = go.Scatter(x=t_oleo_df['datetime'],
                                            y=limite_t_oleo_array,
                                            name='limite T Óleo',
                                            mode='lines',
                                            line=dict(color='darkred', width=3, dash='dash')
                                            )

oil_fig = make_subplots()
oil_fig.add_trace(error_bar_t_oleo_plotly)
oil_fig.add_trace(error_bar_t_oleo_plotly_limite)

oil_fig.update_layout(height = 300,
                  width = 1200,
                  title='Temperatura do Óleo - 30 minutos',
                  title_xanchor='center',
                  title_x=0.5,
                  title_y=1,
                  xaxis_title='Tempo [h]',
                  yaxis_title='Temperatura [ºC]',
                  showlegend=False,
                  margin=dict(t=30, b=30),
                  plot_bgcolor=color)

# TAB 4 (Temperatura Ar) - Valores Agg 30 min
t_ar_90quartil = dados[['t_ar', 'datetime']].resample('30min', on='datetime').agg(func='quantile', **{'q':[0.9]})['t_ar'].reset_index()
t_ar_90quartil.columns = ['datetime90quartil','level','90quartil']

t_ar_10quartil = dados[['t_ar', 'datetime']].resample('30min', on='datetime').agg(func='quantile', **{'q':[0.1]})['t_ar'].reset_index()
t_ar_10quartil.columns = ['datetime10quartil','level','10quartil']

t_ar_media = dados[['t_ar', 'datetime']].resample('30min', on='datetime').agg(func='mean')['t_ar'].reset_index()
t_ar_media.columns = ['datetime', 'media']

t_ar_df = pd.concat([t_ar_media, t_ar_10quartil, t_ar_90quartil], axis=1).dropna()
t_ar_df['10quartil'] = t_ar_df['media']-t_ar_df['10quartil']
t_ar_df['90quartil'] = -t_ar_df['media']+t_ar_df['90quartil']

limite_t_ar_array = np.empty(len(t_ar_df.index))
limite_t_ar_array.fill(limite_t_ar)

t_ar_index = np.arange(1, len(t_ar_media)+1, 1)

error_bar_t_ar_plotly = go.Scatter(
                                        x=t_ar_df['datetime'], 
                                        y=t_ar_df['media'], 
                                        name='30 min T Ar',
                                        mode='markers', 
                                        marker=dict(color=('rgba(00,00,00,0.8)'), size=4), 
                                        error_y=dict(
                                            type='data',
                                            symmetric=False,
                                            array=t_ar_df['90quartil'],
                                            arrayminus=t_ar_df['10quartil'],
                                            thickness=1,
                                            width=0)
                                        )

error_bar_t_ar_plotly_limite = go.Scatter(x=t_ar_df['datetime'],
                                            y=limite_t_ar_array,
                                            name='limite T Ar',
                                            mode='lines',
                                            line=dict(color='darkred', width=3, dash='dash')
                                            )
air_fig = make_subplots()
air_fig.add_trace(error_bar_t_ar_plotly)
air_fig.add_trace(error_bar_t_ar_plotly_limite)

air_fig.update_layout(height = 300,
                  width = 1200,
                  title='Temperatura do Ar - 30 minutos',
                  title_xanchor='center',
                  title_x=0.5,
                  title_y=1,
                  xaxis_title='Tempo [h]',
                  yaxis_title='Temperatura [ºC]',
                  showlegend=False,
                  margin=dict(t=30, b=30),
                  plot_bgcolor=color)

# TAB 1 (Sistema Completo) - Ajuste de Layout dos Gráficos
color = 'rgba(00,100,00,0.12)' if (t_ar_df['media'].iat[-1] < limite_t_ar and t_oleo_df['media'].iat[-1] < limite_t_oleo and t_mot_df['media'].iat[-1] < limite_t_mot) else 'rgba(100,00,00,0.2)'
mcf_fig.update_layout(height = 400,
                  width = 600,
                  title='Mean Cumulative Function',
                title_x=0.42,
                  title_y=1,
                  title_xanchor='center',
                  xaxis_title='Tempo [h]',
                  yaxis_title='Número Acumulado de Eventos',
                  margin=dict(t=30, b=30),
                  plot_bgcolor=color)

color = 'rgba(00,100,00,0.12)' if (t_ar_df['media'].iat[-1] < limite_t_ar and t_oleo_df['media'].iat[-1] < limite_t_oleo and t_mot_df['media'].iat[-1] < limite_t_mot) else 'rgba(100,00,00,0.2)'
intens.update_layout(height = 400,
                  width = 600,
                  title='Intensidade de Falhas Acumulada',
                  title_x=0.5,
                  title_y=1,
                  title_xanchor='center',
                  xaxis_title='Tempo [h]',
                  yaxis_title='Intensidade',
                  margin=dict(t=30, b=30),
                  plot_bgcolor=color)

# TAB 2 (Temperatura Motor) - Ajuste de Layout dos Gráficos
color = 'rgba(00,100,00,0.12)' if (t_mot_df['media'].iat[-1] < limite_t_mot) else 'rgba(100,00,00,0.2)'
mcf_fig_motor.update_layout(height = 400,
                  width = 600,
                  title='Mean Cumulative Function',
                title_x=0.42,
                  title_y=1,
                  title_xanchor='center',
                  xaxis_title='Tempo [h]',
                  yaxis_title='Número Acumulado de Eventos',
                  margin=dict(t=30, b=30),
                  plot_bgcolor=color)

intens_motor.update_layout(height = 400,
                  width = 600,
                  title='Intensidade de Falhas Acumulada',
                  title_x=0.5,
                  title_y=1,
                  title_xanchor='center',
                  xaxis_title='Tempo [h]',
                  yaxis_title='Intensidade',
                  margin=dict(t=30, b=30),
                  plot_bgcolor=color)

# TAB 3 (Temperatura Óleo) - Ajuste de Layout dos Gráficos
color = 'rgba(00,100,00,0.12)' if (t_oleo_df['media'].iat[-1] < limite_t_oleo) else 'rgba(100,00,00,0.2)'
mcf_fig_oil.update_layout(height = 400,
                  width = 600,
                  title='Mean Cumulative Function',
                title_x=0.42,
                  title_y=1,
                  title_xanchor='center',
                  xaxis_title='Tempo [h]',
                  yaxis_title='Número Acumulado de Eventos',
                  margin=dict(t=30, b=30),
                  plot_bgcolor=color)

intens_oil.update_layout(height = 400,
                  width = 600,
                  title='Intensidade de Falhas Acumulada',
                  title_x=0.5,
                  title_y=1,
                  title_xanchor='center',
                  xaxis_title='Tempo [h]',
                  yaxis_title='Intensidade',
                  margin=dict(t=30, b=30),
                  plot_bgcolor=color)

# TAB 4 (Temperatura Ar) - Ajuste de Layout dos Gráficos
color = 'rgba(00,100,00,0.12)' if (t_ar_df['media'].iat[-1] < limite_t_ar) else 'rgba(100,00,00,0.2)'
mcf_fig_air.update_layout(height = 400,
                  width = 600,
                  title='Mean Cumulative Function',
                title_x=0.42,
                  title_y=1,
                  title_xanchor='center',
                  xaxis_title='Tempo [h]',
                  yaxis_title='Número Acumulado de Eventos',
                  margin=dict(t=30, b=30),
                  plot_bgcolor=color)

intens_air.update_layout(height = 400,
                  width = 600,
                  title='Intensidade de Falhas Acumulada',
                  title_x=0.5,
                  title_y=1,
                  title_xanchor='center',
                  xaxis_title='Tempo [h]',
                  yaxis_title='Intensidade',
                  margin=dict(t=30, b=30),
                  plot_bgcolor=color)

# Setup the style from the link:
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# Embed the style to the dashabord:
app = JupyterDash(__name__)

# TAB 1 (Sistema Completo)
mcf_fig_graph = dcc.Graph(
        id='mcf_fig',
        figure=mcf_fig,
        className="four columns" 
    )

intens_graph = dcc.Graph(
        id='intens',
        figure=intens,
        className="four columns" 
    )

# confiab_graph = dcc.Graph(
#         id='confiab',
#         figure=confiab,
#         className="four columns" 
#     )

row_complete = html.Div(children=[
                            html.Div(children = mcf_fig_graph, style={'width': '33%', 'display': 'inline-block'}),
                            html.Div(children = intens_graph, style={'width': '33%', 'display': 'inline-block'}),
                            html.Div(children = intens_graph, style={'width': '33%', 'display': 'inline-block'})
                                    ])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TAB 2 (Temperatura Motor)
mot_fig_graph = dcc.Graph(
        id='mot_fig',
        figure=mot_fig,
        className="twelve columns" 
    )

mcf_fig_graph_motor = dcc.Graph(
        id='mcf_fig_motor',
        figure=mcf_fig_motor,
        className="four columns" 
    )

intens_graph_motor = dcc.Graph(
        id='intens_motor',
        figure=intens_motor,
        className="four columns" 
    )

# confiab_graph_motor = dcc.Graph(
#         id='confiab_motor',
#         figure=confiab_motor,
#         className="four columns" 
#     )

row_complete_motor = html.Div(children=[
                            html.Div(children = mcf_fig_graph_motor, style={'width': '33%', 'display': 'inline-block'}),
                            html.Div(children = intens_graph_motor, style={'width': '33%', 'display': 'inline-block'}),
                            html.Div(children = intens_graph_motor, style={'width': '33%', 'display': 'inline-block'})
                                    ])
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TAB 3 (Temperatura Óleo)

oil_fig_graph = dcc.Graph(
        id='oil_fig',
        figure=oil_fig,
        className="eight columns" 
    )

mcf_fig_graph_oil = dcc.Graph(
        id='mcf_fig_oil',
        figure=mcf_fig_oil,
        className="four columns" 
    )

intens_graph_oil = dcc.Graph(
        id='intens_oil',
        figure=intens_oil,
        className="four columns" 
    )

# confiab_graph_oil = dcc.Graph(
#         id='confiab_oil',
#         figure=confiab_oil,
#         className="four columns" 
#     )

row_complete_oil = html.Div(children=[
                            html.Div(children = mcf_fig_graph_oil, style={'width': '33%', 'display': 'inline-block'}),
                            html.Div(children = intens_graph_oil, style={'width': '33%', 'display': 'inline-block'}),
                            html.Div(children = intens_graph_oil, style={'width': '33%', 'display': 'inline-block'})
                                    ])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TAB 4 (Temperatura Ar)
air_fig_graph = dcc.Graph(
        id='air_fig',
        figure=air_fig,
        className="eight columns" 
    )

mcf_fig_graph_air = dcc.Graph(
        id='mcf_fig_air',
        figure=mcf_fig_air,
        className="four columns" 
    )

intens_graph_air = dcc.Graph(
        id='intens_air',
        figure=intens_air,
        className="four columns" 
    )

# confiab_graph_air = dcc.Graph(
#         id='confiab_air',
#         figure=confiab_air,
#         className="four columns" 
#     )

row_complete_air = html.Div(children=[
                            html.Div(children = mcf_fig_graph_air, style={'width': '33%', 'display': 'inline-block'}),
                            html.Div(children = intens_graph_air, style={'width': '33%', 'display': 'inline-block'}),
                            html.Div(children = intens_graph_air, style={'width': '33%', 'display': 'inline-block'})
                                    ])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# setup the header
header = html.H2(children="SENAI Reliability Dashboard - Air Compressor")
# setup to rows, graph 1-3 in the first row, and graph4 in the second:
row1 = html.Div(children=[
                            html.Div(children = mcf_fig_graph, style={'width': '30%', 'display': 'inline-block'}),
                            html.Div(children = [oil_fig_graph, mot_fig_graph, air_fig_graph], style={'width': '70%', 'display': 'inline-block'})
                                    ])

# Creating TABS for Oil Temp , Air Temp, Motor Temp and Complete System
tabs_dash = html.Div([
            html.H1('SENAI Reliability Dashboard - Air Compressor', style={'textAlign': 'center'}),
            html.H4('Last update time: ' + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")), style={'textAlign': 'center'}),
            dcc.Tabs(id="tabs-example-graph", value='tab-1-example-graph', children=[
                dcc.Tab(label='Sistema Completo', value='tab-complete-system'),
                dcc.Tab(label='Motor Temp', value='tab-motor-temp'),
                dcc.Tab(label='Oil Temp', value='tab-oil-temp'),
                dcc.Tab(label='Air Temp', value='tab-air-temp')
            ]),
            html.Div(id='tabs-content-example-graph')
        ])

@app.callback(Output('tabs-content-example-graph', 'children'),
              Input('tabs-example-graph', 'value'))
def render_content(tab):
    if tab == 'tab-complete-system':
        return html.Div([
            html.H4(' '),
            row_complete
        ])
    elif tab == 'tab-motor-temp':
        return html.Div([
            html.H4(' '),
            mot_fig_graph,
            row_complete_motor
        ], style={'textAlign': 'center'})
    elif tab == 'tab-oil-temp':
        return html.Div([
            html.H4(' '),
            oil_fig_graph,
            row_complete_oil
        ])
    elif tab == 'tab-air-temp':
        return html.Div([
            html.H4(' '),
            air_fig_graph,
            row_complete_air
        ])



# row3 = html.Div(children=[graph3])
# row4 = html.Div(children=[graph4])

# setup & apply the layout
# app.layout = html.Div(
#                 children=[header, row1], style={"text-align": "center"},
#                 )

def serve_layout():
    # return html.Div(
    #             children=[header, row1], style={"text-align": "center"},
    #             )
    return tabs_dash

app.layout = serve_layout
    
if __name__ == '__main__':
    app.run_server(debug=True)

# TODO: após popular o banco, tratar somente os dados novos (capturados após a última inserção na tabela)
# TODO: criar nova tabela no banco e popular com os dados

