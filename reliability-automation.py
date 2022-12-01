# import pymysql
import pandas as pd
import numpy as np
from datetime import timedelta
import sqlalchemy as db

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
if __name__ == "__main__":
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
# TODO: após popular o banco, tratar somente os dados novos (capturados após a última inserção na tabela)
# TODO: criar nova tabela no banco e popular com os dados

