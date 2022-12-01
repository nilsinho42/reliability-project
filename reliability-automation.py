import pymysql
import pandas as pd
import numpy as np

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

    def connect(self):
        conn = pymysql.connect(
            host=self.host,
            user=self.db_user,
            password=self.db_password,
            db=self.db_name,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return conn


def get_data():
    mysql = MySQLConnection()
    try:
        conn = mysql.connect()
    except Exception as e:
        return {
            "result": None,
            "detail": e.args[1],
        }, 500

    cursor = conn.cursor()
    sql_query = \
        """ SELECT * 
            FROM reg_comp_senai_resp
            LIMIT 100000
        """
    cursor.execute(sql_query)
    results = cursor.fetchall()

    if len(results) > 0:
        return {
            "result": results,
            "detail": "",
        }, 200
    else:
        return {
            "result": None,
            "detail": "Query returned no results",
        }, 404

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
    data = data[data['di_00']!=AIR_COMPRESSOR_OFF_STATUS]
    data = data[data['t_mot']<UPPER_RANGE_T_MOT]
    data = data[data['t_mot']>LOWER_RANGE_T_MOT]
    data = data[data['t_oleo']<UPPER_RANGE_T_OIL]
    data = data[data['t_oleo']>LOWER_RANGE_T_OIL]
    data = data[data['freq']>LOWER_RANGE_FREQ]
    return data

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

# MAIN
if __name__ == "__main__":
    print("-----------------------------------------------------------------------------")
    print("----------- Starting SENAI Reliability Automation for monitoring ------------")
    print("-----------------------------------------------------------------------------")
    print("--- This automation extract, treat and load data from reg_comp_senai_resp ---")
    print("--- table, which collects events from multiple IoT sensors installed in   ---")
    print("--- an air compressor located in SENAI laboratory. This data is used to   ---")
    print("--- populate two other tables (aggregated data and failures events) and   ---")
    print("--- it is also used to fed a dashboard for near-real-time monitoring.     ---")
    print(" ")

    print("----------- 1. Extracting data from reg_comp_senai_resp table ---------------")
    data = get_data()
    raw_data = pd.DataFrame.from_dict(data[0]['result'])
    data = variable_selection(raw_data)
    print("Data Period: ")
    print("From ", min(data['datetime']), " to ", max(data['datetime']), " today")
    oper_hours_over_period = get_oper_hours_sum(data)
    print('Operation hours for the entire period: ', oper_hours_over_period)
    print("----------- COMPLETED -------------------------------------------------------")
    print(" ")

    print("-----------------------------------------------------------------------------")
    print("----------- 2. Cleaning unwanted data ---------------------------------------")
    cleaned_data = clean_data(data)
    print(f'Events (raw data): \t{raw_data.shape[0]}')
    print(f'Events (after cleanup): \t{cleaned_data.shape[0]}')
    print("----------- COMPLETED -------------------------------------------------------")
    print(" ")
          
    print("-----------------------------------------------------------------------------")
    print("----------- 3. Resampling data for 30 minutes interval ----------------------")
    data_30min = resample_data(data)
    cleaned_data_30min = clean_data(data_30min)
    print(f'Events (raw data): \t{raw_data.shape[0]}')
    print(f'Events (after cleanup): \t{cleaned_data_30min.shape[0]}')
    print("----------- COMPLETED -------------------------------------------------------")
    print(" ")

    print("-----------------------------------------------------------------------------")
    print("----------- 4. Adding limits to dataframe -----------------------------------")
    cleaned_data_30min = add_limits_to_dataframe(cleaned_data_30min)
    print("----------- COMPLETED -------------------------------------------------------")
    print(" ")
    print(cleaned_data_30min.head())

    print("-----------------------------------------------------------------------------")
    print("----------- 5. Populating database and saving backup xlsx file --------------")
    cleaned_data_30min.to_excel("cleaned_data_30_min.xlsx")
    print("----------- COMPLETED -------------------------------------------------------")
    print(" ")


# TODO: após popular o banco, tratar somente os dados novos (capturados após a última inserção na tabela)
# TODO: criar nova tabela no banco e popular com os dados

