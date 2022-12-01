import pymysql
import pandas as pd


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


# MAIN
if __name__ == "__main__":
    data = get_data()

    dataframe = pd.DataFrame.from_dict(data[0]['result'])
    print(dataframe)


