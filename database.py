#!/usr/bin/env python3
import psycopg2
import sys
import numpy as np
import filters

class DataBaseInterface():
    def __init__(self, hostname, database, username, password, port):
        self._host = hostname
        self._database = database
        self._user = username
        self._password = password
        self._port = port
        self._connection = None
        self._cursor = None

    def database_connect(self):
        try:
            self._connection = psycopg2.connect(
                host=self._host,
                database=self._database,
                user=self._user,
                password=self._password,
                port=self._port,
                connect_timeout=5
            )
            self._cursor = self._connection.cursor()
        except psycopg2.OperationalError as e:
            print(f"❌ Cannot connect to database: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

    def database_disconnect(self):
        if self._cursor:
            self._cursor.close()
        if self._connection:
            self._connection.close()

    def get_data(self, table_name):
        try:
            self.database_connect()
            self._cursor.execute("SELECT samples FROM " + table_name)
            rows = self._cursor.fetchall()
            self._cursor.close()
            self._connection.close()
            return rows
        except psycopg2.OperationalError as e:
            print(f"❌ Cannot connect to database: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

    def insert_data(self, table_name, data):
        try:
            self.database_connect()
            self._cursor.execute("INSERT INTO " + table_name + " (samples) VALUES (%s)", (data,))
            print(f"Rows affected: { self._cursor.rowcount}")
            print(f"✅ Data inserted into '{table_name}' successfully.")
            self._connection.commit()
            self._cursor.close()
        except psycopg2.OperationalError as e:
            print(f"❌ Cannot connect to database: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
        return True

    def create_table(self, table_name):
        try:
            self.database_connect()
            drop_table_query = f"DROP TABLE IF EXISTS {table_name} CASCADE;"
            self._cursor.execute(drop_table_query)
            print(f"✅ Table '{table_name}' dropped successfully.")
            create_table_query = f"CREATE TABLE {table_name} (samples DOUBLE PRECISION[]);"
            print(f"✅ Table '{table_name}' created successfully.")
            self._cursor.execute(create_table_query)
            self._connection .commit()
            self._cursor.close()
        except psycopg2.OperationalError as e:
            print(f"❌ Cannot connect to database: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
        return True

if __name__ == "__main__":
    db = DataBaseInterface('localhost','myapp','postgres','password',5432)
    for shape_index in range(len(filters.shapes)):
        for key in filters.shapes[shape_index]['filters']:
            print("")
            array = db.get_data(key)
            if array:
                print("filter:", key)
                for row in array:
                    print("-----------------")
                    for item in row:
                            print(item)