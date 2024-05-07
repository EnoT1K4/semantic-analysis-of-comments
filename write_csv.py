from cassandra.cluster import Cluster
import csv
import uuid

cluster = Cluster(['127.0.0.1'])
session = cluster.connect()

keyspace = 'kursed'
table = 'labeled_tweets'

session.execute(f"USE {keyspace}")
session.execute(f"CREATE TABLE IF NOT EXISTS {table} (id uuid Primary Key, text text)")


csv_file = '/home/vladislav/kursed/youtube_comments_data.csv'
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  
    for row in reader:
        id = uuid.uuid4()
        text = str(row[0])
        session.execute(f"INSERT INTO {table} (id, text, label) VALUES (%s, %s)", (id, text))

session.shutdown()
