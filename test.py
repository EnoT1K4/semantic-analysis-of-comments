from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm
import pandas as pd
import time
from cassandra.cluster import Cluster

# Connect to the local Cassandra instance
cluster = Cluster(['127.0.0.1'])
session = cluster.connect()

# Set the keyspace and table name
keyspace = 'kursed'
table = 'Comments'

# Now you can execute queries using the session

options = Options()
options.binary_location = '/home/vladislav/drivers/chromedriver'
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(options=options)
comment_section = 'ytd-comment-thread-renderer.ytd-item-section-renderer'



scrapped = []
author_id = 'author-text'
comment_id = 'content-text'
comments = []

driver.get("https://www.youtube.com/watch?v=YbJu8y5mZek")



element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, comment_section)))
last_height = driver.execute_script('return document.documentElement.scrollTop;')
while True:
    driver.execute_script(f'window.scrollTo(0, document.documentElement.scrollHeight);')
    time.sleep(2)
    new_height = driver.execute_script('return document.documentElement.scrollTop;')
    if new_height == last_height:
        break
    last_height = new_height

comment_blocks = driver.find_elements(By.CSS_SELECTOR,comment_section)
for comment_element in comment_blocks:
    author = comment_element.find_element(By.ID, author_id)
    comment = comment_element.find_element(By.ID, comment_id)
    comments.append(comment.text)


comments_df = pd.DataFrame(list(zip(comments)), columns=['comment'], dtype='string')
driver.quit()



# Create a table in Cassandra if it doesn't exist
session.execute(f"CREATE TABLE IF NOT EXISTS {keyspace}.{table} (comment text PRIMARY KEY)")

# Convert the DataFrame to a list of rows
rows = [(comments_df.at[i, 'comment']) for i in range(comments_df.shape[0])]

# Insert the rows into the Cassandra table
session.execute(f"INSERT INTO {keyspace}.{table} (comment) VALUES (%s)", rows)

# Close the connection
cluster.shutdown()
comments_df.to_csv("/home/vladislav/kursed/youtube_comments_data.csv",index=False)

