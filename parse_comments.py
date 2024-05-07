from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm
import pandas as pd
import time






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

def tabular_text_to_string(tabular_text: str) -> str:
    """
    Converts a tabular text into a string.

    Args:
        tabular_text (str): The tabular text to convert.

    Returns:
        str: The converted string.
    """
    # Split the tabular text into lines
    lines = tabular_text.strip().split('\n')

    # Split each line into words
    words = [line.split() for line in lines]

    # Join the words into a single string
    string = ' '.join([' '.join(word) for word in words])

    return string



element = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, comment_section)))
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
    comments.append(str(tabular_text_to_string(comment.text)))


comments_df = pd.DataFrame(list(zip(comments)), columns=['comment'], dtype='string')
comments_df.dropna()

driver.quit()
comments_df.to_csv("/home/vladislav/kursed/youtube_comments_data.csv",index=False)