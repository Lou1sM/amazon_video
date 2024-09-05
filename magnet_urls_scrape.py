from selenium import webdriver
import os
import shutil
import json
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import selenium
import io
from os.path import join
import requests
import time


with open('moviesumm_testset_names.txt') as f:
    movie_names = f.read().split('\n')

mn2magnets = {}
with webdriver.Chrome() as wd:
    actions = ActionChains(wd)
    wd.get('https://www.pirateproxy-bay.com/')
    for mn_with_date in movie_names:
        mn, date = mn_with_date.split('_')
        to_search = f'{mn} ({date})'
        search_box = wd.find_element(By.ID, 'search_input')
        search_box.clear()
        search_box.send_keys(to_search)
        search_box.send_keys(Keys.ENTER)
        magnet_link_icons = wd.find_elements(By.CSS_SELECTOR, "[title='Download this torrent using magnet']")
        magnet_links = [x.get_attribute('href') for x in magnet_link_icons]
        mn2magnets[mn] = magnet_links
        wd.back()

with open('scraped_magnet_links.json', 'w') as f:
    json.dump(mn2magnets, f)

for k,v in mn2magnets.items():
    if v != []:
         with open(f'watch-folder-deluge-magnets/{k}.magnet','w') as f:
            f.write(v[0])
