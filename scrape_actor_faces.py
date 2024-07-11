from selenium import webdriver, common
import io
from os.path import join
import numpy as np
from bs4 import BeautifulSoup
from dl_utils.misc import check_dir
import requests
from PIL import Image
from facenet_pytorch import MTCNN
from utils import prepare_for_pil


mtc = MTCNN(image_size=160, margin=10, min_face_size=20, thresholds=[0.8, 0.8, 0.9], factor=0.709, post_process=True, device='cuda', keep_all=True)

def save_and_crop_img_from_url(file_path:str,url:str):
    try:
        image_content = requests.get(url).content
    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")
    image_file = io.BytesIO(image_content)
    image = Image.open(image_file).convert('RGB')
    face_img = mtc(image)
    pil_face_img = Image.fromarray(prepare_for_pil(face_img.squeeze(0)))
    with open(file_path, 'wb') as f:
        pil_face_img.save(f, "JPEG", quality=95)
    print(f"SUCCESS - saved {url} - as {file_path}")

def scrape_from_url(url, movie_name):
    dir_name = join('data/scraped_faces', movie_name)
    check_dir(dir_name)
    #with webdriver.Chrome('/home/louis/chromedriver-linux64/chromedriver') as wd:
    with webdriver.Chrome() as wd:
        wd.get(url)
        #wd.maximize_window()
        #time.sleep(2)
        soup = BeautifulSoup(wd.page_source, 'html.parser')
        aface_divs = [d for d in soup.find_all('div') if d.get('data-testid') == 'title-cast-item']
        for afd in aface_divs:
            char_name = afd.find('span').text
            img_elm = afd.find('img')
            if img_elm is None:
                print('No image found for char_name')
                continue
            actor_name = img_elm['alt']
            img_fpath = join(dir_name, f'{char_name}.jpg')
            save_and_crop_img_from_url(file_path=img_fpath, url=img_elm['src'])


scrape_from_url('https://www.imdb.com/title/tt0102926/?ref_=fn_al_tt_1', 'silence-of-lambs')
