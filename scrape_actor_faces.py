from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import io
from os.path import join
import time
from dl_utils.misc import check_dir
import requests
from PIL import Image
from faces_train.facenet_pytorch import MTCNN
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
    if face_img is None: # just skip if other chars in the image
        print(f"found 0 faces in {url} so skipping")
        return
    if len(face_img) > 1: # just skip if other chars in the image
        print(f"found {len(face_img)} faces in {url} so skipping")
        return
    pil_face_img = Image.fromarray(prepare_for_pil(face_img.squeeze(0)))
    if 'Krendler' in file_path:
        breakpoint()
    with open(file_path, 'wb') as f:
        pil_face_img.save(f, "JPEG", quality=95)
    print(f"SUCCESS - saved {url} - as {file_path}")

def scrape_from_url(url, movie_name):
    check_dir(char_faces_dir:=join('data/scraped_char_faces', movie_name))
    with webdriver.Chrome() as wd:
        actions = ActionChains(wd)
        wd.get(url)
        char_names = [x.find_element(By.TAG_NAME, 'span').text for x in  wd.find_elements(By.CSS_SELECTOR, 'a[data-testid=cast-item-characters-link]')]
        time.sleep(1)
        wd.find_element(By.CSS_SELECTOR, 'button[data-testid=reject-button]').click() # cookie consent shite
        wd.maximize_window()
        time.sleep(1)
        wd.execute_script("window.scrollTo(0, 2000)")
        time.sleep(1)

        for char_name in char_names:
            check_dir(char_dir:=join(char_faces_dir, char_name))
            tcd = wd.find_element(By.LINK_TEXT, char_name).find_element(By.XPATH, '..')
            actions.move_to_element(tcd).perform()
            tcd.click()
            char_face_elements = wd.find_elements(By.CSS_SELECTOR, 'a[class=titlecharacters-image-grid__thumbnail-link]')
            for i, cfe in enumerate(char_face_elements):
                img_children = cfe.find_elements(By.TAG_NAME, 'img')
                assert len(img_children) == 1
                img_url = img_children[0].get_attribute('src')
                save_fpath = join(char_dir, f'img{i}.jpg')
                save_and_crop_img_from_url(file_path=save_fpath, url=img_url)
            time.sleep(1)
            wd.back()
            time.sleep(0.5)


scrape_from_url('https://www.imdb.com/title/tt0102926/?ref_=fn_al_tt_1', 'silence-of-lambs')
