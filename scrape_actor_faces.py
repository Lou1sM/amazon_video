from selenium import webdriver
import os
import shutil
import json
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import selenium
import io
from os.path import join
import time
import numpy as np
from dl_utils.misc import check_dir
import requests
from PIL import Image
#from faces_train.facenet_pytorch import MTCNN
#from utils import prepare_for_pil
from deepface import DeepFace
from nltk.corpus import names
from utils import path_list, shim

male_names = names.words('male.txt')
female_names = names.words('female.txt')


def save_and_crop_img_from_url(file_path:str,url:str, char_name):
    try:
        image_content = requests.get(url).content
    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")
    image_file = io.BytesIO(image_content)
    image_ar = np.array(Image.open(image_file))
    detected_faces = DeepFace.extract_faces(image_ar, enforce_detection=False, detector_backend='fastmtcnn')
    detected_faces.sort(key=lambda x:x['confidence'], reverse=True)

    if len(detected_faces)>1:
        print(f'Skipping cuz detected {len(detected_faces)} faces')
    elif (conf:=detected_faces[0]['confidence']) < 0.5:
        print(f'Skipping cuz conf only {conf}')
    else:
        assert len(detected_faces)==1
        det_gender = DeepFace.analyze(image_ar, actions='gender', detector_backend='fastmtcnn')[0]['dominant_gender']
        if char_name.split(' ')[0] in female_names and det_gender == 'Man':
            print(f'excluding cuz name is female but face is male')
            return
        if char_name.split(' ')[0] in male_names and det_gender == 'Woman':
            print(f'excluding cuz name is male but face is female')
            return
        np.save(file_path, image_ar) # deepface expects whole image for comparison and extraction, not just the face
        assert (np.load(file_path)==image_ar).all()
        assert DeepFace.extract_faces(np.load(file_path), enforce_detection=False, detector_backend='fastmtcnn')[0]['confidence'] > 0.5
        print(f"SUCCESS - saved {url} - as {file_path}")

def scrape_from_url(url, movie_name):
    check_dir(raw_char_faces_dir:=join('data/raw_scraped_char_faces', movie_name))
    check_dir(char_faces_dir:=join('data/scraped_char_faces', movie_name))
    with webdriver.Chrome() as wd:
        actions = ActionChains(wd)
        wd.get(url)
        char_names = [x.find_element(By.TAG_NAME, 'span').text for x in  wd.find_elements(By.CSS_SELECTOR, 'a[data-testid=cast-item-characters-link]')]
        time.sleep(1)
        try:
            wd.find_element(By.CSS_SELECTOR, 'button[data-testid=reject-button]').click() # cookie consent shite
        except selenium.common.exceptions.NoSuchElementException:
            pass
        wd.maximize_window()
        time.sleep(1)
        wd.execute_script("window.scrollTo(0, 2000)")
        time.sleep(1)

        nimages_per_char = {}
        for char_name in char_names:
            check_dir(raw_char_dir:=join(raw_char_faces_dir, char_name))
            tcd = wd.find_element(By.LINK_TEXT, char_name).find_element(By.XPATH, '..')
            actions.move_to_element(tcd).perform()
            tcd.click()
            char_face_elements = wd.find_elements(By.CSS_SELECTOR, 'a[class=titlecharacters-image-grid__thumbnail-link]')
            for i, cfe in enumerate(char_face_elements):
                img_children = cfe.find_elements(By.TAG_NAME, 'img')
                assert len(img_children) == 1
                img_url = img_children[0].get_attribute('src')
                #save_fpath = join(char_dir, f'img{i}.jpg')
                save_fpath = join(raw_char_dir, f'img{i}.npy')
                save_and_crop_img_from_url(file_path=save_fpath, url=img_url, char_name=char_name)
            nimages_per_char[char_name] = len(char_face_elements)
            maybe_more_imgs = wd.find_elements(By.CSS_SELECTOR, 'a[class=titlecharacters-image-grid__see-more-link]')
            if len(maybe_more_imgs) > 0:
                assert len(maybe_more_imgs) == 1
                more_imgs_text = maybe_more_imgs[0].text
                assert more_imgs_text.endswith(' more photos')
                nimages_per_char[char_name] += int(more_imgs_text.removesuffix(' more photos'))

            print(nimages_per_char[char_name])
            import networkx as nx
            saved_fpaths = path_list(raw_char_dir)
            N = len(saved_fpaths)
            if N > 0:
                cross_verified = np.eye(N)
                for i in range(N):
                    for j in range(i+1,N):
                        is_verified = DeepFace.verify(np.load(saved_fpaths[i]), np.load(saved_fpaths[j]), detector_backend='fastmtcnn')['verified']
                        cross_verified[i,j] = is_verified
                        cross_verified[j,i] = is_verified

                G = nx.from_numpy_array(cross_verified)
                max_clique = max(nx.algorithms.clique.find_cliques(G), key=len)
                check_dir(char_dir:=join(char_faces_dir, char_name))
                for i,fp in enumerate(saved_fpaths):
                    if i in max_clique:
                        new_fp = os.path.join(char_dir, os.path.basename(fp))
                        shutil.copy(fp, new_fp)

            time.sleep(1)
            wd.back()
            time.sleep(0.5)

        #nimages_per_char = {k:(v+1)/(sum(nimages_per_char.values())+1) for k,v in nimages_per_char.items()}
        with open(f'data/nimages-per-char/{movie_name}-nimages-per-char.json', 'w') as f:
            json.dump(nimages_per_char, f)




#scrape_from_url('https://www.imdb.com/title/tt0102926/?ref_=fn_al_tt_1', 'silence-of-lambs')
scrape_from_url('https://www.imdb.com/title/tt1798709/?ref_=tt_ch', 'her_2013')
