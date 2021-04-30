import requests
import os
from tqdm import tqdm
from bs4 import BeautifulSoup as bs
from urllib.parse import urljoin, urlparse
import shutil # to save it locally
import pandas as pd
import configs
import urllib

def is_valid(url):
    """
    Checks whether `url` is a valid URL.
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def download_new(image_url, pathname):
    ## Set up the image URL and filename
    if pathname not in os.listdir():
        os.mkdir(pathname)
    
    # initializing bad_chars_list
    bad_chars = [';', ':', '!', "*","%","_"]

    # remove bad_chars
    image_name = ''.join((filter(lambda i: i not in bad_chars, image_url.split("/")[-1])))

    filename = os.path.join(pathname, image_name)

    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(image_url,proxies=urllib.request.getproxies(),stream=True)

    # Check if the image was retrieved successfully
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True

        # Open a local file with wb ( write binary ) permission.
        with open(filename,'wb') as f:
            shutil.copyfileobj(r.raw, f)

        #print('Image sucessfully Downloaded: ',filename)
        return filename
    else:
        return None
        #print('Image Couldn\'t be retreived')

def get_all_images_with_name_wiki(url):
    soup = bs(requests.get(url).content, "html.parser")  
    gallerybox = soup.find_all("li", {"class": "gallerybox"})
    urls = []
    titles =[]
    for imgs in gallerybox:
        img_url = imgs.find_all('img')[0].attrs.get("src")
        title = imgs.find_all("div", {"class": "gallerytext"})[0].find_all('a')[0]['title']
        img_url = urljoin(url, img_url)
        # finally, if the url is valid
        if is_valid(img_url):
            filename, file_extension = os.path.splitext(img_url)
            if file_extension == '.jpg':
                urls.append(img_url)
                titles.append(title)
    return pd.DataFrame({"urls":urls,"titles":titles})

def start_image_scraping(download_list,url):
    img_data = []
    for d_list in download_list:
        url_full = urljoin(url,d_list)
        print("Getting Data From:", url_full)
        img_data.append(get_all_images_with_name_wiki(url_full))

    return pd.concat(img_data,ignore_index=True)

def start_image_downloading(img_data,folder_name):
    local_path = []
    for img in tqdm(img_data['urls'].to_list()):
        # for each image, download it
        filename = download_new(img,folder_name)
        local_path.append(filename)
    img_data['local_path']  = local_path
    return img_data.dropna().reset_index(drop=True)

download_list = configs.download_list
url = configs.wiki_url
save_folder = configs.image_folder

def main():
    print("_______Image Scraping From Wikipedia Pages Started_________")
    img_data = start_image_scraping(download_list,url)
    print("_______Image Scraping From Wikipedia Pages Done_________")
    print("\n_______Image Downloading From Wikipedia Pages Started_________")
    final_data = start_image_downloading(img_data,save_folder)
    print("_______Image Downloading From Wikipedia Pages Done_________")
    print("\n Number of Images Stored:",len(os.listdir(save_folder)))
    print("\n_______Storing the Result [image_data.csv] file_________")
    final_data.to_csv(configs.image_data_csv,index=False)
    print("\n_______Storing the Result Done_________")

if __name__ == '__main__':
    main()
