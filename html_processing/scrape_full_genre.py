import os
import argparse
import subprocess
import math

import bs4
import progressbar

import genre


BASE_URL = 'https://myanimelist.net/anime/genre/'
PER_PAGE = 100

parser = argparse.ArgumentParser()
parser.add_argument('genre_number', type=int)
parser.add_argument('--save_dir', default='../html_raw')
args = parser.parse_args()

url = BASE_URL + str(args.genre_number) + '/?page='

dummy_path = os.path.join(args.save_dir, 'page1.raw')
subprocess.run(['wget', '-q', '-O', dummy_path, url+str(1)])
with open(dummy_path, 'r') as f:
    parsed = bs4.BeautifulSoup(f, 'html.parser')

g = genre.Genre(parsed)
genre_name = g.name
num_pages = int(math.ceil(g.num_shows/PER_PAGE))

subprocess.run(['mv', dummy_path, os.path.join(args.save_dir, g.name+'1.raw')])

def url_path_pairs(start, end):
    page_num = start
    while page_num <= end:
        page_url = BASE_URL + str(args.genre_number) + '/?page=' + str(page_num)
        yield page_url, \
                os.path.join(args.save_dir, g.name+str(page_num)+'.raw')
        page_num += 1

for url, path in url_path_pairs(2, num_pages):
    print('getting page', url, '...')
    subprocess.run(['wget', '-q', '-O', path, url])

