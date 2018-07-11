import os
import glob
import argparse
import pickle

import bs4

import anime


def process_page(raw):
    soup = bs4.BeautifulSoup(raw, 'html.parser')
    page = anime.Page(soup)
    return page.anime_list


parser = argparse.ArgumentParser()
parser.add_argument('dir')
parser.add_argument('outfile')
parser.add_argument('--genre', default='')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

genre = args.genre.capitalize() # they're all capitalized when we get them
glob_str = genre + '*.raw'
path = os.path.join(args.dir, glob_str)

all_anime = set() # many have multiple genres attached to them
for raw in glob.glob(path):
    if args.verbose:
        print('processing', raw, '...')
    with open(raw, 'r') as raw_html:
        all_anime.update(process_page(raw_html))

with open(args.outfile, 'wb') as f:
    pickle.dump(all_anime, f)

print('data pickled in {}.'.format(args.outfile))

