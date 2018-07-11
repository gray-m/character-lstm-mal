ANIME_DIV_CLASS = 'seasonal-anime'


class Anime(object):

    def __init__(self, title, synopsis):
        self.title = title
        self.synopsis = synopsis


class Page(object):

    def __init__(self, parsed_html):
        self.anime_list = self._build_anime_list(parsed_html)

    def _build_anime_list(self, parsed):
        ret = []
        anime_divs = parsed.find_all('div', class_=ANIME_DIV_CLASS)
        for div in anime_divs:
            ret.append(self._anime_from_div(div))
        return ret

    def _anime_from_div(self, div):
        title = div.find('p', class_='title-text').text
        title = title.replace('\n', '')
        synopsis = self._synopsis(div)
        return Anime(title, synopsis)

    def _synopsis(self, div):
        div = div.find('div', class_='synopsis')
        synopsis = div.find('span', class_='preline').text
        return synopsis.replace('\n', '')

