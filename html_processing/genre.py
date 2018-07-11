SPAN_CLASS = 'fw-n'

class Genre(object):

    def __init__(self, parsed_html):
        self.name = parsed_html.find('title').text.split(' ')[0]
        self.name = self.name.replace('\n', '')
        self.num_shows = self._num_shows(parsed_html)


    def _has_one_class(self, tag):
        return len(tag.attrs['class']) == 1


    def _num_shows(self, parsed_html):
        # hacky as all hell
        spans = parsed_html.find_all('span', class_=SPAN_CLASS)
        candidates = list(filter(self._has_one_class, spans))
        assert len(candidates)==1
        return int(candidates[0].text[1:-1].replace(',',''))

