import math
import argparse
import pickle


class Vocabulary(object):


    def __init__(self, raw_data, size):
        self.mapping = self._build_dict(raw_data, size)


    def _build_dict(self, raw_data, size):
        counts = {}
        total = 0
        for anime in raw_data:
            total += len(anime.synopsis + anime.title)
            for char in anime.synopsis + anime.title:
                if char in counts:
                    counts[char] += 1
                else:
                    counts[char] = 1

        if size is None:
            l = list(counts)
            return {c: i for i, c in enumerate(l)}

        l = list(sorted(counts, key=lambda c: counts[c]))

        if size >= 1: # return top <size> most common characters
            l = l[-size:]
            return {c: i for i, c in enumerate(l)}

        # size < 1: return the characters that make up this percent of all chars
        j = 0
        n_chars = 0
        limit = size*len(l)
        while n_chars < limit:
            j += 1
            n_chars += counts[l[-j]]

        l = l[-j:]
        return {c: i for i, c in enumerate(l)}

