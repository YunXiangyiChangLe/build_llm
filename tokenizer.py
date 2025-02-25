import re


class SimpleTokenizerV1:
    def __int__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encoder(self,text):
        preprocessed=re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed=[item.strip() for item in preprocessed if item.strip()]
        ids=[self.str_to_int[s] for s in preprocessed]
        return ids

    def decoder(self,ids):
        text=" ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
