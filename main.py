with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_txt = f.read()
print(len(raw_txt))
print(raw_txt[0:10])
