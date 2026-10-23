from collections import Counter

# 文字列を１文字ごとに分解する
def tokenize_japanese(text: str) -> list[str]:
  result = []
  for char in text:
    if not char.isspace():
      result.append(char)
  return result
  # List Comprehension
  # return [
  #   char for char in text if not char.isspace()
  #   ]

# １文字毎の登場確率を算出する
def train_unigram(text):
  tokens = tokenize_japanese(text)
  counts = Counter(tokens) # {値：登場回数}の形式にできる
  total = len(tokens)
  probabilities = {}

  for token, count in counts.items():
    probabilities[token] = count / total
  return probabilities

  # Dictionary Comprehension
  # return {
  #   token: count / total
  #   for token, count in counts.items()
  # }

text = "プログラミングは楽しいですか？プログラムは動きますか？プロですか？ローミング"
probs = train_unigram(text)

# prob.itemsに対してvalueのdesc整列しprintする
for char, prob in sorted(
  probs.items(),
  key=lambda x: x[1],
  reverse=True
):
  print(f"{char}: {prob:.3f}")
