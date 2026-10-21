from collections import Counter

def tokenizer(text):
  tokens = []
  for element in text:
    if not element.isspace():
      tokens.append(element)
  return tokens

"""
  文字列をtokenizeし、2文字毎の登場確率を返却する
"""
def training_bigram(text):
  tokenized = tokenizer(text)
  bi_tokens = {}
  for i in range(len(tokenized)-1):
    key = tokenized[i] + tokenized[i+1]
    if not key in bi_tokens:
      bi_tokens[key] = 1
    else:
      bi_tokens[key] += 1

  total = sum(bi_tokens.values())

  for key, count in bi_tokens.items():
    bi_tokens[key] = count / total

  return bi_tokens

# text = "ABACA"
# print(training_bigram(text))

"""
  とある文字列の次にくる文字列を場合分けする
"""
def count_next_chars(text):
    tokenized = tokenizer(text)
    result = {}

    for i in range(len(tokenized)-1):
      firstChar = tokenized[i]
      nextChar = tokenized[i+1]

      if not firstChar in result:
        result[firstChar] = {}

      if not nextChar in result[firstChar]:
        result[firstChar][nextChar] = 1
      else:
        result[firstChar][nextChar] += 1

    return result

# text = "ABACA"
# print(count_next_chars(text))

"""
  今度は、確率で表示する
"""
def cal_next_chars(text):
  tokenized = tokenizer(text)
  result = {}

  for i in range(len(tokenized)-1):
    firstChar = tokenized[i]
    nextChar = tokenized[i+1]

    if not firstChar in result:
      result[firstChar] = {}

    if not nextChar in result[firstChar]:
      result[firstChar][nextChar] = 1
    else:
      result[firstChar][nextChar] += 1

  for k, v in result.items():
    s = sum(v.values())
    for innerK, innerV in v.items():
      result[k][innerK] = innerV / s

  return result

# text = "ABACA"
# print(cal_next_chars(text))

"""
  次の文字列を提示する
"""
def predict_next_chars(text, q):
  prob = cal_next_chars(text)
  specificProb = {}
  if(q in prob): specificProb = prob[q]
  return specificProb

# q = "A"
# text = "ABACA"
# print(predict_next_chars(text, q))

"""
  次の文字列を含めて返却する
"""
def return_next_chars(text, q):
  cand = predict_next_chars(text, q)
  ordered = sorted(cand.items(), key=lambda item:item[1], reverse=True)
  return q+ordered[0][0]

q = "A"
text = "ABACACA"
print(return_next_chars(text, q))