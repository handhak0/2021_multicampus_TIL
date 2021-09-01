# NLP 

# 02. 텍스트 전처리 

## 1. 토큰화 

### 1) 단어 토큰화 

- 구두점을 제거하고 정제하는 과정 
- 하지만, 예상치 못한 경우들이 있기에 기준을 생각해봐야 하는 경우가 발생 



### 2) 종류 

`from nltk.tokenize import word_tokenize  `

```
['Do', "n't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr.', 'Jone', "'s", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']  
```



`from nltk.tokenize **import** WordPunctTokenizer` 

```
['Don', "'", 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr', '.', 'Jone', "'", 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']  
```



`from tensorflow.keras.preprocessing.text import text_to_word_sequence`

```
["don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'mr', "jone's", 'orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']
```



### 3) 문장 토큰화

> 토큰의 단위가 문장인 경우 

- **from** nltk.tokenize **import** sent_tokenize

```
['His barber kept his word.', 'But keeping such a huge secret to himself was driving him crazy.', 'Finally, the barber went up a mountain and almost to the edge of a cliff.', 'He dug a hole in the midst of some reeds.', 'He looked about, to make sure no one was near.']
```

```
['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']
```

- 한국어는 kss **import** kss 

```
['딥 러닝 자연어 처리가 재미있기는 합니다.', '그런데 문제는 영어보다 한국어로 할 때 너무 어려워요.', '농담아니에요.', '이제 해보면 알걸요?']
```



### 4) 이진 분류기 

> 문장 토큰화에서의 예외 사항을 발생시키는 마침표의 처리를 위해서 입력에 따라 두 개의 클래스로 분류하는 것 

1) 마침표가 단어의 일부분일 경우 
2) 마침표가 정마로 문장의 구분자일 경우 



### 5) 품사 태깅 

- 품사에 따라 단어의 의미가 달라지기도 한다. 
- 단어토큰화 과정에서 각 단어가 어떤 품사로 쓰였는지 구분하는 것 = 품사 태깅 



### 6) NLTK와 KoNLPy를 이용한 영어, 한국어 토큰화 실습 

- 영어 토큰화 

  - **from** nltk.tokenize **import** word_tokenize

  ```
  ['I', 'am', 'actively', 'looking', 'for', 'Ph.D.', 'students', '.', 'and', 'you', 'are', 'a', 'Ph.D.', 'student', '.']
  ```

  - **from** nltk.tag **import** pos_tag (태깅)

    ```
    [('I', 'PRP'), ('am', 'VBP'), ('actively', 'RB'), ('looking', 'VBG'), ('for', 'IN'), ('Ph.D.', 'NNP'), ('students', 'NNS'), ('.', '.'), ('and', 'CC'), ('you', 'PRP'), ('are', 'VBP'), ('a', 'DT'), ('Ph.D.', 'NNP'), ('student', 'NN'), ('.', '.')]
    ```

    

- 한국어 토큰화 

  - **from** konlpy.tag **import** Okt  

  ```python
  okt=Okt()  
  print(okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요")) # 형태소 추출 
  ```

  ```
  ['열심히', '코딩', '한', '당신', ',', '연휴', '에는', '여행', '을', '가봐요']  
  ```

  ```python
  print(okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))   # 품사 태깅 
  ```

  ```
  [('열심히','Adverb'), ('코딩', 'Noun'), ('한', 'Josa'), ('당신', 'Noun'), (',', 'Punctuation'), ('연휴', 'Noun'), ('에는', 'Josa'), ('여행', 'Noun'), ('을', 'Josa'), ('가봐요', 'Verb')]  
  ```

  ```python
  print(okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))  # 명사 추출 
  ```

  ```
  ['코딩', '당신', '연휴', '여행']  
  ```

- - **from** konlpy.tag **import** Kkma  

  ```python
  kkma=Kkma()  
  print(kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
  ['열심히', '코딩', '하', 'ㄴ', '당신', ',', '연휴', '에', '는', '여행', '을', '가보', '아요']  
  
  print(kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))  
  [('열심히','MAG'), ('코딩', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('당신', 'NP'), (',', 'SP'), ('연휴', 'NNG'), ('에', 'JKM'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가보', 'VV'), ('아요', 'EFN')]  
  
  print(kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))  
  ['코딩', '당신', '연휴', '여행']  
  ```



## 2. 정제 및 정규화 

- 규칙에 기반한 표기가 다른 단어들의 통합 ex) US, USA 
- 대, 소문자 통합 
- 불필요한 단어의 제거 
  - 등장빈도가 적은 단어 
  - 길이가 짧은 단어 
- 정규 표현식 



## 3. 어간 추출 및 표제어 추출 

### 1) 표제어 추출 (Lemmatization)

> 기본 사전형 단어 

- 형태학적 파싱을 먼저 진행하는 것 

1) 어간 
2) 접사 



### 2) 어간 추출 (**Stemming**)

- **from** nltk.stem **import** PorterStemmer



### 3) 표제어 추출과 어간 추출의 차이점 

**Stemming**
am → am
the going → the go
having → hav

**Lemmatization**
am → be
the going → the going
having → have



### 4) 한국어에서의 어간 추출 

#### 활용

- 용언의 어간이 어미를 가지는 일 

#### 규칙활용 

- 어간이 어미를 취할 때, 어간의 모습이 일정 

```
잡/어간 + 다/어미
```

#### 불규칙활용 

- 어간이 어미를 취할 때 어간의 모습이 바귀거나 취하는 어미가 특수한 어미일 경우를 말합니다. 





## 4. 불용어 

> 큰 의미가 없는 단어 토큰을 제거하는 작업 

### 1) 영어 불용어 제거하기 

- NLTK에서 불용어 확인하기 

```python
from nltk.corpus import stopwords
stopwords.words('english')[:10]
```

```
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your']  
```



- NLTK를 통해서 불용어 제거하기 

```python
from nltk.corpus import stopwords # 불용어 가져오기 
from nltk.tokenize import word_tokenize # 토큰화 

example = "Family is not an important thing. It's everything."
stop_words = set(stopwords.words('english')) # 불용어 담기

word_tokens = word_tokenize(example) # 토큰화하기 

result = []
for w in word_tokens: 
    if w not in stop_words: # 불용어에 없으면 
        result.append(w) # 리스트 append

print(word_tokens) 
print(result) 
```

```
['Family', 'is', 'not', 'an', 'important', 'thing', '.', 'It', "'s", 'everything', '.'] # 제거 전 
['Family', 'important', 'thing', '.', 'It', "'s", 'everything', '.'] # 제거 후 
```



### 2) 한국어에서 불용어 제거하기 

- 조사, 접속사 등을 제거 
- 불용어 사전을 참고로 불용어 제거 

```python
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

example = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
stop_words = "아무거나 아무렇게나 어찌하든지 같다 비슷하다 예컨대 이럴정도로 하면 아니거든"

# 위의 불용어는 명사가 아닌 단어 중에서 저자가 임의로 선정한 것으로 실제 의미있는 선정 기준이 아님
stop_words=stop_words.split(' ')
word_tokens = word_tokenize(example)

result = [] 
for w in word_tokens: 
    if w not in stop_words: 
        result.append(w) 
# 위의 4줄은 아래의 한 줄로 대체 가능
# result=[word for word in word_tokens if not word in stop_words]

print(word_tokens) 
print(result)
```



## 5. 정규표현식 

### 1) 정규 표현식 문법

| 특수 문자      | 설명                                                         |
| :------------- | :----------------------------------------------------------- |
| .              | 한 개의 임의의 문자를 나타냅니다. (줄바꿈 문자인 \n는 제외)  |
| ?              | 앞의 문자가 존재할 수도 있고, 존재하지 않을 수도 있습니다. (문자가 0개 또는 1개) |
| *              | 앞의 문자가 무한개로 존재할 수도 있고, 존재하지 않을 수도 있습니다. (문자가 0개 이상) |
| +              | 앞의 문자가 최소 한 개 이상 존재합니다. (문자가 1개 이상)    |
| ^              | 뒤의 문자로 문자열이 시작됩니다.                             |
| $              | 앞의 문자로 문자열이 끝납니다.                               |
| {숫자}         | 숫자만큼 반복합니다.                                         |
| {숫자1, 숫자2} | 숫자1 이상 숫자2 이하만큼 반복합니다. ?, *, +를 이것으로 대체할 수 있습니다. |
| {숫자,}        | 숫자 이상만큼 반복합니다.                                    |
| [ ]            | 대괄호 안의 문자들 중 한 개의 문자와 매치합니다. [amk]라고 한다면 a 또는 m 또는 k 중 하나라도 존재하면 매치를 의미합니다. [a-z]와 같이 범위를 지정할 수도 있습니다. [a-zA-Z]는 알파벳 전체를 의미하는 범위이며, 문자열에 알파벳이 존재하면 매치를 의미합니다. |
| [^문자]        | 해당 문자를 제외한 문자를 매치합니다.                        |
| l              | AlB와 같이 쓰이며 A 또는 B의 의미를 가집니다.                |

정규 표현식 문법에는 역 슬래쉬(\)를 이용하여 자주 쓰이는 문자 규칙들이 있습니다.

| 문자 규칙 | 설명                                                         |
| :-------- | :----------------------------------------------------------- |
| \         | 역 슬래쉬 문자 자체를 의미합니다                             |
| \d        | 모든 숫자를 의미합니다. [0-9]와 의미가 동일합니다.           |
| \D        | 숫자를 제외한 모든 문자를 의미합니다. [^0-9]와 의미가 동일합니다. |
| \s        | 공백을 의미합니다. [ \t\n\r\f\v]와 의미가 동일합니다.        |
| \S        | 공백을 제외한 문자를 의미합니다. [^ \t\n\r\f\v]와 의미가 동일합니다. |
| \w        | 문자 또는 숫자를 의미합니다. [a-zA-Z0-9]와 의미가 동일합니다. |
| \W        | 문자 또는 숫자가 아닌 문자를 의미합니다. [^a-zA-Z0-9]와 의미가 동일합니다. |



### 2) 정규표현식 모듈 함수 

정규표현식 모듈에서 지원하는 함수는 이와 같습니다.

| 모듈 함수     | 설명                                                         |
| :------------ | :----------------------------------------------------------- |
| re.compile()  | 정규표현식을 컴파일하는 함수입니다. 다시 말해, 파이썬에게 전해주는 역할을 합니다. 찾고자 하는 패턴이 빈번한 경우에는 미리 컴파일해놓고 사용하면 속도와 편의성면에서 유리합니다. |
| re.search()   | 문자열 전체에 대해서 정규표현식과 매치되는지를 검색합니다.   |
| re.match()    | 문자열의 처음이 정규표현식과 매치되는지를 검색합니다.        |
| re.split()    | 정규 표현식을 기준으로 문자열을 분리하여 리스트로 리턴합니다. |
| re.findall()  | 문자열에서 정규 표현식과 매치되는 모든 경우의 문자열을 찾아서 리스트로 리턴합니다. 만약, 매치되는 문자열이 없다면 빈 리스트가 리턴됩니다. |
| re.finditer() | 문자열에서 정규 표현식과 매치되는 모든 경우의 문자열에 대한 이터레이터 객체를 리턴합니다. |
| re.sub()      | 문자열에서 정규 표현식과 일치하는 부분에 대해서 다른 문자열로 대체합니다. |

앞으로 진행될 실습에서는 re.compile()에 정규 표현식을 컴파일하고, re.search()를 통해서 해당 정규 표현식이 입력 텍스트와 매치되는지를 확인하면서 각 정규 표현식에 대해서 이해해보도록 하겠습니다. re.search() 함수는 매치된다면 Match Object를 리턴하고, 매치되지 않으면 아무런 값도 출력되지 않습니다.



## 6. 정수인코딩 

> 자연어 처리를 위해서 텍스트를 숫자로 바꾸는 기법 

보통은 전처리 또는 빈도수가 높은 단어들만 사용하기 위해서 단어에 대한 빈도수를 기준으로 정렬한 뒤에 부여한다. 



### 1) 정수인코딩 

- 빈도수 순으로 정렬한 단어 집합 만들기 
- 빈도수가 높은 순서대로 차례로 낮은 숫자부터 정수 부여 

```python
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = '<넣을 문장>'

text = sent_tokenize(text)
print(text)

# 정제와 단어 토큰화
vocab = {} # 파이썬의 dictionary 자료형
sentences = []
stop_words = set(stopwords.words('english'))

for i in text:
    sentence = word_tokenize(i) # 단어 토큰화를 수행합니다.
    result = []

    for word in sentence: 
        word = word.lower() # 모든 단어를 소문자화하여 단어의 개수를 줄입니다.
        if word not in stop_words: # 단어 토큰화 된 결과에 대해서 불용어를 제거합니다.
            if len(word) > 2: # 단어 길이가 2이하인 경우에 대하여 추가로 단어를 제거합니다.
                result.append(word)
                if word not in vocab:
                    vocab[word] = 0 
                vocab[word] += 1
    sentences.append(result) 
print(sentences)
```



### 2) Counter 사용하기

```python
from collections import Counter

vocab = Counter(words) # words의 자료형은 리스트임 
vocab_size = 5
vocab = vocab.most_common(vocab_size) # 등장 빈도수가 높은 상위 5개의 단어만 저장
word_to_index = {}
i = 0
for (word, frequency) in vocab :
    i = i+1
    word_to_index[word] = i
print(word_to_index)
```



### 3) NLTK의 FreqDist 사용하기 

```python
from nltk import FreqDist
import numpy as np

# np.hstack으로 문장 구분을 제거하여 입력으로 사용 . ex) ['barber', 'person', 'barber', 'good' ... 중략 ...
vocab = FreqDist(np.hstack(sentences))

vocab_size = 5
vocab = vocab.most_common(vocab_size) # 등장 빈도수가 높은 상위 5개의 단어만 저장

word_to_index = {word[0] : index + 1 for index, word in enumerate(vocab)}
print(word_to_index)
```



### 4) enumerate 이해하기 

> 순서가 있는 자료형 (list, set ...)을 입력 받아 인덱스를 순차적으로 return해준다 

```python
test=['a', 'b', 'c', 'd', 'e']
for index, value in enumerate(test): # 입력의 순서대로 0부터 인덱스를 부여함.
  print("value : {}, index: {}".format(value, index))
```

```
value : a, index: 0
value : b, index: 1
value : c, index: 2
value : d, index: 3
value : e, index: 4
```



## 7. Keras의 텍스트 전처리 

> 정수 인코딩을 위해 케라스의 토크나이저를 사용하기도 함 

- 인덱싱, 카운트

```python
from tensorflow.keras.preprocessing.text import Tokenizer

sentences=[['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences) # fit_on_texts()안에 코퍼스를 입력으로 하면 빈도수를 기준으로 단어 집합을 생성한다.

print(tokenizer.word_index)
print(tokenizer.word_counts)
print(tokenizer.texts_to_sequences(sentences))
```

```
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7, 'good': 8, 'knew': 9, 'driving': 10, 'crazy': 11, 'went': 12, 'mountain': 13}

OrderedDict([('barber', 8), ('person', 3), ('good', 1), ('huge', 5), ('knew', 1), ('secret', 6), ('kept', 4), ('word', 2), ('keeping', 2), ('driving', 1), ('crazy', 1), ('went', 1), ('mountain', 1)])

[[1, 5], [1, 8, 5], [1, 3, 5], [9, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [7, 7, 3, 2, 10, 1, 11], [1, 12, 3, 13]]
```



- 빈도수 상위 5개 

```python
vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 1) # 상위 5개 단어만 사용
tokenizer.fit_on_texts(sentences)

print(tokenizer.word_index)
```

```
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7, 'good': 8, 'knew': 9, 'driving': 10, 'crazy': 11, 'went': 12, 'mountain': 13}
```

=> 결과가 13개 다 나온다. `word_counts`에서도 마찬가지 

=> 실제 적용은 `texts_to_sequences`를 사용할 때 적용됨 

```python
tokenizer = Tokenizer() # num_words를 여기서는 지정하지 않은 상태
tokenizer.fit_on_texts(sentences)

vocab_size = 5
words_frequency = [w for w,c in tokenizer.word_index.items() if c >= vocab_size + 1] # 인덱스가 5 초과인 단어 제거
for w in words_frequency:
    del tokenizer.word_index[w] # 해당 단어에 대한 인덱스 정보를 삭제
    del tokenizer.word_counts[w] # 해당 단어에 대한 카운트 정보를 삭제
print(tokenizer.word_index)
print(tokenizer.word_counts)
print(tokenizer.texts_to_sequences(sentences))
```

```
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
OrderedDict([('barber', 8), ('person', 3), ('huge', 5), ('secret', 6), ('kept', 4)])
[[1, 5], [1, 5], [1, 3, 5], [2], [2, 4, 3, 2], [3, 2], [1, 4], [1, 4], [1, 4, 2], [3, 2, 1], [1, 3]]
```



## 8. 패딩 

> 자연어 처리 중 문장(or 문서)의 길이가 서로 다를 수 있다. 길이가 전부 동일한 문서들에 대해서는 하나의 행렬로 보고, 한꺼번에 처리 가능 

### 1) numpy로 패딩하기 

### 2) 케라스 전처리 도구로 패딩하기 



## 9. 원-핫 인코딩 

> 단어 집합의 크기를 벡터의 차원으로, 표현하고 싶은 단어의 인덱스에 1의 값 부여, 다른 인덱스에는 0 부여 

1) 각 단어에 고유한 인덱스 부여 (정수 인코딩)
2) 표현하고 싶은 단어의 인덱스의 위치에 1을 부여, 다른 인덱스 위치에는 0 부여 



### 1) keras를 이용한 원-핫 인코딩 



### 2) 원-핫 인코딩의 한계 

- 단어의 개수가 늘어날 수록 벡터를 저장하기 위한 공간이 계속 늘어난다는 점

- 단언 간 유사성을 알 수 없음 



## 10. 한국어 전처리 패키지 

### 1) PyKpSpacing 



### 2) Py-Hanspell 



### 3) SOYNLP

- 신조어 문제 
- 학습하기 
- 응집확률 
- 브랜칭 엔트로피 
- L tokenizer 
- 최대 점수 토크나이저 
- 반복되는 문자 정제 









# 03. 언어모델 

> 단어 시퀀스에 확률을 할당하는 일을 하는 모델 

ex) 이전 단어들이 주어졌을 대, 다음 단어를 예측하도록 하는 것 

**언어모델링** : 주어진 단어들로부터 아직 모르는 단어를 예측하는 작업을 말한다. 



## 단어 시퀀스의 확률 할당 

- 기계 번역 
- 오타 교정 
- 음성 인식 



## 





# 통계적 언어 모델 (SLM)

## 문장에 대한 확률 P

- 단어 : 문맥이라는 관계로 인해 이전 단어의 영향을 받아 나온 단어 

- 모든 단어로부터 하나의 문장이 완성됨 
- 