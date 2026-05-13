# Word2Vec Embeddings from Scratch

Учебная реализация Word2Vec / Skip-Gram Negative Sampling на `NumPy` для построения эмбеддингов русскоязычного текста.

## Что было реализовано

- очистка текста и построение словаря;
- генерация положительного и отрицательного контекста;
- обучение эмбеддингов градиентным спуском;
- сохранение `vocab`, матриц `W`, `C` и усредненных эмбеддингов;
- поиск ближайших слов и проверка через cosine similarity.

## Стек

`Python`, `NumPy`, `Jupyter Notebook`, `Google Colab`

## Структура

```text
02-word2vec-embeddings/
├── README.md
├── requirements.txt
├── .gitignore
└── word2vec_embeddings.ipynb
```

## Запуск

### Google Colab

1. Открыть `word2vec_embeddings.ipynb`.
2. Положить корпус в Google Drive.
3. Проверить пути в первой ячейке:

```python
BASE_PATH = "/content/drive/MyDrive/LW2/data"
SAVE_DIR = Path("/content/drive/MyDrive/PSRSII_embeddings_LR1")
```

4. Выполнить ноутбук сверху вниз.

### Локально

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ai-development-course/02-word2vec-embeddings/requirements.txt
jupyter notebook ai-development-course/02-word2vec-embeddings/word2vec_embeddings.ipynb
```

При необходимости можно переопределить пути:

```bash
export WORD2VEC_BASE_PATH="ai-development-course/02-word2vec-embeddings/data"
export WORD2VEC_SAVE_DIR="ai-development-course/02-word2vec-embeddings/outputs"
```

## Результат

В конце ноутбука выводятся ближайшие слова для выбранных target words, значения cosine similarity и пример de-embedding через перебор.

Большие данные, `.npy`, JSON-словари и `outputs/` не добавляются в Git и игнорируются через `.gitignore`.
