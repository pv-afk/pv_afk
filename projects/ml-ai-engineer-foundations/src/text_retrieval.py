from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DOCUMENTS = [
    {
        "id": "ml-pipeline",
        "title": "ML pipeline",
        "text": "Pipeline объединяет подготовку признаков и модель, снижая риск утечки данных.",
    },
    {
        "id": "cross-validation",
        "title": "Cross-validation",
        "text": "Кросс-валидация оценивает устойчивость модели на нескольких разбиениях выборки.",
    },
    {
        "id": "classification-metrics",
        "title": "Classification metrics",
        "text": "Precision, recall, F1 и ROC-AUC описывают разные стороны качества классификатора.",
    },
    {
        "id": "embeddings",
        "title": "Text embeddings",
        "text": "Текстовые embeddings представляют смысл фрагмента числовым вектором.",
    },
    {
        "id": "rag",
        "title": "Retrieval augmented generation",
        "text": "RAG сначала находит релевантный контекст, затем передаёт его языковой модели.",
    },
    {
        "id": "vector-database",
        "title": "Vector database",
        "text": "Векторная база хранит embeddings и выполняет быстрый поиск ближайших документов.",
    },
    {
        "id": "prompting",
        "title": "Prompt design",
        "text": "Хороший prompt задаёт роль, контекст, ограничения и формат ожидаемого ответа.",
    },
    {
        "id": "monitoring",
        "title": "Model monitoring",
        "text": "Мониторинг отслеживает качество, задержку, ошибки и изменение входных данных.",
    },
]


def search(query: str, limit: int = 3) -> list[dict[str, float | str]]:
    texts = [document["text"] for document in DOCUMENTS]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    document_matrix = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, document_matrix).ravel()
    order = scores.argsort()[::-1][:limit]

    return [
        {
            "id": DOCUMENTS[index]["id"],
            "title": DOCUMENTS[index]["title"],
            "score": float(scores[index]),
        }
        for index in order
    ]


def run() -> dict[str, str | list[dict[str, float | str]]]:
    query = "как найти контекст для ответа языковой модели"
    return {"query": query, "results": search(query)}


if __name__ == "__main__":
    print(run())

