"""
Рекомендательная система фильмов: логика рекомендаций.
Загружает сохранённые артефакты и предоставляет функции для Streamlit.
"""
import pickle
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class MovieRecommender:
    """
    Класс-обёртка над сохранёнными артефактами рекомендательной системы.
    Предоставляет методы для получения рекомендаций разных типов.
    """

    def __init__(self, artifacts_path: str = "model_artifacts/recommender.pkl"):
        if not os.path.exists(artifacts_path):
            raise FileNotFoundError(
                f"Не найден файл модели: {artifacts_path}. "
                "Запустите ноутбук movies-recommendation-system.ipynb для его создания."
            )

        with open(artifacts_path, "rb") as f:
            art = pickle.load(f)

        # Загружаем артефакты из pickle
        self.user_knn = art["user_knn"]
        self.movie_knn = art["movie_knn"]
        self.users_pivot = art["users_pivot"]      # User-ID × Movie-Title
        self.movie_pivot = art["movie_pivot"]      # Movie-Title × User-ID
        self.users_sparse = art["users_sparse"]
        self.movie_sparse = art["movie_sparse"]
        self.popularity = art["popularity"]

    # ---------- 1. Популярные фильмы (weighted rating) ----------
    def popular_movies(self, n: int = 10) -> pd.DataFrame:
        """
        Возвращает топ-N фильмов по взвешенному рейтингу (IMDb Bayesian Average).

        Returns:
            DataFrame с колонками: title, avg_rating, count_rating, w_score
        """
        return self.popularity.head(n)[
            ["title", "avg_rating", "count_rating", "w_score"]
        ].reset_index(drop=True)

    # ---------- 2. Похожие фильмы (item-based KNN) ----------
    def similar_movies(self, movie_title: str, n: int = 10) -> pd.DataFrame:
        """
        Возвращает n фильмов, наиболее похожих на заданный, используя
        косинусное расстояние между векторами оценок пользователей.

        Args:
            movie_title: точное название фильма из movie_pivot.index
            n: количество рекомендаций

        Returns:
            DataFrame с колонками: title, Similarity
        """
        if movie_title not in self.movie_pivot.index:
            return pd.DataFrame(columns=["title", "Similarity"])

        idx = self.movie_pivot.index.get_loc(movie_title)
        vec = self.movie_sparse[idx]

        # n+1, потому что первым соседом всегда будет сам фильм
        distances, indices = self.movie_knn.kneighbors(vec, n_neighbors=n + 1)
        distances = distances.flatten()[1:]
        indices = indices.flatten()[1:]

        return pd.DataFrame({
            "title": self.movie_pivot.index[indices],
            "Similarity": np.round(1 - distances, 4),
        }).reset_index(drop=True)

    # ---------- 3. Персональные рекомендации (user-based KNN) ----------
    def recommend_for_user(self, user_id: int, n: int = 10) -> pd.Series:
        """
        Рекомендует фильмы пользователю на основе оценок похожих пользователей.

        Args:
            user_id: ID пользователя из users_pivot.index
            n: количество рекомендаций

        Returns:
            Series с индексом = название фильма, значения = предсказанный скор
        """
        if user_id not in self.users_pivot.index:
            return pd.Series(dtype=float)

        idx = self.users_pivot.index.get_loc(user_id)
        vec = self.users_sparse[idx]

        distances, indices = self.user_knn.kneighbors(vec, n_neighbors=11)
        distances = distances.flatten()
        indices = indices.flatten()

        # Первый сосед — сам пользователь, исключаем его
        similar_users = indices[1:]
        similarities = 1 - distances[1:]

        # Взвешенное среднее оценок соседей по схожести
        neigh_ratings = self.users_pivot.iloc[similar_users]
        weighted = neigh_ratings.T.dot(similarities) / similarities.sum()
        weighted = pd.Series(weighted, index=self.users_pivot.columns)

        # Исключаем уже просмотренные фильмы
        seen = self.users_pivot.loc[user_id]
        seen = seen[seen > 0].index
        weighted = weighted.drop(seen, errors="ignore")

        return weighted.sort_values(ascending=False).head(n)

    # ---------- 4. Гибридные рекомендации ----------
    def hybrid_recommend(
        self,
        user_id: int | None = None,
        liked_movie: str | None = None,
        n: int = 10,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3,
    ) -> pd.Series:
        """
        Гибридный рекомендатель, комбинирующий три сигнала:
            alpha — вес популярности (w_score)
            beta  — вес схожести с liked_movie (item-based KNN)
            gamma — вес персональных предпочтений (user-based KNN)

        Args:
            user_id: опциональный ID пользователя
            liked_movie: опциональное название любимого фильма
            n: количество рекомендаций
            alpha, beta, gamma: веса сигналов в диапазоне [0, 1]

        Returns:
            Series с индексом = название фильма, значения = итоговый скор
        """
        scores = pd.Series(dtype=float)

        # 1. Популярность
        if alpha > 0:
            pop = self.popularity.set_index("title")["w_score"]
            pop_norm = (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)
            scores = scores.add(pop_norm * alpha, fill_value=0)

        # 2. Схожесть с любимым фильмом
        if liked_movie and beta > 0:
            sim = self.similar_movies(liked_movie, n=200)
            if not sim.empty:
                sim_scores = sim.set_index("title")["Similarity"]
                scores = scores.add(sim_scores * beta, fill_value=0)

        # 3. Персональные предпочтения
        if user_id is not None and gamma > 0:
            recs = self.recommend_for_user(user_id, n=200)
            if len(recs) > 0:
                norm = (recs - recs.min()) / (recs.max() - recs.min() + 1e-9)
                scores = scores.add(norm * gamma, fill_value=0)

        # Исключаем уже просмотренные пользователем фильмы
        if user_id is not None and user_id in self.users_pivot.index:
            seen = self.users_pivot.loc[user_id]
            seen = seen[seen > 0].index
            scores = scores.drop(seen, errors="ignore")

        return scores.sort_values(ascending=False).head(n)

    # ---------- Служебные свойства ----------
    @property
    def known_users(self) -> list:
        """Список известных пользователей (индексы users_pivot)."""
        return self.users_pivot.index.tolist()

    @property
    def known_movies(self) -> list:
        """Отсортированный список известных фильмов (индексы movie_pivot)."""
        return sorted(self.movie_pivot.index.tolist())

    # ---------- Обратная совместимость (алиасы) ----------
    # Если app.py использует старые имена "books" — они будут работать
    popular_books = popular_movies
    similar_books = similar_movies
    known_books = known_movies
