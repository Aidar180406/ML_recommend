import pickle
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, artifacts_path: str = "model_artifacts/recommender.pkl"):
        if not os.path.exists(artifacts_path):
            raise FileNotFoundError(
                f"Не найден файл модели: {artifacts_path}. "
                "Запустите ноутбук movies-recommendation-system.ipynb для его создания."
            )
        with open(artifacts_path, "rb") as f:
            art = pickle.load(f)

        self.user_knn = art["user_knn"]
        self.movie_knn = art["movie_knn"]
        self.users_pivot = art["users_pivot"]
        self.movie_pivot = art["movie_pivot"]
        self.users_sparse = art["users_sparse"]
        self.movie_sparse = art["movie_sparse"]
        self.popularity = art["popularity"]

        # Артефакты для content-based рекомендаций
        self.movies_content = art["movies_content"]
        self.content_matrix = art["content_matrix"]

    # ---------- 1. Популярные фильмы ----------
    def popular_movies(self, n: int = 10) -> pd.DataFrame:
        return self.popularity.head(n)[
            ["title", "avg_rating", "count_rating", "w_score"]
        ].reset_index(drop=True)

    # ---------- 2. Похожие фильмы (item-based KNN) ----------
    def similar_movies(self, movie_title: str, n: int = 10) -> pd.DataFrame:
        if movie_title not in self.movie_pivot.index:
            return pd.DataFrame(columns=["title", "Similarity"])
        idx = self.movie_pivot.index.get_loc(movie_title)
        vec = self.movie_sparse[idx]
        distances, indices = self.movie_knn.kneighbors(vec, n_neighbors=n + 1)
        distances = distances.flatten()[1:]
        indices = indices.flatten()[1:]
        return pd.DataFrame({
            "title": self.movie_pivot.index[indices],
            "Similarity": np.round(1 - distances, 4),
        }).reset_index(drop=True)

    # ---------- 3. Персональные рекомендации (user-based KNN) ----------
    def recommend_for_user(self, user_id: int, n: int = 10) -> pd.Series:
        if user_id not in self.users_pivot.index:
            return pd.Series(dtype=float)
        idx = self.users_pivot.index.get_loc(user_id)
        vec = self.users_sparse[idx]
        distances, indices = self.user_knn.kneighbors(vec, n_neighbors=11)
        distances = distances.flatten()
        indices = indices.flatten()
        similar_users = indices[1:]
        similarities = 1 - distances[1:]
        neigh_ratings = self.users_pivot.iloc[similar_users]
        weighted = neigh_ratings.T.dot(similarities) / similarities.sum()
        weighted = pd.Series(weighted, index=self.users_pivot.columns)
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
        scores = pd.Series(dtype=float)
        if alpha > 0:
            pop = self.popularity.set_index("title")["w_score"]
            pop_norm = (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)
            scores = scores.add(pop_norm * alpha, fill_value=0)
        if liked_movie and beta > 0:
            sim = self.similar_movies(liked_movie, n=200)
            if not sim.empty:
                sim_scores = sim.set_index("title")["Similarity"]
                scores = scores.add(sim_scores * beta, fill_value=0)
        if user_id is not None and gamma > 0:
            recs = self.recommend_for_user(user_id, n=200)
            if len(recs) > 0:
                norm = (recs - recs.min()) / (recs.max() - recs.min() + 1e-9)
                scores = scores.add(norm * gamma, fill_value=0)
        if user_id is not None and user_id in self.users_pivot.index:
            seen = self.users_pivot.loc[user_id]
            seen = seen[seen > 0].index
            scores = scores.drop(seen, errors="ignore")
        return scores.sort_values(ascending=False).head(n)

    # ---------- 5. Content-based рекомендации (TF-IDF) ----------
    def content_based_recommend(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        """Рекомендует фильмы, похожие на movie_id по жанрам и названию."""
        if movie_id not in self.movies_content["movieId"].values:
            return pd.DataFrame(columns=["movieId", "title", "genres", "Similarity"])
        idx = self.movies_content[self.movies_content["movieId"] == movie_id].index[0]
        movie_vec = self.content_matrix[idx]
        sim_scores = cosine_similarity(movie_vec, self.content_matrix).flatten()
        top_indices = sim_scores.argsort()[-(n + 1):][::-1]
        top_indices = [i for i in top_indices if i != idx][:n]
        result = self.movies_content.iloc[top_indices][["movieId", "title", "genres"]].copy()
        result["Similarity"] = np.round(sim_scores[top_indices], 4)
        return result.reset_index(drop=True)

    # ---------- Свойства ----------
    @property
    def known_users(self) -> list:
        return self.users_pivot.index.tolist()

    @property
    def known_movies(self) -> list:
        return sorted(self.movie_pivot.index.tolist())

    @property
    def known_movie_ids(self) -> list:
        return self.movies_content["movieId"].tolist()

    # ---------- Обратная совместимость ----------
    popular_books = popular_movies
    similar_books = similar_movies
    known_books = known_movies