"""
Streamlit-приложение: система рекомендаций фильмов.
Запуск: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from recommender import MovieRecommender


# ------------------- Настройка страницы -------------------
st.set_page_config(
    page_title="Movie Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------- Стили -------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4b6cb7, #182848);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle { color: #888; font-size: 1.1rem; margin-bottom: 2rem; }

    /* Карточка фильма */
    .movie-card {
        background: #f8f9fa;
        border-left: 4px solid #4b6cb7;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        transition: transform 0.15s ease;

        /* Принудительно задаём тёмный цвет текста для светлой карточки */
        color: #1a1a1a !important; 
    }
    .movie-card b {
        color: #1a1a1a !important;
        font-weight: 600;
    }
    .movie-card:hover {
        transform: translateX(4px);
        background: #eef1f7;
    }

    /* Кружок с номером */
    .movie-rank {
        display: inline-block;
        background: #4b6cb7;
        color: #ffffff !important;
        border-radius: 50%;
        width: 28px; height: 28px;
        line-height: 28px;
        text-align: center;
        font-weight: 700;
        margin-right: 0.6rem;
    }

    /* Цвет скора */
    .score-text {
        color: #4b6cb7 !important;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ------------------- Загрузка модели (кэшируем) -------------------
@st.cache_resource(show_spinner="Загружаем модель рекомендаций...")
def load_model():
    return MovieRecommender()


try:
    rec = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


# ------------------- Заголовок -------------------
st.markdown('<div class="main-header">Movie Recommender</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Персональные рекомендации фильмов на основе '
    'коллаборативной фильтрации и популярности</div>',
    unsafe_allow_html=True,
)


# ------------------- Сайдбар -------------------
st.sidebar.header("Настройки")

mode = st.sidebar.radio(
    "Режим рекомендаций",
    options=[
        "Популярные фильмы",
        "Похожие фильмы",
        "Персональные (по User-ID)",
        "Гибридные рекомендации",
        "Справка",
    ],
    index=0,
)

n_recs = st.sidebar.slider("Количество рекомендаций", 5, 30, 10)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"В базе: **{len(rec.known_movies):,}** фильмов, "
    f"**{len(rec.known_users):,}** пользователей"
)


# ------------------- Хелпер для отображения карточек -------------------
def render_movie_cards(df: pd.DataFrame, score_col: str, score_label: str, percent: bool = False):
    """Отрисовывает красивый список фильмов в виде карточек."""
    for i, row in df.iterrows():
        score = row[score_col]
        score_str = f"{score:.2%}" if percent else f"{score:.3f}"

        # Оборачиваем текст скора в <span class="score-text">
        st.markdown(
            f'<div class="movie-card">'
            f'<span class="movie-rank">{i + 1}</span>'
            f'<b>{row["title"]}</b> '
            f'<span class="score-text">— {score_label} {score_str}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
# ------------------- Основной блок -------------------

# ----- 1. Популярные фильмы -----
if mode == "Популярные фильмы":
    st.subheader("Топ популярных фильмов (взвешенный рейтинг)")
    st.caption(
        "Рейтинг IMDb-стиля: учитывает и среднюю оценку, "
        "и количество оценок, чтобы фильмы с 1–2 отзывами "
        "не обгоняли проверенную классику."
    )

    top = rec.popular_movies(n=n_recs)

    # Таблица
    st.dataframe(
        top.rename(columns={
            "title": "Название",
            "avg_rating": "Средняя оценка",
            "count_rating": "Кол-во оценок",
            "w_score": "Взвешенный рейтинг",
        }).style.format({
            "Средняя оценка": "{:.2f}",
            "Взвешенный рейтинг": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # График
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=top, y="title", x="w_score",
        palette="viridis", ax=ax,
    )
    ax.set_xlabel("Взвешенный рейтинг (w_score)")
    ax.set_ylabel("")
    ax.set_title(f"Топ-{n_recs} фильмов")
    plt.tight_layout()
    st.pyplot(fig)


# ----- 2. Похожие фильмы -----
elif mode == "Похожие фильмы":
    st.subheader("Фильмы, похожие на выбранный")
    st.caption(
        "Ищем фильмы, которые смотрели и оценивали похожие "
        "пользователи, используя косинусное сходство."
    )

    movie = st.selectbox(
        "Выберите фильм:",
        options=rec.known_movies,
        index=0,
    )

    if st.button("Найти похожие", type="primary"):
        with st.spinner("Ищем похожие фильмы..."):
            sim = rec.similar_movies(movie, n=n_recs)

        if sim.empty:
            st.warning("Для этого фильма не найдено похожих.")
        else:
            st.success(f"Топ-{len(sim)} фильмов, похожих на «{movie}»:")

            render_movie_cards(sim, "Similarity", "схожесть", percent=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=sim, y="title", x="Similarity",
                palette="magma", ax=ax,
            )
            ax.set_xlabel("Косинусная схожесть")
            ax.set_ylabel("")
            ax.set_title(f"Похожие фильмы на «{movie}»")
            plt.tight_layout()
            st.pyplot(fig)


# ----- 3. Персональные рекомендации -----
elif mode == "Персональные (по User-ID)":
    st.subheader("Персональные рекомендации по User-ID")
    st.caption(
        "Найдём пользователей с похожими вкусами и порекомендуем "
        "фильмы, которые им понравились, а вы ещё не смотрели."
    )

    user_id = st.number_input(
        "User-ID:",
        min_value=int(min(rec.known_users)),
        max_value=int(max(rec.known_users)),
        value=int(rec.known_users[0]),
        step=1,
    )
    user_id = int(user_id)

    # Сколько фильмов пользователь уже оценил
    if user_id in rec.users_pivot.index:
        seen_count = int((rec.users_pivot.loc[user_id] > 0).sum())
        st.info(f"Пользователь **{user_id}** уже оценил **{seen_count}** фильмов.")
    else:
        st.warning(
            f"Пользователь **{user_id}** отсутствует в обучающей выборке. "
            "Возможно, он был отфильтрован из-за малого числа оценок."
        )

    if st.button("Показать рекомендации", type="primary"):
        with st.spinner("Подбираем фильмы..."):
            recs = rec.recommend_for_user(user_id, n=n_recs)

        if len(recs) == 0:
            st.warning(
                "Не удалось построить рекомендации. "
                "Возможно, пользователь отсутствует в выборке "
                "или у него слишком мало соседей."
            )
        else:
            st.success(f"Топ-{len(recs)} рекомендаций для пользователя {user_id}:")

            df_recs = recs.reset_index()
            df_recs.columns = ["title", "score"]

            render_movie_cards(df_recs, "score", "скор")

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=df_recs, y="title", x="score",
                palette="coolwarm", ax=ax,
            )
            ax.set_xlabel("Предсказанный скор")
            ax.set_ylabel("")
            ax.set_title(f"Рекомендации для User-ID = {user_id}")
            plt.tight_layout()
            st.pyplot(fig)

elif mode == "Справка":
    st.subheader("Справка по режимам и параметрам")
    st.caption(
        "Ниже — описание всех режимов рекомендаций, их параметров "
        "и форматов данных."
    )

    st.markdown("---")

    # ---------------- Режимы ----------------
    st.markdown("## Режимы рекомендаций")
    st.markdown(
        """
        Приложение поддерживает **4 режима рекомендаций** и **справочный раздел**.
        Режим выбирается в левом меню. Для каждого режима доступны свои
        параметры, влияющие на результат.
        """
    )

    with st.expander("1. Популярные фильмы", expanded=False):
        st.markdown(
            """
            **Что делает:** показывает топ фильмов по *взвешенному рейтингу*
            (IMDb Bayesian Average). Эта метрика учитывает и среднюю оценку,
            и количество оценок, чтобы фильмы с 1–2 отзывами не обгоняли
            проверенную классику.

            **Формула:**

            $$
            w = \\frac{v \\cdot R + m \\cdot c}{v + m}
            $$

            где:
            - $v$ — количество оценок у фильма;
            - $R$ — средняя оценка фильма;
            - $m$ — порог доверия (90-й процентиль числа оценок);
            - $c$ — средний рейтинг по всем фильмам.

            **Параметры:**
            - **Количество рекомендаций (N)** — сколько фильмов показать
              (5–30, по умолчанию 10).

            **Результат:** таблица с колонками `Название`, `Средняя оценка`,
            `Кол-во оценок`, `Взвешенный рейтинг` + столбчатая диаграмма.

            **API-эндпоинт:** `GET /popular?n={N}`
            """
        )

    with st.expander("2. Похожие фильмы", expanded=False):
        st.markdown(
            """
            **Что делает:** находит фильмы, наиболее похожие на выбранный,
            используя косинусное сходство между векторами оценок пользователей
            (item-based collaborative filtering + KNN).

            **Как работает:**
            1. Для выбранного фильма берётся вектор его оценок всеми пользователями.
            2. Модель `NearestNeighbors(metric='cosine')` находит N ближайших
               векторов других фильмов.
            3. Схожесть = `1 - косинусное_расстояние` (от 0 до 1).

            **Параметры:**
            - **Выберите фильм** — фильм-«источник» из базы (~9 978 названий).
            - **Количество рекомендаций (N)** — сколько похожих показать.

            **Результат:** список карточек с процентом схожести + диаграмма.

            **API-эндпоинт:** `GET /similar?title={title}&n={N}`
            """
        )

    with st.expander("3. Персональные рекомендации", expanded=False):
        st.markdown(
            """
            **Что делает:** рекомендует фильмы пользователю на основе оценок
            *похожих пользователей* (user-based collaborative filtering + KNN).

            **Как работает:**
            1. Для выбранного `User-ID` берётся вектор его оценок.
            2. Модель `NearestNeighbors` находит K ближайших пользователей.
            3. Для каждого фильма считается взвешенное среднее оценок
               соседей: `score = Σ(sim_i · rating_i) / Σ sim_i`.
            4. Фильмы, которые пользователь уже оценил, исключаются.
            5. Возвращается топ-N по скору.

            **Параметры:**
            - **User-ID** — ID пользователя из обучающей выборки
              (только пользователи с > 200 оценок; всего ~132 человека).
            - **Количество рекомендаций (N)**.

            **Результат:** карточки с предсказанным скором + диаграмма.

            **API-эндпоинт:** `GET /recommendations/{user_id}?n={N}`
            """
        )

    with st.expander("4. Гибридные рекомендации", expanded=False):
        st.markdown(
            """
            **Что делает:** объединяет три сигнала — популярность, схожесть
            с любимым фильмом и персональные предпочтения.

            **Как работает:**

            $$
            \\text{score}(movie) = \\alpha \\cdot P(movie) +
            \\beta \\cdot S(movie) + \\gamma \\cdot U(movie)
            $$

            где:
            - $P$ — нормированный `w_score` (популярность);
            - $S$ — косинусная схожесть с `liked_movie`;
            - $U$ — нормированный скор от user-based KNN.

            Итоговый скор нормируется делением на сумму активных весов
            $\\alpha + \\beta + \\gamma$, поэтому всегда лежит в диапазоне [0, 1].

            **Параметры:**
            - **Использовать User-ID** — включить персонализацию.
            - **User-ID** — ID пользователя.
            - **Использовать любимый фильм** — включить сигнал схожести.
            - **Любимый фильм** — фильм-опора.
            - **α (Популярность)** — вес популярности (0.0–1.0).
            - **β (Схожесть)** — вес схожести с любимым фильмом (0.0–1.0).
            - **γ (Персонализация)** — вес user-based сигнала (0.0–1.0).
            - **Количество рекомендаций (N)**.

            ⚠️ Хотя бы один из сигналов (α, β или γ) должен быть > 0.

            **API-эндпоинт:**
            `GET /hybrid?user_id={id}&movie={title}&n={N}&alpha={a}&beta={b}&gamma={g}`
            """
        )

    st.markdown("---")

    # ---------------- Параметры ----------------
    st.markdown("## ️ Общие параметры")
    st.markdown(
        """
        | Параметр | Где задаётся | Диапазон | По умолчанию | Описание |
        |----------|-------------|----------|--------------|----------|
        | **Режим** | Сайдбар, radio | 5 вариантов | Популярные | Тип рекомендаций |
        | **N** | Сайдбар, slider | 5–30 | 10 | Количество рекомендаций |
        | **User-ID** | Вкладки «Персональные» / «Гибридные» | ≥ 1 | min(users) | ID пользователя |
        | **Фильм** | Вкладки «Похожие» / «Гибридные» | из списка | первый | Фильм-источник |
        | **α, β, γ** | Вкладка «Гибридные» | 0.0–1.0, шаг 0.05 | 0.4 / 0.3 / 0.3 | Веса сигналов |
        """
    )

    st.markdown("---")

    # ---------------- API ----------------
    st.markdown("## REST API")
    st.markdown(
        """
        Приложение работает через отдельный REST API (`api.py`, FastAPI).
        Полная интерактивная документация — по адресу
        [http://localhost:8000/docs](http://localhost:8000/docs).

        Запуск API:
        ```bash
        python -m uvicorn api:app --reload --port 8000
    """
    )


# ----- 4. Гибридные рекомендации -----
else:
    st.subheader("Гибридные рекомендации")
    st.caption(
        "Объединяем три сигнала: популярность, схожесть с любимым фильмом "
        "и персональную коллаборативную фильтрацию."
    )

    col1, col2 = st.columns(2)

    with col1:
        use_user = st.checkbox("Использовать User-ID", value=True)
        if use_user:
            user_id = int(st.number_input(
                "User-ID:",
                min_value=int(min(rec.known_users)),
                max_value=int(max(rec.known_users)),
                value=int(rec.known_users[0]),
            ))
        else:
            user_id = None

    with col2:
        use_movie = st.checkbox("Использовать любимый фильм", value=True)
        if use_movie:
            liked_movie = st.selectbox(
                "Любимый фильм:",
                options=rec.known_movies,
                index=0,
            )
        else:
            liked_movie = None

    st.markdown("**Веса сигналов:** (сумма может быть ≠ 1 — скоры нормируются внутри)")
    c1, c2, c3 = st.columns(3)
    alpha = c1.slider("Популярность (α)", 0.0, 1.0, 0.4, 0.05)
    beta = c2.slider("Схожесть (β)", 0.0, 1.0, 0.3, 0.05)
    gamma = c3.slider("Персонализация (γ)", 0.0, 1.0, 0.3, 0.05)

    if st.button("Сгенерировать", type="primary"):
        with st.spinner("Считаем гибридные рекомендации..."):
            recs = rec.hybrid_recommend(
                user_id=user_id,
                liked_movie=liked_movie,
                n=n_recs,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )

        if len(recs) == 0:
            st.warning(
                "Не удалось построить рекомендации. "
                "Проверьте входные данные: хотя бы один из сигналов "
                "должен быть ненулевым (α, β или γ)."
            )
        else:
            st.success(f"Гибридные рекомендации (топ-{len(recs)}):")

            df_recs = recs.reset_index()
            df_recs.columns = ["title", "score"]

            render_movie_cards(df_recs, "score", "итоговый скор")

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=df_recs, y="title", x="score",
                palette="crest", ax=ax,
            )
            ax.set_xlabel("Итоговый скор")
            ax.set_ylabel("")
            ax.set_title(
                f"Гибридные рекомендации "
                f"(α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f})"
            )
            plt.tight_layout()
            st.pyplot(fig)


# ------------------- Футер -------------------
st.markdown("---")
st.caption(
    "Movie Recommender · Streamlit + scikit-learn · "
    "Данные: MovieLens Dataset"
)