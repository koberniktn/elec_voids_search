"""Объединение сырых кандидатов двух методов без удаления и смещения точек."""

import pandas as pd


def combine_candidates(grid, voronoi):
    """Собираем кандидатов сетки и Вороного в одну таблицу.

    Сначала идут строки сетки, затем Вороного. Сохраняем метод и исходный ID,
    а общему списку присваиваем новые уникальные ID. Все точки остаются:
    кластеризация и симметрийный отсев выполняются следующими этапами."""

    combined = pd.concat(
        [
            grid.rename(columns={"candidate_id": "source_candidate_id"}).assign(
                source_method="grid"
            ),
            voronoi.rename(columns={"candidate_id": "source_candidate_id"}).assign(
                source_method="voronoi"
            ),
        ],
        ignore_index=True,
    )

    combined.insert(0, "candidate_id", range(len(combined)))

    return combined
