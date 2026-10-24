# elec_voids_search

Пакет запускается из notebook через `run_search`, без консольных аргументов.

```python
from elec_voids_search import SearchConfig
from elec_voids_search.run_search import run_search

settings = SearchConfig(
    probe_element="Li", probe_oxidation_state=1,
    geometry_oxidation_states={"Li": 1, "Cl": -1},
)
result = run_search(input_cif, new_output_folder, config=settings)
final_coordinates = result["geometry"]
```

Последовательность: независимые сырые кандидаты сетки и Вороного → общий список и объединение близких кандидатов в кластеры → отсев по симметрии → отсев по геометрии. Координаты не перемещаются и не усредняются. После геометрии никаких отборов нет.

По умолчанию используются связные компоненты по периодическому расстоянию < 0.75 Å, как в исходном механизме кластеризации. В каждом кластере выбирается существующая точка с максимальным расстоянием до ближайшего атомного центра. Из симметрийной группы выбирается первая точка, как в исходной функции.

Геометрия вычисляет радиусы каждого сайта каркаса и пробного иона с учётом его окружения. Ручные степени окисления обязательны. Политика отсутствующих Shannon-значений: [RADIUS_POLICY.md](RADIUS_POLICY.md). Настройки: [PARAMETERS.md](PARAMETERS.md), этапы и таблицы: [CANDIDATE_MERGING.md](CANDIDATE_MERGING.md).

Сохранены исходные модули shannon_radii.py и site_atomic_data.py побайтно и исходное форматирование блоков site_hard_radius/point_geometry. Сетка происходит из test_easy.ipynb, Вороной и кластеризация — из single_structure_voronoi.ipynb. Исходники не перезаписываются; происхождение зафиксировано в *_SOURCE.json и COPIED_FUNCTIONS.json. periodic_groups.py вычисляет те же периодические расстояния пакетами; оригинальные функции остаются в symmetry.py и voronoi_merging.py. final_selection.py — исторический модуль, текущий run_search его не вызывает.

Результаты CSV/JSON: grid, voronoi, combined, merged, nonequivalent, geometry_evaluated, geometry, geometry_rejected. geometry_indeterminate — пустая таблица совместимости; geometry_retained совпадает с geometry. Дополнительно сохраняются радиусы, geometry_candidate_audit, occupied_probe_controls, состав кластеров, summary и metadata. CIF по умолчанию отключены.

[run_search_tests.ipynb](run_search_tests.ipynb) подготовлен с проверками неизменности координат и невозрастания числа кандидатов. Выборка оставлена ограниченной четырьмя ранее неуспешными структурами; ALL_INPUTS содержит все семь входов. Notebook и тесты после этих правок не выполнялись. Выполненные notebook и ранее сохранённые отчёты относятся к историческим версиям пайплайна.
