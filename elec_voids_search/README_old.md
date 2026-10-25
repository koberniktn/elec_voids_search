# elec_voids_search

Настройки описаны в [справочнике параметров](docs/PARAMETERS.md), финальные фильтры и их последовательность — в [описании скрининга](docs/POST_GEOMETRY_PIPELINE.md).

Поиск полостей и отбор позиций пробного иона. Запуск из ноутбука, без консольных аргументов.

Сырая сетка + вершины Вороного → общий список → кластеры близких точек → симметрия → геометрия. Затем независимо применяются сравнение координации и расстояние до занятого пробного иона. Координаты кандидатов на стадиях отбора не меняются.

## Полный пайплайн

```python
from elec_voids_search.config import SearchConfig
from elec_voids_search.filtering.post_geometry_config import PostGeometryConfig
from elec_voids_search.screening_pipeline import run_screening

settings = SearchConfig(
    probe_element="Li",
    probe_oxidation_state=1,
    geometry_oxidation_states={"Li": 1, "Cl": -1},
    geometry_reference_mode="percentage",
    geometry_overlap_tolerance_percent=10.0,
)

result = run_screening(
    input_cif,
    search_config=settings,
    post_config=PostGeometryConfig(),
    output_dir=new_output_folder,
)

geometry = result["search"]["geometry"]
coordination = result["post_geometry"]["branches"]["coordination"]["passed"]
li_distance = result["post_geometry"]["branches"]["li_distance"]["passed"]
```

Для настройки поиска используются только полные имена полей `SearchConfig`. Прежние псевдонимы аргументов удалены; неизвестное имя вызывает `TypeError`.

Основным остаётся процентный геометрический фильтр с допуском 10% радиуса пробника. Калибровка по занятым сайтам включается только явным `geometry_reference_mode="occupied_sites"`.

`run_screening` последовательно выполняет полный скрининг: поиск, объединение, симметрию, геометрию и финальные ветки. По умолчанию считаются координация, Li–Li и оба порядка их последовательного применения. `output_dir` включает единый экспорт таблиц и CIF всех стадий; без него результаты остаются в памяти. `run_search` — отдельный запуск только до геометрии. Для продолжения по сохранённой геометрической выборке используется `continue_from_saved_geometry`; повторного поиска при этом нет.

Ручные степени окисления обязательны. Радиус каждого атома каркаса и пробника определяется отдельно по его окружению. Основные функции пользователя сохранены вместе с исходным форматированием и сведениями о происхождении.

## Примеры и описание

- [Полный пример с результатами и визуализациями](../examples/full_pipeline/README.md).
- [Архитектура и назначение модулей](docs/ARCHITECTURE.md).
- [Параметры поиска и геометрии](docs/PARAMETERS.md).
- [Радиусы и ручные заряды](docs/RADIUS_POLICY.md).
- [Независимые и последовательные фильтры](docs/POST_GEOMETRY_PIPELINE.md).
- [Экспериментальная геометрия по занятым сайтам](docs/OCCUPIED_GEOMETRY.md).

Актуальный пример: [полный пайплайн](../examples/full_pipeline/full_pipeline.ipynb). Отдельный эксперимент: [калибровка по занятым сайтам](../examples/full_pipeline/occupied_geometry_calibration.ipynb). Проверки функций находятся в [tests](tests). Устаревшие копии тестовых ноутбуков из папки пакета удалены.

## Короткий запуск для одного или нескольких CIF

```python
from elec_voids_search import SearchConfig, run_structures

report = run_structures(
    "Li3PO4.cif",                    # либо список путей / {имя: путь}
    oxidation_states={"Li": 1, "P": 5, "O": -2},
    search_config=SearchConfig(geometry_overlap_tolerance_percent=10.0),
    output_dir="results_Li3PO4",
)
report["summary"]                   # количества на всех этапах
report["branch_steps"]              # каждый шаг финальных маршрутов
```

Для разных составов передайте `files={"LiCl": "LiCl.cif", "Li3PO4": "Li3PO4.cif"}` и `oxidation_states={"LiCl": {"Li": 1, "Cl": -1}, "Li3PO4": {"Li": 1, "P": 5, "O": -2}}`. Имена связывают файлы с зарядами и задают подпапки результатов. Список файлов использует имена без расширения; совпадающие имена требуют явного словаря.

Общие допуски задаются один раз через `search_config=SearchConfig(...)` и `post_config=PostGeometryConfig(...)`. Необязательно перечислять неизменённые параметры. Без `post_config` обёртка запускает координацию, Li–Li и оба порядка их последовательного применения. Переданный `PostGeometryConfig` задаёт свои маршруты без дополнений. Пробный элемент и заряд наследуются из `SearchConfig`, если финальная конфигурация не передана явно.

Без `output_dir` расчёт возвращает данные в памяти. С ним сохраняются общие `summary.csv`, `branch_steps.csv`, `metadata.json`, а для каждой структуры — одна таблица кандидатов, радиусы, контрольные сайты и CIF стадий с X-маркерами. `export_cif=False` отключает только CIF. Существующая папка не перезаписывается. `quiet=True` скрывает служебную печать; предупреждения доступны в `report["run_notes"]` и метаданных. Ошибка расчёта останавливает запуск; неполные результаты не выдаются за полный отчёт.

Уже рассчитанные результаты можно сохранить отдельно, без повторного поиска:

```python
from elec_voids_search import run_screening, export_screening_results

result = run_screening("LiCl.cif", search_config=SearchConfig(
    geometry_oxidation_states={"Li": 1, "Cl": -1},
))
report = export_screening_results(result, "results_LiCl")
# Для нескольких: export_screening_results({"LiCl": result, ...}, "results")
```

`screening_tables(result)` собирает такие же таблицы без записи на диск. Эти функции используют фактически рассчитанные маршруты `run_screening`; недостающие ветки самостоятельно не рассчитываются. `reference_recovery(results, targets)` отдельно проверяет известные позиции по симметрийным образам и не участвует в отборе кандидатов.

Вспомогательные сценарии вынесены отдельно:

- `saved_screening.read_geometry_candidates` и `saved_screening.continue_from_saved_geometry` — чтение CSV и продолжение сохранённого расчёта.
- `screening_diagnostics.reference_recovery` — диагностика известных вакансий.
- `search_exports.export_search_result` — подробный экспорт отдельного поиска до геометрии.

Их перенос не меняет основной порядок фильтров. Прямые импорты используйте из указанных модулей; `reference_recovery` также доступна из корня пакета.

## Организация файлов

```text
elec_voids_search/
├── config.py                 # общая конфигурация поиска
├── screening_pipeline.py     # полный run_screening
├── screening_batch.py        # запуск для одного или нескольких файлов
├── run_search.py             # отдельный запуск до геометрии
├── candidates/               # сетка, Вороной, общий список и кластеры
├── geometry/                 # геометрический отсев, радиусы и калибровка
├── descriptors/              # первая сфера и дескрипторы Вороного
├── filtering/                # координация, Li–Li и маршруты фильтров
├── symmetry/                 # симметрия, периодические группы, занятые сайты
├── results/                  # итоговые таблицы, метаданные и CIF
├── tools/                    # продолжение расчётов и диагностика
├── research/                 # исследовательские методы вне основного пути
├── docs/                     # справочники; provenance/ — происхождение кода
└── tests/                    # тесты и отчёт регрессионной проверки
```

Основной пользовательский импорт сохраняется:

```python
from elec_voids_search import SearchConfig, PostGeometryConfig, run_screening, run_structures
```

Для прямого импорта вспомогательных функций используйте подпакеты, например `from elec_voids_search.tools.saved_screening import continue_from_saved_geometry`. Актуальные примеры и тесты используют новые пути; старые плоские модули не дублируются.
