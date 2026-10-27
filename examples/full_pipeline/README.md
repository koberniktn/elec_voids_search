# Полный скрининг тестовых структур

Пример включает **7 структур**: LATP, LLZO + Ga, LGPS, Li₃PO₄, LiCl, LiCl с вакансией Li и Li₃PO₄ с вакансией Li.

- **Входные структуры:** [inputs/](inputs/).
- **Запуск расчёта:** [full_pipeline.ipynb](full_pipeline.ipynb). В ноутбуке задаются ручные степени окисления и параметры поиска.
- **Итоговые количества по этапам:** [voids_gallery/summary.csv](voids_gallery/summary.csv).
- **Количества на формульную единицу:** [voids_gallery/summary_norm.csv](voids_gallery/summary_norm.csv).
- **Шаги координационного и Li–Li-фильтров:** [voids_gallery/branch_steps.csv](voids_gallery/branch_steps.csv).
- **Координаты кандидатов:** `voids_gallery/<имя_структуры>/candidates.csv`.
- **Визуализации:** `voids_gallery/<имя_структуры>/cif/` — каркас и X-маркеры кандидатов после каждого этапа, для просмотра в VESTA.

[Сохранённые результаты](voids_gallery/) относятся к контрольному прогону. Повторный запуск ноутбука создаёт отдельную папку с временной меткой.

Назначение пакета и последовательность этапов — в [основном README](../../README.md); настройки — в [PARAMETERS.md](../../elec_voids_search/docs/PARAMETERS.md).
