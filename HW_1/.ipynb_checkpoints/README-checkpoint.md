# ДЗ-1 ML
## В ходе выполнения задания были сделаны следующие пункты
1. Проведен базовый анализ данных (EDA):
   - Проверка пропусков и дубликатов.
   - Анализ числовых и категориальных переменных.
   - Построение визуализаций для выявления зависимостей.
2. Обработка данных:
   - Приведение признаков (`mileage`, `engine`, `max_power`, `torque`) к числовому виду.
   - Устранение аномалий в признаках полученных из переменной `torque`.
   - Устранение мультиколлинеарности через One-Hot Encoding (OHE).
3. Построение моделей c участием и без участия категориальных переменных:
   - Обучение линейной регрессии.
   - Реализация Ridge и Lasso регрессий.
   - Использование ElasticNet с оптимизацией гиперпараметров.
4. Разработка кастомных метрик:
   - Метрика "бизнес-качество" (разница предсказаний с реальными ценами не более 10%).
   - Введение метрики, которая штрафует за недопрогноз цены.
5. Эксперименты с логарифмированием признаков и введением новых категориальных признаков.
6. Созадние сервиса основанного на лучшей модели и скрипта для трансформации данных в пригодный для предсказаний вид.

## Результататы 
- Выявлено, что `selling_price` наиболее коррелирован с `max_power`, `torque`, `engine` и `year`. При этом зависимость не совсем линейная. Также судя по матрице `phik` зависимая переменная также довольно сильно коррелирует с переменной `mileage`. Однако зависимость столь нелинейная, что на матрице корреляций Пирсона корреляция составляет всего -0.1.
- При обучении моделей на вещественных признаках метрики $MSE$ и $R^2$ получаются примерно одинаковыми в случаях Lasso и линейной регрессии. Однако при добавлении L-2 регуляризации в ElasticNet метрики в значительной степени ухудшаются. В случае обучения на вещественных и категориальных признаках ситуация остается такой же. Лучше всего себя показывают Lasso и линейная регрессия, а Ridge выдает метрики похуже. В ходе экспериментов зависимая переменная `selling_price`, а также независимые `max_power`, `torque` и `engine` были преобразованны в логарифмы, что привело к более линейной зависимости и наилучшим в ходе экспериментов при использовании базовой линейной регресии результатам с $R^2 = 0.908$ на тестовых данных при использовании `np.exp` на предсказаниях, которые выводятся в логарифмах.
- В ходе тестирования построенных бизнес метрик модель с логарифмами также проявила себя лучше всего. При этом лишь примерно 30% предсказаний лучшей модели не выходят за рамки завышенных или заниженных на 10% прогнозов.
- Был реализован сервис на FAST API, котрый был основан на лучшей модели с использованием логарифмов. Он успешно справляется со своей задачей как было продемонстрированно на скриншотах. Однако модель основана на пайплайне с использованием кастомного трансформера, который был записан в файл `c_transformer.py`. Поэтому при использовании этого сервиса обязательно необходимо импортировать класс оттуда.

## Что сделать не вышло сделать
- Не удалось наладить вывод дашборда `ProfileReport` в Jupiter. Однако все работает, если использовать Colab от Google. Вероятно это связанно с какими-либо необновленными библиотеками. 
- Хотелось, но не удалось использовать деление на классы автомобилей (A, B, C, D и т.д.), так как ChatGPT даже с вводной информацией не справился с таким делением, а делать это вручную с таким количеством автомобилей просто невозможно. Поэтому для улучшения предсказаний модели было выполненно деление на премиальность бренда и страну производителя. Однако это не дало каких либо ощутимых результатов.
- Не удалось наладаить работу L-0 регуялризации с данными включающими категориальные переменные. Это так-то не вызывает проблем, так как это-то пункт находится в разделе посвященному обучению только на вещественных переменных.