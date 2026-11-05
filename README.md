# Квантовый рубеж

Самохвалов В. В., Аналитический центр «Брейншторм», 2026.

Пять вычислительных ядер, данные SPARC, отчёты, графики и авторские материалы.

| Версия | Код | Отчёт | Медианная ошибка |
|---|---|---|---|
| v1.0 | [TGS.Quantum_Frontier.py](TGS.Quantum_Frontier.py) | [TGS_QF_Report.txt](reports/TGS_QF_Report.txt) | 12.90% sMAPE |
| v2.0 | [TGS.QF_v2.0.py](TGS.QF_v2.0.py) | [TGS_v2.0_Epicycles_Report.txt](reports/TGS_v2.0_Epicycles_Report.txt) | 12.18% sMAPE |
| v3.0 | [TGS.QF_v3.0.py](TGS.QF_v3.0.py) | [TGS_v3.0_Hybrid_EFE_Report.txt](reports/TGS_v3.0_Hybrid_EFE_Report.txt) | 11.66% sMAPE |
| v4.0 | [TGS.QF_4.0.py](TGS.QF_4.0.py) | [TGS_v4.0_CosmoOS_Report.txt](reports/TGS_v4.0_CosmoOS_Report.txt) | 11.25% sMAPE |
| QF.cheater | [TGS.QF.cheater.py](TGS.QF.cheater.py) | [TGS_v17_2_Loss_Balance_Report.txt](reports/TGS_v17_2_Loss_Balance_Report.txt) | 11.49% sMAPE¹ |

У v1/v2/v3/v4 сначала вычисляется медиана точечных sMAPE каждой галактики, затем медиана по галактикам.

Для v1.0 среднее погалактических sMAPE — 17.86%, медианный Bias по всем точкам — −2.38 км/с.

¹ Итоговые показатели новой авторской редакции QF.cheater: медиана sMAPE 11.49%, среднее 16.93%, медианный Bias +3.97 км/с. Новый файл и отчёт ожидают замены: пока в репозитории сохранён предыдущий QF.cheater, считающий MAPE; эти sMAPE не являются результатом запуска старого файла.

## Запуск

Нужен Python 3.11+. Скачайте весь репозиторий, сохранив SPARC и Table1.mrt рядом со скриптами.

```bash
python -m venv .venv
```

Linux/macOS: `source .venv/bin/activate`. Windows PowerShell: `.venv\Scripts\Activate.ps1`.

```bash
python -m pip install -r requirements.txt
python TGS.Quantum_Frontier.py
python TGS.QF_v2.0.py
python TGS.QF_v3.0.py
python TGS.QF_4.0.py
python TGS.QF.cheater.py
```

Команды независимы. Отчёт и папка графиков сохраняются на рабочий стол; если он не найден — рядом со скриптом. Переменная QF_OUTPUT_DIR позволяет задать иной выходной каталог. Повторный запуск перезаписывает результаты.

## Статья и приложения

[Статья](article/article.md) · [Статья Word](article/article.docx)

- [P_vs_NP_Complexity_TGS](https://raw.githubusercontent.com/LegionTGS/Quantum-Frontier/main/appendices/P_vs_NP_Complexity_TGS.pdf)
- [Приложение 12.2 Лагранжиан теории](https://raw.githubusercontent.com/LegionTGS/Quantum-Frontier/main/appendices/%D0%9F%D1%80%D0%B8%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5%2012.2%20%D0%9B%D0%B0%D0%B3%D1%80%D0%B0%D0%BD%D0%B6%D0%B8%D0%B0%D0%BD%20%D1%82%D0%B5%D0%BE%D1%80%D0%B8%D0%B8.pdf)
- [Приложение 13 Геометрия](https://raw.githubusercontent.com/LegionTGS/Quantum-Frontier/main/appendices/%D0%9F%D1%80%D0%B8%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5%2013%20%D0%93%D0%B5%D0%BE%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D1%8F.pdf)
- [Приложение 13.1 ТГС. Геометрия вывернутого Невода](https://raw.githubusercontent.com/LegionTGS/Quantum-Frontier/main/appendices/%D0%9F%D1%80%D0%B8%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5%2013.1%20%D0%A2%D0%93%D0%A1.%20%D0%93%D0%B5%D0%BE%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D1%8F%20%D0%B2%D1%8B%D0%B2%D0%B5%D1%80%D0%BD%D1%83%D1%82%D0%BE%D0%B3%D0%BE%20%D0%9D%D0%B5%D0%B2%D0%BE%D0%B4%D0%B0.pdf)
- [Сказание Матрёшка](https://raw.githubusercontent.com/LegionTGS/Quantum-Frontier/main/appendices/%D0%A1%D0%BA%D0%B0%D0%B7%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%9C%D0%B0%D1%82%D1%80%D1%91%D1%88%D0%BA%D0%B0.pdf)
- [Эпициклы Птолемея](https://raw.githubusercontent.com/LegionTGS/Quantum-Frontier/main/appendices/%D0%AD%D0%BF%D0%B8%D1%86%D0%B8%D0%BA%D0%BB%D1%8B%20%D0%9F%D1%82%D0%BE%D0%BB%D0%B5%D0%BC%D0%B5%D1%8F.pdf)

- [Приложение 4.3.1 Три КЭД-барьера](https://raw.githubusercontent.com/LegionTGS/Quantum-Frontier/main/appendices/%D0%9F%D1%80%D0%B8%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5%204.3.1%20%D0%A2%D1%80%D0%B8%20%D0%9A%D0%AD%D0%94-%D0%B1%D0%B0%D1%80%D1%8C%D0%B5%D1%80%D0%B0.pdf)
- [Чёрная дыра в ТГС](https://raw.githubusercontent.com/LegionTGS/Quantum-Frontier/main/appendices/%D0%A7%D1%91%D1%80%D0%BD%D0%B0%D1%8F%20%D0%B4%D1%8B%D1%80%D0%B0%20%D0%B2%20%D0%A2%D0%93%D0%A1.pdf)
- [Приложение 4.3 Геометрический лагранжиан ТГС](https://raw.githubusercontent.com/LegionTGS/Quantum-Frontier/main/appendices/%D0%9F%D1%80%D0%B8%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5%204.3%20%D0%93%D0%B5%D0%BE%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B9%20%D0%BB%D0%B0%D0%B3%D1%80%D0%B0%D0%BD%D0%B6%D0%B8%D0%B0%D0%BD%20%D0%A2%D0%93%D0%A1.pdf)

## Условия участия в штурме

Предлагайте изменения через Pull Request с описанием изменений и результатами расчёта. Все вклады в код принимаются на условиях GNU GPL версии 3 или любой более поздней версии; полный текст в [LICENSE](LICENSE). Маркер: DOMASHKA_04.11.2026.

## Данные

SPARC: Lelli F., McGaugh S. S., Schombert J. M. (2016), The Astronomical Journal, 152, 157. [Каталог](https://astroweb.case.edu/SPARC/).
