# Buradan başla

1. ZIP-i **Extract All** et. İçindəki `ML_Final_Project` qovluğunu aç.
2. Bu dəfə **house_sale.csv data qovluğundadır**. Köhnə layihədən əlavə köçürməyə ehtiyac yoxdur.
3. Jupyter-də həmin qovluqdakı **FINAL_Project.ipynb** faylını aç.
4. Əvvəl istifadə etdiyin Python 3.12 / ML Final Project kernelini seç.
5. **Run All** et. `RUN_FULL_EXPERIMENT=False` qalsın: hazır nəticələri və modeli göstərəcək.

Paket köhnə Stage 1/2-nin üzərinə yazılmaq üçün update deyil, ayrıca tam layihədir.

## Əsas fayllar

- `FINAL_Project.ipynb`: yekun notebook; Azərbaycan dilində izahlar.
- `report/report.pdf`: ingilis dilində 5 səhifəlik IEEE üslubunda hesabat.
- `presentation/ML_Final_Slides.pdf`: 11 slaydlıq təqdimat.
- `presentation/ML_Final_Slides.pptx`: slaydları redaktə etmək üçün.
- `DEFENSE_GUIDE_AZ.md`: müəllimin suallarını cavablandırmaq üçün izahlar.

## Kitabxana xətası olsa

Anaconda Prompt-da istifadə etdiyin mühiti aktivləşdir, layihə qovluğuna keç və yaz:

```bash
python -m pip install -r requirements.txt
```

Sonra notebook kernelini restart et və yenidən Run All et.
Yeni mühit qurmaq əvəzinə əvvəl işlək olan mühitindən başlaya bilərsən.

## Nəticəni necə oxuyasan?

**Premium/standart sinifləndirmə:** seçilmiş öz Decision Tree modelinin test
F1-i 0.9218-dir. Xətti və qeyri-xətti öz SVM modelləri də ayrıca qiymətləndirilib.

**Qiymət proqnozu:** median nisbi səhv 8.74%-dir; qiymətlərin 77.38%-i ±20%
daxilində proqnozlaşdırılır. Amma böyük qiymət xətalarına görə RMSE 667507 AZN,
R² 0.104-dür. Buna “92% qiymət dəqiqliyi” demək düzgün deyil.

Ofis, obyekt, torpaq və qaraj final əhatəyə daxil deyil. Yeni/köhnə tikili və
həyət evi/bağ evi qalır. Premium həddi 207000 AZN-dir; əvvəlki bütün-kateqoriya
210000 AZN həddi ilə qarışdırma.

## Hamısını yenidən hesablamaq istəsən

```bash
python -m src.run_all
```

Yaxud notebookda `RUN_FULL_EXPERIMENT=True` et. Bu, bütün sınaqları yenidən
işlədir və vaxt/yaddaş tələb edir; ilk baxış üçün lazım deyil.

## Təslimdən əvvəl bizə qalan məlumat

Komanda üzvlərinin adları, onların real töhfələri və GitHub repo linki verilməyib.
Bunlar `submission_metadata.json`, `contribution_report.md`, hesabat və slayd
üzlüyünə əlavə edilməlidir. AI yardımı açıq yazılıb. Qiyməti müəllim verir;
100 bal və ya bütün evlərdə dəqiq proqnoz zəmanəti yoxdur.
