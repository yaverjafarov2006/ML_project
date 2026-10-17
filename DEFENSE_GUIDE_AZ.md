# Müdafiəyə hazırlıq

## Layihəni 40 saniyədə izah et

Layihədə bina.az elanlarından iki məsələ qurmuşuq: evin satış üçün elan edilən
qiymətini proqnozlaşdırmaq və onu premium/standart kimi ayırmaq. Əsas alqoritmlər
NumPy ilə yazılmış Decision Tree və Pegasos SVM-dir. SVM üçün Random Fourier
Features ilə qeyri-xətti variant da hazırlanıb. Modelləri validation-da seçib,
ayrılmış testdə və ayrıca 5-fold qrup yoxlamasında qiymətləndirmişik. Tipik qiymət
xətası qənaətbəxşdir, amma çox böyük və şübhəli qiymətlər RMSE-ni kəskin artırır.

## Sual veriləndə

**Niyə Random Forest əsas model deyil?** Tapşırıq öz Decision Tree və öz SVM
implementasiyamızı tələb edir. Əvvəlki Random Forest yalnız əlavə müqayisə idi.

**Niyə ofis, obyekt, torpaq və qaraj çıxarılıb?** Məqsəd yaşayış əmlakıdır.
Beləliklə əhatəni dəqiq müəyyən etmişik. Nəticə bütün əmlak növlərinə aid deyil.
Həyət evi/bağ evi saxlanılıb; qiymət üçün mənzillərdən ayrıca ağac öyrənir.

**Train, validation, test nə üçündür?** Train parametrləri öyrədir, validation
modeli və hiperparametrləri seçir, test sabit seçimdən sonra yekun ölçü verir.
Eyni əmlaka oxşar elan qrupları bölgülər arasında ayrıdır.

**Niyə hədd 207000-dir?** Yaşayış train qiymətlərinin medianıdır. Bundan böyük
qiymət premiumdur. Əvvəlki 210000 bütün kateqoriyaların train medianı idi.

**Data leakage necə əngəllənib?** unit_price, total_price, qiymətdən yaranan
xüsusiyyətlər və identifikatorlar girişə daxil deyil. İmputasiya, encoding,
scaling və məkan mərkəzləri yalnız train-də öyrənilir; hər CV fold-da yenilənir.

**Öz ağacımız necə işləyir?** Hər xüsusiyyəti sıralayıb fərqli qonşu dəyərlər
arasındakı hədləri yoxlayır. Ən böyük impurity azalmasını seçir. Regression-da
MSE, classification-da Gini/entropy var. Yarpaqda orta və ya çoxluq sinfi verilir.

**Niyə log(price)?** Qiymət paylanması ağır quyruqludur. Log transformasiyası
böyük qiymətlərin train təsirini yumşaldır. Geri eksponent götürülür. Yarpaq
qiyməti geometrik ortadır; çox bahalı evləri aşağı qiymətləndirmə riski qalır.

**Pegasos nə edir?** L2 regularizasiya və hinge loss cəmini mini-batch subgradient
addımları ilə azaldır. Addım eta=1/(lambda*t), batch=256-dır. Bias ayrıca və
regularizasiya olunmadan yenilənir. Çəki proyeksiya edilir, son yarı iterasiyalar
ortalaması final çəkini verir. Objective hər addımda monoton enmək məcburiyyətində deyil.

**RFF niyə var?** RBF kernelini təsadüfi kosinus xüsusiyyətləri ilə təxmini
qurur; sonra eyni öz xətti SVM işləyir. n×n kernel matrisindən qaçırıq, amma
2048 xüsusiyyətli sıx matris yenə yaddaş tələb edir. Bu, tələb olunan nonlinear variantdır.

**Margin və support vectors?** Margin eni 2/||w||-dir. y*score<=1 olan train
sətirləri margin daxilində və ya sərhədindədir. Bunları outputs/svm_margin_violators.csv
göstərir. Bu təxmini primal həllin siyahısıdır, dual optimal alpha siyahısı deyil.

**92% qiymət dəqiqliyimiz var?** Xeyr. F1=0.9218 premium sinfini ayırma nəticəsidir.
Qiymət üçün median nisbi səhv 8.74%, MAE 56568 AZN, RMSE 667507 AZN,
R² 0.104-dür. Ev proqnozlarının 77.4%-i həqiqi elan qiymətinə ±20% daxilindədir.

**R² niyə zəifdir?** Testdə çox böyük qiymətlər və böyük səhvlər var; ən böyük
elan qiyməti 60 milyon AZN-dir. Ən böyük 1% kvadrat səhv ümumi kvadrat səhvin
98.8%-ni təşkil edir. Bu elanların həqiqi qiyməti müstəqil təsdiqlənməyib.
Nəticəni yaxşı göstərmək üçün onları testdən silməmişik.

**CV RMSE niyə milyonlarla çıxır?** Train-də 600 milyon AZN-lik şübhəli qeyd
var. CV-nin bir fold-unda qiymətləndiriləndə çox böyük xəta yaradır. MedAPE
fold-larda təxminən 9.22±0.05%-dir. Bu, məlumat keyfiyyəti riskini göstərir.

**Koordinat böyüyəndə qiymət artır?** Belə ümumi və səbəb-nəticə iddiası yoxdur.
Ağac məkan bölgülərini öyrənir. Əlavə məsafələr və train mərkəzləri coğrafi
yaxınlığı ifadə edir; mərkəzlər rəsmi rayon və metro stansiyası deyil.

**sklearn ilə niyə eyni rəqəm deyil?** Ağacda bərabər qazanc seçimi və float
hədləri fərqlənə bilər. SVM-də objective uyğun olsa da SGD və Pegasos-un addım,
batch, proyeksiya və averaging qaydaları fərqlidir.

**AI istifadə olunub?** Bəli, substantial ChatGPT/Codex yardımı açıq göstərilib.
Öz adından etmədiyin işi və ya töhfə faizini yazma. Kodun məntiqini müdafiədən
əvvəl başa düş və real komanda iş bölgüsünü tamamla.

## 2 dəqiqəlik demo

1. FINAL_Project.ipynb aç; hazır nəticələrin olduğu xanaları göstər.
2. Seçilmiş modellər və test cədvəlini izah et; F1 ilə qiymət səhvini qarışdırma.
3. Objective və CV qrafiklərini göstər; böyük qiymət problemini izah et.
4. Nümunə evin sahə/otaq/mərtəbə dəyərlərini dəyişib predict_listing işlətdiyini göstər.
5. Mənbə kodunda tree split və Pegasos update hissəsini aç.
