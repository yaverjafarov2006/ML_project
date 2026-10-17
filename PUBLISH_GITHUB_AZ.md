# GitHub və yekun təhvil

Texniki paket hazırdır. Repozitoriya yaradılmayıb və internetə göndərilməyib;
komandanın adları və mövcud GitHub ünvanı verilməyib. Aşağıdakı addımlar
həmin məlumatlar tamamlandıqdan sonra yerinə yetirilir.

1. `submission_metadata.json` faylına real müəllif adlarını və repo URL-ni yaz.
2. `contribution_report.md` faylında hər üzvün faktiki gördüyü işi göstər.
   Keçmiş tarixli commit və ya görülməyən iş barədə iddia əlavə etmə.
3. Müəllif adlarını slaydın üzlüyünə əlavə et. Hesabatı LaTeX varsa
   `python -m src.build_report --compile` əmri ilə yenilə.
4. GitHub hesabında boş repo yarat. Sonra layihə qovluğunda terminal aç.

Git quraşdırılıbsa, öz hesabınla aşağıdakı əmrləri işlət. Dırnaq içindəki
nümunələri öz real məlumatlarınla əvəz et. Bu addımlar boş, yeni repo üçündür;
mövcud layihənin Git tarixçəsini silmə və ya yenidən yaratma.

```bash
git init
git config user.name "AD SOYAD"
git config user.email "GITHUB HESABININ EMAILI"
git add .
git status --short
```

`data/house_sale.csv` siyahıda görünməməlidir: `.gitignore` onu çıxarır.
Yoxlamadan sonra:

```bash
git commit -m "Complete residential ML project with disclosed AI assistance"
git branch -M main
git tag -a v1.0-final -m "Reviewed final course submission"
git remote add origin "OZ REPO URL-NI BURAYA YAZ"
git push -u origin main
git push origin v1.0-final
```

Repo-nu klonlayan şəxs kursun CSV faylını ayrıca `data` qovluğuna qoymalıdır.
ZIP-də CSV sənin rahatlığın üçün var; GitHub-a brauzerdən toplu fayl yükləyərkən
CSV-ni əlavə etmə.

Kursa göndərilənlər: **repo linki və v1.0-final tag, report/report.pdf,
presentation/ML_Final_Slides.pdf**. ZIP şəxsi işlək nüsxəndir.
