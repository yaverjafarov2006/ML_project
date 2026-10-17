import fs from 'node:fs/promises';
import path from 'node:path';
import {Presentation,PresentationFile,FileBlob} from '@oai/artifact-tool';
import {finalizePresentation,applyPresentationChartFont} from '/root/.codex/skills/builtins/presentations/container_tools/artifact_tool_utils.mjs';
const root=process.argv[2];
if(!root)throw new Error('Pass the absolute project root');
const output=path.join(root,'presentation');
const work=path.dirname(path.dirname(root));
const build=path.join(work,'presentation_build');
const d=JSON.parse(await fs.readFile(path.join(output,'deck_data.json'),'utf8'));
const metrics=(name,subset='test')=>d.metrics.find(x=>x.model===name&&x.subset===subset);
const fmt=(v,n=0)=>Number(v).toLocaleString('en-US',{minimumFractionDigits:n,maximumFractionDigits:n});
const font='DejaVu Sans',navy='#142D45',teal='#167D72',blue='#2563A6',muted='#53667A',line='#DAE3EC';
const p=Presentation.create({slideSize:{width:1280,height:720}});
function txt(s,name,text,x,y,w,h,size=28,color=navy,bold=false){
 const a=s.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{style:'solid',fill:'none',width:0}});
 a.text=text;a.text.style={typeface:font,fontSize:size,color,bold,verticalAlignment:'top',insets:0,autoFit:'none'};return a;
}
function slide(title,notes=''){
 const s=p.slides.add();s.background.fill='#FFFFFF';
 txt(s,'headline',title,64,42,1152,104,38,navy,true);
 txt(s,'page-number',String(p.slides.items.length).padStart(2,'0'),1160,664,56,28,16,muted);
 s.speakerNotes.text=notes;
 return s;
}
function table(s,values,x,y,width,height,colWidths){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width,height,columnWidths:colWidths,values});
 t.borders.assign({style:'solid',fill:line,width:.7});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?navy:(r%2?'#F1F5F8':'#FFFFFF');
  cell.text.style={typeface:font,fontSize:r===0?21:22,color:r===0?'#FFFFFF':navy,bold:r===0,insets:10,verticalAlignment:'middle'};
 }
 return t;
}
function bars(s,cats,values,x,y,w,h,{max=1,format='0.000',color=teal,title='',axis=''}={}){
 const c=s.charts.add('bar',{position:{left:x,top:y,width:w,height:h},categories:cats,
  series:[{name:title||axis||'Value',values:values.map(v=>Number(v.toFixed(6))),fill:color}],hasLegend:false,
  barOptions:{direction:'column',grouping:'clustered',gapWidth:90},
  dataLabels:{showValue:true,position:'outEnd',numberFormatCode:format,textStyle:{typeface:font,fontSize:21,fill:navy,bold:true}},
  xAxis:{textStyle:{typeface:font,fontSize:21,fill:navy}},
  yAxis:{min:0,max,numberFormatCode:format,title:axis,textStyle:{typeface:font,fontSize:17,fill:muted},majorGridlines:{style:'solid',fill:line,width:1}},
  chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF'});
 applyPresentationChartFont(c,{fontFamily:font});return c;
}
const reg=metrics('own_tree_regression'),tree=metrics('own_tree'),rff=metrics('own_rff');
{
 const s=slide('Residential asking-price prediction','Müdafiə üçün giriş: iki ayrı tapşırıq var — qiymət proqnozu və qiymət kateqoriyasının təyini. F1 qiymət dəqiqliyi deyil. Source: outputs/final_metrics.json.');
 txt(s,'subtitle','NumPy decision trees, Pegasos and random Fourier features',64,170,1110,82,32,muted);
 txt(s,'course','AI Academy / National AI Center\nMLE-AI-201 · Final Project V3',64,298,1050,80,27);
 txt(s,'finding','Selected classifier: F1 '+tree.f1.toFixed(4),64,440,1040,52,34,teal,true);
 txt(s,'price-finding','Price model: median relative error '+reg.median_ape_pct.toFixed(2)+'%',64,506,1100,54,32,blue,true);
 txt(s,'scope','Residential listings only · historical asking prices · held-out test n = 8,724',64,600,1100,44,22,muted);
}
{
 const s=slide('A declared residential scope and disjoint property groups','100775 xam sətirdən eyni elanın köhnə nüsxələri çıxarıldı. Eyni əmlaka bənzəyən elanların qrupları train/validation/test arasında qarışmır. Premium həddi yalnız yaşayış train-dən 207000 AZN hesablanıb. Source: outputs/final_data_summary.json; src/data_prep.py.');
 table(s,[['Subset','Rows','Property groups'],...Object.entries(d.data.split).map(([k,v])=>[k[0].toUpperCase()+k.slice(1),fmt(v.n),fmt(v.groups)])],64,178,650,280,[260,180,210]);
 txt(s,'scope-heading','Included',774,176,430,44,28,teal,true);
 txt(s,'scope-text','New / old apartments\nDetached / garden houses',774,230,430,115,27);
 txt(s,'exclude-heading','Excluded',774,370,430,42,28,muted,true);
 txt(s,'exclude-text','Offices, commercial units,\nland and garages',774,422,430,108,27);
 txt(s,'leakage','Train-only encoding and scaling. Price-derived columns and owner/address identifiers never enter X.',64,544,1110,86,25);
}
{
 const s=slide('Own trees search exact split thresholds','NumPy ağacı Gini, entropy və MSE dəstəkləyir. Sıralanmış fərqli dəyərlər arasındakı hədlər yoxlanır; prefix cəmlər sürəti artırır. Qiymətdə log transformasiyası və category əsasında iki model var. Məkan mərkəzləri train-də qurulur. Source: src/decision_tree.py; src/residential_models.py.');
 txt(s,'algorithm-title','Core implementation',64,174,520,44,30,blue,true);
 txt(s,'algorithm','1. Sort candidate feature values\n2. Score adjacent thresholds\n3. Maximize impurity reduction\n4. Recurse until stopping rules\n5. Predict mean / majority at leaves',64,242,550,285,27);
 txt(s,'price-title','Selected price pipeline',692,174,520,44,30,teal,true);
 txt(s,'price-method','Log-price target\nApartment / house routing\nTrain-only geographic distances\nDepth 16 · minimum leaf 5\nExponentiate leaf predictions',692,242,520,285,27);
 txt(s,'test-note','Toy splits agree with sklearn; tests also cover zero-gain XOR splits and every stopping rule.',64,584,1110,66,23,muted);
}
{
 const s=slide('Own Pegasos learns linear and nonlinear margins','SVM məqsədi regularizasiya və hinge loss cəmidir. Bias ayrıca və regularizasiya olunmadan yenilənir. Mini-batch 256, eta=1/(lambda*t), proyeksiya və son yarı iterasiyaların ortası. RFF RBF kernelini NumPy ilə təxmini qurur. Sources: src/svm.py; outputs/svm_diagnostics.csv; Shalev-Shwartz et al. 2011 doi:10.1007/s10107-010-0420-4; Rahimi and Recht 2007.');
 txt(s,'objective','J(w, b) = λ‖w‖² / 2 + mean[max(0, 1 − y(w·x + b))]',64,176,1152,58,29);
 table(s,[['Own model','λ','Epochs','Final objective','Margin 2/‖w‖'],
  ['Linear','0.001','120',d.diagnostics[0].objective.toFixed(4),d.diagnostics[0].margin_width.toFixed(3)],
  ['RFF + linear','0.00001','120',d.diagnostics[1].objective.toFixed(4),d.diagnostics[1].margin_width.toFixed(3)]
 ],64,282,1152,218,[270,180,150,280,272]);
 txt(s,'rff','Nonlinear mapping: γ = 0.005, 2,048 Fourier features.\nZero-vector objective = 1.0; final objectives are lower.',64,540,1100,98,27,teal);
}
{
 const s=slide('Validation selects the configurations before test scoring','Qiymət üçün validation RMSE, classification üçün positive-class F1 əsas götürüldü. Tree validation F1 ən yüksəkdir, buna görə əsas classifier ağacdır. Test nəticəsi heç bir parametr seçimini dəyişməyib. Source: outputs/frozen_config.json; final_*_search.csv.');
 table(s,[['Candidate','Selected configuration','Validation'],
  ['Price tree','Spatial + category routing; depth 16, leaf 5',fmt(metrics('own_tree_regression','validation').rmse)+' AZN RMSE'],
  ['Tier tree','Entropy; depth 16, leaf 10',metrics('own_tree','validation').f1.toFixed(4)+' F1'],
  ['Linear SVM','λ = 0.001; 120 epochs',metrics('own_linear','validation').f1.toFixed(4)+' F1'],
  ['RFF SVM','γ = 0.005; λ = 0.00001; 2,048 features',metrics('own_rff','validation').f1.toFixed(4)+' F1']
 ],64,178,1152,350,[230,630,292]);
 txt(s,'protocol','Frozen configuration → five-fold stability check → held-out test\nFinal training uses the original training subset only.',64,568,1130,86,27,teal);
}
{
 const s=slide('The selected tree reaches test F1 = '+tree.f1.toFixed(4),'Test nəticələri əvvəl seçilən parametrlərlə alınıb. Solda yalnız öz modellərimizin F1 nəticələri, sağda seçilən ağacın confusion matrix-i var. Premium strict price>207000 AZN. Source: outputs/final_metrics.json.');
 bars(s,['Tree','Linear SVM','RFF SVM'],['own_tree','own_linear','own_rff'].map(n=>metrics(n).f1),64,190,670,390,{axis:'Test F1'});
 table(s,[['Actual / predicted','Standard','Premium'],['Standard',...tree.confusion_matrix[0].map(v=>fmt(v))],['Premium',...tree.confusion_matrix[1].map(v=>fmt(v))]],792,210,424,230,[160,132,132]);
 txt(s,'auc','ROC AUC: '+tree.roc_auc.toFixed(4)+'\nPrecision: '+tree.precision.toFixed(4)+'\nRecall: '+tree.recall.toFixed(4),792,476,420,128,26,teal);
}
{
 const s=slide('Price accuracy is useful for typical rows, unstable in the tail','Median nisbi səhv 8.74 faizdir, amma R2 0.104dür. Bu fərq çox böyük qiymət xətaları ilə izah olunur. Heç bir test sətri silinməyib. Məqsəd zəif metrikləri gizlətməməkdir. Source: outputs/final_metrics.json; outputs/final_error_analysis.json.');
 table(s,[['Model','RMSE (AZN)','MAE (AZN)','R²'],...['training_median','own_tree_regression','sklearn_tree_regression'].map((n,i)=>[[ 'Train median','Own spatial trees','sklearn spatial trees'][i],fmt(metrics(n).rmse),fmt(metrics(n).mae),metrics(n).r2.toFixed(3)])],64,184,1152,276,[400,260,260,232]);
 txt(s,'typical','Median absolute percentage error: '+reg.median_ape_pct.toFixed(2)+'%\n'+reg.within_20_pct.toFixed(1)+'% of test prices are predicted within ±20%',64,510,1100,102,31,teal,true);
}
{
 const s=slide('Grouped cross-validation exposes price-label risk','CV ayrı train hissəsində 5 fold aparılıb, testin təkrar ölçülməsi deyil. 600 milyon AZN-lik qeyd bir fold-da evaluation tərəfinə düşüb RMSE-ni kəskin artırır. Buna görə qeyri-sabitliyi hesabatda göstəririk. Source: outputs/final_cv_folds.csv; final_cv_summary.csv.');
 const rr=d.cv_folds.filter(r=>r.model==='own_tree_regression');
 bars(s,rr.map(r=>'Fold '+r.fold),rr.map(r=>r.rmse/1e6),64,186,660,350,{max:7.5,format:'0.00',axis:'RMSE (million AZN)',color:blue});
 table(s,[['Classifier','CV F1 mean ± SD'],...['own_tree','own_linear','own_rff'].map(n=>{const r=d.cv.find(r=>r.model===n);return [{own_tree:'Own tree',own_linear:'Linear SVM',own_rff:'RFF SVM'}[n],r.f1_mean.toFixed(4)+' ± '+r.f1_std.toFixed(4)]})],784,200,432,276,[180,252]);
 txt(s,'outlier','Unverified extreme training price: 600,000,000 AZN.\nFixed configurations; CV is a stability check, not nested tuning.',64,558,1130,92,25,muted);
}
{
 const s=slide('Detached houses and threshold-adjacent prices are harder','Həyət evlərində median nisbi səhv daha böyükdür. Premium həddinə yaxın 1041 sətirdə ağacın səhvi 27.95 faiz, qalan hissədə 5.17 faizdir. Ümumi yaxşı F1 bütün altqruplarda eyni keyfiyyət demək deyil. Source: outputs/final_regression_error_by_category.csv; final_classification_error_groups.csv.');
 const cats=['yeni tikili','köhnə tikili','həyət evi/bağ evi'];
 bars(s,['New apartments','Old apartments','Houses'],cats.map(c=>d.category.find(r=>r.category===c).median_ape_pct/100),64,190,680,360,{max:.25,format:'0.0%',axis:'Median relative price error',color:teal});
 txt(s,'near-title','Tier-classification error',790,194,420,62,28,blue,true);
 txt(s,'near','28.0% near the threshold\n5.2% farther away',790,294,420,120,31);
 txt(s,'definition','Near = price within 10%\nof the 207,000 AZN threshold.',790,438,420,100,23,muted);
 txt(s,'tail','The worst 1% of test rows contribute '+(d.errors.top_one_percent_sse_share*100).toFixed(1)+'% of squared price error. All rows stay in evaluation.',64,580,1130,68,24);
}
{
 const s=slide('Matched sklearn baselines closely track the own models','Baseline-lar eyni input, eyni lambda və model parametrləri ilə müqayisə olunur. SGD-nin optimizer addımları tam eyni deyil. Tree-də tie-breaking və float fərqləri ola bilər. Core algoritm fayllarında sklearn import yoxdur. Source: outputs/final_metrics.json; src/final_models.py.');
 table(s,[['Task / model','Own result','sklearn result'],['Price RMSE (AZN)',fmt(reg.rmse),fmt(metrics('sklearn_tree_regression').rmse)],
  ['Tree F1',tree.f1.toFixed(4),metrics('sklearn_tree').f1.toFixed(4)],
  ['Linear SVM F1',metrics('own_linear').f1.toFixed(4),metrics('sklearn_linear').f1.toFixed(4)],
  ['RFF SVM F1',rff.f1.toFixed(4),metrics('sklearn_rff').f1.toFixed(4)]
 ],64,180,1152,352,[520,316,316]);
 txt(s,'fairness','SVM scaling is inside Pipeline. Both RFF solvers share the same mapper.\nDifferent optimizers and tree tie-breaking explain small gaps.',64,570,1130,86,25,muted);
}
{
 const s=slide('Reproducible code, honest limits and a short demo','Demo: FINAL_Project.ipynb açılır; hazır metriklər, seçilmiş parametrlər, CV, diaqnostika və nümunə qiymət göstərilir. Tam təkrar python -m src.run_all. AI dəstəyi açıqlanır. Komanda adları, real töhfələr və repo URL istifadəçi tərəfindən tamamlanmalıdır. Sources: README.md; contribution_report.md; submission_metadata.json.');
 txt(s,'command','python -m src.run_all',64,180,1100,60,40,teal,true);
 txt(s,'deliverables','NumPy models + tests + saved estimators\nExecuted notebook + numerical outputs\nIEEE-style report + editable slides',64,286,1110,166,30);
 txt(s,'limits','Historical asking prices; no independent transaction validation.\nFuture-period performance and extreme labels remain unresolved.',64,490,1130,82,25,muted);
 txt(s,'disclosure','Developed with substantial ChatGPT/Codex assistance; human contributions must be confirmed by the team.',64,600,1100,52,21,muted);
}
await fs.mkdir(build,{recursive:true});
await fs.mkdir(path.join(build,'previews'),{recursive:true});
const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
await finalizePresentation({
 workspaceDir:work,candidatePath:candidate,finalPath:path.join(output,'ML_Final_Slides_Reviewed.pptx'),
 pythonExecutable:process.env.CODEX_PRIMARY_RUNTIME_PYTHON,
 integrityValidatorPath:'/root/.codex/skills/builtins/presentations/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:'/root/.codex/skills/builtins/presentations/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...[2,4,5,6,7,8,10].flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:11,requiredNativeTableOwnerSlides:[2,4,5,6,7,8,10],
 requiredNativeChartOwnerSlides:[6,8,9],materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
 receiptPath:path.join(build,'validation_reviewed.json')
});
const final=await PresentationFile.importPptx(await FileBlob.load(path.join(output,'ML_Final_Slides_Reviewed.pptx')));
for(let i=0;i<final.slides.items.length;i++){
 const blob=await final.export({slide:final.slides.items[i],format:'png',scale:2});
 await fs.writeFile(path.join(build,'previews','slide-'+String(i+1).padStart(2,'0')+'.png'),new Uint8Array(await blob.arrayBuffer()));
}
console.log('Exported reviewed PPTX and high-resolution slide previews');
