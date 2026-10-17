"""Single-file Streamlit interface for the existing ML_Final_Project.
Place beside FINAL_Project.ipynb; run: python -m streamlit run app.py.
Uses existing src/, models/, outputs/, report/ and presentation/ files.
Requires streamlit==1.51.0 and pyarrow==18.1.0 plus project dependencies.
"""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from src.predict import load_models, listing_frame, predict_listing

ROOT = Path(__file__).resolve().parent
# Aggregated training locations embedded so no demo_assets folder is needed.
LOCATIONS = json.loads('{"astara": {"n": 4, "locations": {"astara": {"n": 4, "latitude": 38.46298089314335, "longitude": 48.877165383835646}}}, "ağdam": {"n": 2, "locations": {"ağdam": {"n": 2, "latitude": 40.12277466800056, "longitude": 47.10229202035343}}}, "bakı": {"n": 40806, "locations": {"1-ci mikrorayon q.": {"n": 2, "latitude": 40.3892052429612, "longitude": 49.83376387432935}, "2-ci alatava q.": {"n": 2, "latitude": 40.412755276270616, "longitude": 49.800520883018805}, "2-ci mikrorayon q.": {"n": 6, "latitude": 40.3923258, "longitude": 49.8288055}, "20 yanvar m.": {"n": 956, "latitude": 40.400463043916346, "longitude": 49.8068256765243}, "20-ci sahə q.": {"n": 34, "latitude": 40.33361581807298, "longitude": 49.81900389999998}, "28 may m.": {"n": 1503, "latitude": 40.3836431819486, "longitude": 49.84727740287781}, "28 may q.": {"n": 1, "latitude": 40.459781, "longitude": 49.643429}, "3-cü mikrorayon q.": {"n": 8, "latitude": 40.39395674915768, "longitude": 49.828805500000016}, "4-cü mikrorayon q.": {"n": 11, "latitude": 40.40570659694362, "longitude": 49.82396923084144}, "5-ci mikrorayon q.": {"n": 5, "latitude": 40.3923258, "longitude": 49.8288055}, "6-cı mikrorayon q.": {"n": 15, "latitude": 40.41633513697491, "longitude": 49.83079094751883}, "7-ci mikrorayon q.": {"n": 230, "latitude": 40.43179163232364, "longitude": 49.85311720000004}, "8 noyabr m.": {"n": 1159, "latitude": 40.3979898201916, "longitude": 49.8186120876096}, "8-ci kilometr q.": {"n": 91, "latitude": 40.4087847, "longitude": 49.956158405837}, "8-ci mikrorayon q.": {"n": 189, "latitude": 40.41643621026604, "longitude": 49.8414062089844}, "9-cu mikrorayon q.": {"n": 175, "latitude": 40.42502597696882, "longitude": 49.8135998}, "abşeron r.": {"n": 144, "latitude": 40.45657684360009, "longitude": 49.73728772303343}, "albalılıq q.": {"n": 33, "latitude": 40.5348886759323, "longitude": 50.04792}, "avtovağzal m.": {"n": 155, "latitude": 40.42265102311419, "longitude": 49.7989800528963}, "azadlıq prospekti m.": {"n": 818, "latitude": 40.425594395075805, "longitude": 49.84264437909725}, "ağ şəhər q.": {"n": 580, "latitude": 40.38142347444355, "longitude": 49.8875794981187}, "aşağı güzdək q.": {"n": 7, "latitude": 40.4656171786638, "longitude": 49.6621513366699}, "badamdar q.": {"n": 620, "latitude": 40.3395229, "longitude": 49.807119968803036}, "bahar q.": {"n": 5, "latitude": 40.368953835405, "longitude": 50.02724048144531}, "bakmil m.": {"n": 17, "latitude": 40.4105114567525, "longitude": 49.8721853611325}, "bakıxanov q.": {"n": 640, "latitude": 40.423887405775815, "longitude": 49.96036758299035}, "balaxanı q.": {"n": 10, "latitude": 40.466828285192705, "longitude": 49.9232669109721}, "bayıl q.": {"n": 252, "latitude": 40.342833134572146, "longitude": 49.83206257850038}, "bibi heybət q.": {"n": 4, "latitude": 40.30712378325275, "longitude": 49.8220375509262}, "bilgəh q.": {"n": 104, "latitude": 40.579143465909254, "longitude": 50.03929147112281}, "biləcəri q.": {"n": 400, "latitude": 40.440334, "longitude": 49.805675184884706}, "binə q.": {"n": 406, "latitude": 40.45301389958005, "longitude": 50.09046802408557}, "binəqədi q.": {"n": 240, "latitude": 40.47460637186578, "longitude": 49.8223795}, "binəqədi r.": {"n": 235, "latitude": 40.4308376694817, "longitude": 49.83354292572479}, "buzovna q.": {"n": 219, "latitude": 40.51676808900779, "longitude": 50.09938820261888}, "böyükşor q.": {"n": 8, "latitude": 40.4061791713598, "longitude": 49.874988338470445}, "bülbülə q.": {"n": 20, "latitude": 40.43888890076545, "longitude": 49.9726639653161}, "ceyranbatan q.": {"n": 28, "latitude": 40.543139, "longitude": 49.6668291693115}, "digah q.": {"n": 3, "latitude": 40.49456314322376, "longitude": 49.8794000414982}, "dübəndi q.": {"n": 11, "latitude": 40.4199506, "longitude": 50.29582129730308}, "dədə qorqud q.": {"n": 6, "latitude": 40.38772191479734, "longitude": 50.02112991750812}, "dərnəgül m.": {"n": 133, "latitude": 40.4244377652989, "longitude": 49.8562826219482}, "elmlər akademiyası m.": {"n": 1793, "latitude": 40.3785181337562, "longitude": 49.8118231058976}, "fatmayı q.": {"n": 40, "latitude": 40.534455871911824, "longitude": 49.84690579184134}, "görədil q.": {"n": 48, "latitude": 40.5540899374015, "longitude": 49.8288345336914}, "günəşli q.": {"n": 6, "latitude": 40.37535737499444, "longitude": 49.9787019553967}, "gənclik m.": {"n": 986, "latitude": 40.39979202533585, "longitude": 49.85616644237945}, "hökməli q.": {"n": 32, "latitude": 40.43567249041701, "longitude": 49.7297791687815}, "hövsan q.": {"n": 430, "latitude": 40.376790322657385, "longitude": 50.08366584777832}, "həzi aslanov m.": {"n": 1822, "latitude": 40.3720002372081, "longitude": 49.95378681262176}, "həzi aslanov q.": {"n": 349, "latitude": 40.3681299948233, "longitude": 49.94195938110352}, "i̇nşaatçılar m.": {"n": 2128, "latitude": 40.39179112020875, "longitude": 49.80237833436985}, "i̇çəri şəhər m.": {"n": 478, "latitude": 40.367194414859554, "longitude": 49.828727571333744}, "keşlə q.": {"n": 27, "latitude": 40.3959935058258, "longitude": 49.89532689225907}, "koroğlu m.": {"n": 106, "latitude": 40.41961485808376, "longitude": 49.9115875544729}, "kubinka q.": {"n": 12, "latitude": 40.38773599270815, "longitude": 49.83687488385348}, "köhnə günəşli q.": {"n": 155, "latitude": 40.3628462991259, "longitude": 49.9686585881652}, "kürdəxanı q.": {"n": 21, "latitude": 40.5474611148012, "longitude": 49.92060786885111}, "lökbatan q.": {"n": 94, "latitude": 40.32619456875785, "longitude": 49.73354977529822}, "m.ə.rəsulzadə q.": {"n": 40, "latitude": 40.4363194567508, "longitude": 49.8376157934142}, "masazır q.": {"n": 1176, "latitude": 40.4862557, "longitude": 49.75517460535677}, "massiv a q.": {"n": 2, "latitude": 40.388467920930054, "longitude": 49.976010479172}, "massiv d q.": {"n": 33, "latitude": 40.37207867584335, "longitude": 49.97785686876877}, "massiv g q.": {"n": 3, "latitude": 40.37944971084017, "longitude": 49.9776093195435}, "massiv v q.": {"n": 19, "latitude": 40.3812706631891, "longitude": 49.98146981775296}, "maştağa q.": {"n": 113, "latitude": 40.524538986095, "longitude": 50.00042004229736}, "mehdiabad q.": {"n": 105, "latitude": 40.49597009145496, "longitude": 49.8532716}, "memar əcəmi m.": {"n": 1485, "latitude": 40.4136237, "longitude": 49.8138449081363}, "müşfiqabad q.": {"n": 9, "latitude": 40.4693535924041, "longitude": 49.6275631195351}, "məmmədli q.": {"n": 104, "latitude": 40.50986550422709, "longitude": 49.928212991304875}, "mərdəkan q.": {"n": 1387, "latitude": 40.4807244368274, "longitude": 50.14697509521483}, "nardaran q.": {"n": 231, "latitude": 40.58756681438845, "longitude": 49.984684555959}, "neftçilər m.": {"n": 868, "latitude": 40.41013491885985, "longitude": 49.943776418878244}, "nizami m.": {"n": 1071, "latitude": 40.3824220044237, "longitude": 49.8307448888904}, "nizami r.": {"n": 171, "latitude": 40.4087847, "longitude": 49.9536982217707}, "novxanı q.": {"n": 249, "latitude": 40.5535017447222, "longitude": 49.7777582005991}, "nzs q.": {"n": 45, "latitude": 40.3870162169286, "longitude": 49.9165194357335}, "nəriman nərimanov m.": {"n": 2221, "latitude": 40.4034313583125, "longitude": 49.8696151705393}, "nərimanov r.": {"n": 1528, "latitude": 40.39268428563567, "longitude": 49.854755401611314}, "nəsimi m.": {"n": 507, "latitude": 40.42376713599278, "longitude": 49.82191644117711}, "nəsimi r.": {"n": 1690, "latitude": 40.3944110704977, "longitude": 49.836270959695966}, "pirallahı r.": {"n": 10, "latitude": 40.454146215471745, "longitude": 50.331135}, "pirşağı q.": {"n": 45, "latitude": 40.5640901422729, "longitude": 49.8894540853791}, "pirəkəşkül q.": {"n": 2, "latitude": 40.50697014744365, "longitude": 49.54995473799669}, "puta q.": {"n": 1, "latitude": 40.2960324883051, "longitude": 49.6651968308269}, "qala q.": {"n": 39, "latitude": 40.4433858239896, "longitude": 50.16960959403966}, "qara qarayev m.": {"n": 1047, "latitude": 40.4174692151705, "longitude": 49.93414878845215}, "qaradağ r.": {"n": 2, "latitude": 40.34818399734904, "longitude": 49.83612214377565}, "qaraçuxur q.": {"n": 217, "latitude": 40.399836, "longitude": 49.978407}, "qobu q.": {"n": 6, "latitude": 40.4094163136449, "longitude": 49.711872814966654}, "qobustan q.": {"n": 3, "latitude": 40.07653280996777, "longitude": 49.40314468454437}, "ramana q.": {"n": 47, "latitude": 40.4587278932807, "longitude": 49.9848006286629}, "sabunçu q.": {"n": 40, "latitude": 40.44507911776867, "longitude": 49.944488201141354}, "sabunçu r.": {"n": 96, "latitude": 40.50142604539805, "longitude": 49.967202588942754}, "sahil m.": {"n": 325, "latitude": 40.3726980766567, "longitude": 49.8410432423893}, "sahil q.": {"n": 6, "latitude": 40.22124768903217, "longitude": 49.569190624328606}, "saray q.": {"n": 82, "latitude": 40.5317647, "longitude": 49.7132454}, "savalan q.": {"n": 47, "latitude": 40.5149340049449, "longitude": 49.969873}, "sulutəpə q.": {"n": 68, "latitude": 40.4312232234556, "longitude": 49.76411283016205}, "suraxanı q.": {"n": 10, "latitude": 40.40274356179417, "longitude": 50.001277300403004}, "suraxanı r.": {"n": 50, "latitude": 40.38024260776298, "longitude": 49.9794022041627}, "səbail r.": {"n": 254, "latitude": 40.34132561205516, "longitude": 49.82289396655282}, "səngəçal q.": {"n": 3, "latitude": 40.15531325579851, "longitude": 49.470108650975}, "türkan q.": {"n": 27, "latitude": 40.3650213917992, "longitude": 50.2150726318359}, "ulduz m.": {"n": 3, "latitude": 40.4129176153487, "longitude": 49.89977008610165}, "xalqlar dostluğu m.": {"n": 700, "latitude": 40.39798990250573, "longitude": 49.95308901144185}, "xocəsən m.": {"n": 12, "latitude": 40.42621877958911, "longitude": 49.77676650551252}, "xocəsən q.": {"n": 22, "latitude": 40.40957492033851, "longitude": 49.76872289677521}, "xutor q.": {"n": 3, "latitude": 40.4133688027779, "longitude": 49.82841026748963}, "xətai r.": {"n": 946, "latitude": 40.3690432061184, "longitude": 49.9503894285606}, "xəzər r.": {"n": 49, "latitude": 40.4744727488776, "longitude": 50.10286331351381}, "yasamal q.": {"n": 167, "latitude": 40.3795053385315, "longitude": 49.80240127473796}, "yasamal r.": {"n": 493, "latitude": 40.39029796846838, "longitude": 49.8027310033269}, "yeni günəşli q.": {"n": 347, "latitude": 40.3812706631891, "longitude": 49.97888791003417}, "yeni ramana q.": {"n": 42, "latitude": 40.44917103295441, "longitude": 49.99335614311886}, "yeni suraxanı q.": {"n": 11, "latitude": 40.42505366797698, "longitude": 50.0374188653333}, "yeni yasamal q.": {"n": 605, "latitude": 40.39249649202242, "longitude": 49.7901147849029}, "zabrat q.": {"n": 101, "latitude": 40.4832778766452, "longitude": 49.95102681108393}, "zirə q.": {"n": 6, "latitude": 40.3569388056456, "longitude": 50.28159372358286}, "zuğulba q.": {"n": 1, "latitude": 40.5287520956478, "longitude": 50.1011198730469}, "zığ q.": {"n": 91, "latitude": 40.3654219491074, "longitude": 49.9860436904387}, "çiçək q.": {"n": 2, "latitude": 40.44156804542407, "longitude": 49.717592439512835}, "şah i̇smayıl xətai m.": {"n": 1730, "latitude": 40.38317145093596, "longitude": 49.87222648780603}, "şağan q.": {"n": 44, "latitude": 40.4906394, "longitude": 50.11615816627955}, "şimal dres q.": {"n": 7, "latitude": 40.49601909061482, "longitude": 50.2007949953842}, "şüvəlan q.": {"n": 571, "latitude": 40.4898852924193, "longitude": 50.208}, "şıxov q.": {"n": 46, "latitude": 40.30604244605345, "longitude": 49.78273416918945}, "şərq q.": {"n": 1, "latitude": 40.3693418640232, "longitude": 49.9752701262249}, "əhmədli m.": {"n": 887, "latitude": 40.3843922446399, "longitude": 49.9560800212402}, "əhmədli q.": {"n": 349, "latitude": 40.36949562437751, "longitude": 49.96147205913307}, "ələt q.": {"n": 1, "latitude": 39.9524631831995, "longitude": 49.4112384194611}, "əmircan q.": {"n": 73, "latitude": 40.4216673166578, "longitude": 49.994373221162}}}, "gədəbəy": {"n": 2, "locations": {"gədəbəy": {"n": 2, "latitude": 40.62574541353818, "longitude": 45.81708582597993}}}, "naxçıvan mr": {"n": 2, "locations": {"naxçıvan mr": {"n": 2, "latitude": 39.27244555828246, "longitude": 45.40750848916825}}}, "sumqayıt": {"n": 2, "locations": {"sumqayıt": {"n": 2, "latitude": 40.5490804437011, "longitude": 49.7134762585222}}}, "xaçmaz": {"n": 30, "locations": {"xaçmaz": {"n": 30, "latitude": 41.4647380819404, "longitude": 48.79009272466463}}}, "xırdalan": {"n": 84, "locations": {"xırdalan": {"n": 84, "latitude": 40.4551119, "longitude": 49.7445938999999}}}}')
MODEL_LABELS = {'own_tree': 'Decision Tree', 'own_linear': 'Linear Pegasos', 'own_rff': 'RFF + Pegasos'}
CATEGORIES = {'Yeni tikili': 'yeni tikili', 'Köhnə tikili': 'köhnə tikili', 'Həyət / bağ evi': 'həyət evi/bağ evi'}
ANSWERS = {'Var': 'var', 'Yoxdur': 'yoxdur', 'Məlumat yoxdur': None}

def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))

def money(value):
    return f'{value:,.0f}'.replace(',', ' ') + ' AZN'

def display_name(value):
    return value.replace('i\u0307', 'i').capitalize()

def validate(values):
    errors = []
    if values.get('category') not in CATEGORIES.values():
        errors.append('Yalnız yaşayış əmlakı dəstəklənir.')
    for key, label in [('area_m2','Sahə'),('rooms','Otaq sayı')]:
        val = values.get(key)
        if val is None or not math.isfinite(float(val)) or float(val) <= 0:
            errors.append(f'{label} sıfırdan böyük olmalıdır.')
    rooms = values.get('rooms')
    if rooms is not None and math.isfinite(float(rooms)) and float(rooms) % 1:
        errors.append('Otaq sayı tam ədəd olmalıdır.')
    floor, total = values.get('floor'), values.get('total_floors')
    if floor is not None and total is not None and not (1 <= floor <= total):
        errors.append('Mənzilin mərtəbəsi 1 ilə binanın mərtəbə sayı arasında olmalıdır.')
    land = values.get('land_area_m2')
    if land is not None and (not math.isfinite(float(land)) or land <= 0):
        errors.append('Torpaq sahəsi sıfırdan böyük olmalıdır.')
    lat, lon = values.get('latitude'), values.get('longitude')
    if (lat is None) != (lon is None):
        errors.append('Hər iki koordinatı daxil et və ya hər ikisini boş saxla.')
    if lat is not None and lon is not None:
        if not (38 <= lat <= 42.5 and 44 <= lon <= 51):
            errors.append('Koordinatlar modelin qəbul etdiyi Azərbaycan ərazisi diapazonundan kənardadır.')
    return errors

def predict_all(values, bundle):
    errors = validate(values)
    if errors:
        raise ValueError(' '.join(errors))
    result = predict_listing(values, bundle)
    frame = listing_frame(values)
    result['classifications'] = {
        name: ('Premium' if int(bundle['models'][name].predict(frame)[0]) else 'Standart')
        for name in MODEL_LABELS
    }
    result['input'] = dict(values)
    # Tree probabilities and SVM margins are not advertised as confidence.
    return result

def comparison_table(metrics, search, config):
    rows = []
    for name, label in MODEL_LABELS.items():
        kind = name.removeprefix('own_')
        selected = config['classification'][kind]
        matched = search[search.config.map(lambda s: json.loads(s) == selected)]
        val = metrics[(metrics.model == name) & (metrics.subset == 'validation')].iloc[0]
        test = metrics[(metrics.model == name) & (metrics.subset == 'test')].iloc[0]
        rows.append({'Model':label,'Train F1':float(matched.iloc[-1].train_f1) if len(matched) else np.nan,
            'Validation F1':val.f1,'Test F1':test.f1,'Test accuracy (%)':100*test.accuracy,'Test ROC AUC':test.roc_auc})
    return pd.DataFrame(rows)


st.set_page_config(page_title='Residential ML | Canlı demo', page_icon='🏠', layout='wide')
st.markdown("""
<style>
/* Fon, mətn və xanalar Streamlit-in seçilmiş temasına uyğun qalır. */
.block-container {
    max-width: 1320px;
    padding-top: 3.5rem;
    padding-bottom: 3rem;
}

/* Başlıq paneli */
.hero {
    background: linear-gradient(120deg, #102d40, #115e59);
    border: 1px solid #28716e;
    border-radius: 20px;
    padding: 30px 32px;
    margin-bottom: 26px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
}

.hero h1 {
    color: #ffffff !important;
    font-size: clamp(1.6rem, 3vw, 2.2rem);
    font-weight: 750;
    line-height: 1.25;
    margin: 10px 0 14px;
}

.hero p {
    color: #e0f2f1 !important;
    font-size: 1rem;
    line-height: 1.7;
    margin: 0;
}

.hero .eyebrow {
    color: #99f6e4 !important;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.13em;
}

/* Nəticə kartları */
[data-testid="stMetric"] {
    background: transparent;
    border: 1px solid rgba(128, 128, 128, 0.35);
    border-left: 4px solid #14b8a6;
    border-radius: 14px;
    padding: 18px 20px;
}

[data-testid="stMetricValue"] {
    color: inherit;
    font-weight: 700;
}

/* Məlumat forması */
[data-testid="stForm"] {
    border: 1px solid rgba(128, 128, 128, 0.35);
    border-radius: 16px;
    padding: 22px;
}

[data-testid="stWidgetLabel"] p {
    font-weight: 600;
}

/* Əsas düymə */
button[kind="primary"],
button[kind="primaryFormSubmit"] {
    background-color: #0f766e !important;
    border: 1px solid #0f766e !important;
    color: #ffffff !important;
    border-radius: 10px;
    min-height: 46px;
    font-weight: 700;
    transition: background-color 0.15s ease;
}

button[kind="primary"] p,
button[kind="primaryFormSubmit"] p {
    color: #ffffff !important;
}

button[kind="primary"]:hover,
button[kind="primaryFormSubmit"]:hover {
    background-color: #115e59 !important;
    border-color: #115e59 !important;
}

button[kind="primary"]:focus-visible,
button[kind="primaryFormSubmit"]:focus-visible {
    outline: 3px solid #14b8a6 !important;
    outline-offset: 3px;
}

/* Bölmə seçimləri */
[data-baseweb="tab-list"] {
    gap: 20px;
}

button[data-baseweb="tab"] {
    color: inherit !important;
    padding: 12px 4px;
}

button[data-baseweb="tab"][aria-selected="true"] {
    font-weight: 700;
}

[data-baseweb="tab-highlight"] {
    background-color: #14b8a6 !important;
}

/* Kiçik ekranlar */
@media (max-width: 640px) {
    .hero {
        padding: 22px 18px;
    }

    [data-testid="stForm"] {
        padding: 14px;
    }

    [data-baseweb="tab-list"] {
        gap: 10px;
    }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner='Hazır modellər yüklənir...')
def get_bundle(path, modified_ns):
    return load_models(path)

@st.cache_data
def get_tables():
    return (pd.DataFrame(read_json('outputs/final_metrics.json')),
            pd.read_csv(ROOT/'outputs/final_classification_search.csv'),
            read_json('outputs/final_data_summary.json'),
            LOCATIONS)

model_path = ROOT/'models/final_models.pkl'
if not model_path.is_file():
    st.error('models/final_models.pkl tapılmadı. app.py faylını FINAL_Project.ipynb olan əsas layihə qovluğuna qoy.')
    st.stop()
try:
    bundle = get_bundle(str(model_path), model_path.stat().st_mtime_ns)
    metrics, search, summary, locations = get_tables()
except Exception as exc:
    st.error('Model və ya nəticə faylları yüklənmədi. Layihənin .venv mühitindən istifadə et.')
    with st.expander('Xəta məlumatı'):
        st.code(f'{type(exc).__name__}: {exc}')
    st.stop()

test = metrics[metrics.subset == 'test'].set_index('model')
reg = test.loc['own_tree_regression']
threshold = bundle['config']['threshold']

with st.sidebar:
    st.markdown('### RESIDENTIAL ML')
    st.caption('AI Academy · Final layihə')
    st.divider()
    st.markdown('**Modellər**\n\nNumPy Decision Tree\n\nLinear Pegasos SVM\n\nRFF + Pegasos SVM')
    st.divider()
    st.metric('Təlim elanı', f"{summary['split']['train']['n']:,}".replace(',', ' '))
    st.metric('Test elanı', f"{summary['split']['test']['n']:,}".replace(',', ' '))
    st.caption('Yeni/köhnə tikili və həyət/bağ evi. Ofis, obyekt, torpaq və qaraj daxil deyil.')
    st.caption('Demo saxlanmış modelləri işlədir. Təlimi təkrarlamır.')

st.markdown('''<div class="hero"><div class="eyebrow">FROM-SCRATCH MACHINE LEARNING · LIVE DEMO</div>
<h1>Evin məlumatından qiymət proqnozuna</h1>
<p>Elan qiymətini hesabla, üç modelin premium / standart qərarını müqayisə et.</p></div>''', unsafe_allow_html=True)
demo, evaluation, method = st.tabs(['Canlı proqnoz', 'Model nəticələri', 'Metodologiya və təqdimat'])

with demo:
    left, right = st.columns([1.15,1], gap='large')
    with left:
        st.subheader('Əmlak məlumatları')
        cat_label = st.selectbox('Əmlak növü', list(CATEGORIES), key='category_label')
        category = CATEGORIES[cat_label]
        c1,c2 = st.columns(2)
        city_options = sorted(locations, key=lambda c: -locations[c]['n'])
        city = c1.selectbox('Şəhər', city_options, format_func=display_name, key='city')
        loc_options = sorted(locations[city]['locations'])
        loc_default = loc_options.index('yasamal r.') if 'yasamal r.' in loc_options else 0
        location = c2.selectbox('Ərazi / məkan', loc_options, index=loc_default,
                                format_func=display_name, key=f'location_{city}')
        loc_data = locations[city]['locations'][location]
        if locations[city]['n'] < 200:
            st.warning(f"Bu şəhər üçün train-də yalnız {locations[city]['n']} elan var. Nəticənin etibarlılığı daha məhduddur.")
        with st.expander('Koordinat seçimi', expanded=False):
            geo_mode = st.radio('Coğrafi məlumat', ['Ərazinin median koordinatı', 'Dəqiq koordinat', 'Məlumat yoxdur'], key='geo_mode')
            st.caption('Median koordinat eyni ərazinin təlim elanlarından hesablanır; dəqiq ünvan deyil. Dəqiq koordinat daha uyğun giriş verir.')
            if geo_mode == 'Dəqiq koordinat':
                g1,g2 = st.columns(2)
                lat = g1.number_input('Latitude', min_value=38.0, max_value=42.5,
                    value=loc_data['latitude'] or 40.4, format='%.6f', key=f'lat_{city}_{location}')
                lon = g2.number_input('Longitude', min_value=44.0, max_value=51.0,
                    value=loc_data['longitude'] or 49.82, format='%.6f', key=f'lon_{city}_{location}')
            elif geo_mode == 'Məlumat yoxdur':
                lat=lon=None
            else:
                lat,lon=loc_data['latitude'],loc_data['longitude']
                if lat is not None:
                    st.caption(f'Təxmini nöqtə: {lat:.5f}, {lon:.5f}')
        with st.form('listing_form'):
            a,b = st.columns(2)
            area = a.number_input('Sahə (m²)', min_value=1.0, value=100.0, step=5.0, key='area')
            rooms = b.number_input('Otaq sayı', min_value=1, value=3, step=1, key='rooms')
            if category != 'həyət evi/bağ evi':
                a,b = st.columns(2)
                floor = a.number_input('Mənzilin mərtəbəsi', min_value=1, value=7, step=1, key='floor')
                total = b.number_input('Binanın mərtəbə sayı', min_value=1, value=16, step=1, key='total_floors')
                land = None
            else:
                land_unknown = st.checkbox('Torpaq sahəsi məlum deyil', value=False, key='land_unknown')
                land_value = st.number_input('Torpaq sahəsi (m²)', min_value=1.0, value=400.0, step=50.0, key='land')
                land = None if land_unknown else land_value
                floor=total=None
                st.caption('1 sot = 100 m². Həyət evi üçün mənzil mərtəbəsi sahələri boş saxlanır.')
            a,b,c = st.columns(3)
            repair = ANSWERS[a.selectbox('Təmir', list(ANSWERS), key='repair')]
            bill = ANSWERS[b.selectbox('Çıxarış', list(ANSWERS), key='bill_of_sale')]
            mortgage = ANSWERS[c.selectbox('İpoteka', list(ANSWERS), index=2, key='mortgage')]
            submitted = st.form_submit_button('Qiyməti hesabla', type='primary', width="stretch")

        current_context = [category,city,location,geo_mode,lat,lon]
        if 'prediction' in st.session_state and st.session_state.get('context') != current_context:
            del st.session_state['prediction']
        if submitted:
            st.session_state.pop('prediction', None)
            values = dict(category=category, city=city, location=location, area_m2=area,
                land_area_m2=land, rooms=rooms, floor=floor, total_floors=total,
                latitude=lat, longitude=lon, repair=repair, bill_of_sale=bill, mortgage=mortgage)
            problems=validate(values)
            if problems:
                for problem in problems: st.error(problem)
            else:
                try:
                    result=predict_all(values,bundle)
                    result['coordinate_source']=geo_mode
                    st.session_state['prediction']=result
                    st.session_state['context']=current_context
                except Exception as exc:
                    st.error(f'Proqnoz alınmadı: {exc}')
    with right:
        st.subheader('Proqnoz nəticəsi')
        result=st.session_state.get('prediction')
        if result is None:
            with st.container(border=True):
                st.markdown('### Evin məlumatlarını daxil et')
                st.write('“Qiyməti hesabla” düyməsi saxlanmış modellərlə real proqnoz yaradır.')
                st.caption('Başlanğıc rəqəmlər nümunə girişləridir; real satış elanını təsdiqləmir.')
        else:
            used=result['input']
            st.metric('Proqnozlaşdırılan elan qiyməti',money(result['predicted_price_azn']))
            c1,c2=st.columns(2)
            c1.metric('Decision Tree qərarı',result['classifications']['own_tree'])
            c2.metric('Premium həddi',money(threshold))
            st.caption(f"Son hesablanan elan: {used['area_m2']:g} m² · {used['rooms']} otaq · {display_name(used['location'])}")
            st.caption(f"Koordinat mənbəyi: {result['coordinate_source']}. Girişi dəyişdikdən sonra yenidən hesabla.")
            st.markdown('**Eyni ev üçün üç təsnifat modelinin qərarı**')
            predictions=pd.DataFrame([{'Model':label,'Qərar':result['classifications'][name]} for name,label in MODEL_LABELS.items()])
            st.dataframe(predictions, hide_index=True, width="stretch")
            implied='Premium' if result['predicted_price_azn']>threshold else 'Standart'
            if implied != result['classifications']['own_tree']:
                st.info('Qiymət və sinif ayrı modellərdən gəlir. Bu girişdə onların həddə görə qərarı fərqlənir.')
            st.caption('Premium: təlim medianı olan 207 000 AZN-dən yüksək qiymət sinfi. Bu, lüks keyfiyyət sertifikatı deyil.')
            with st.expander('Hesablamada istifadə olunan girişlər'):
                st.json(used)
            st.download_button('Nəticəni JSON kimi yüklə', data=json.dumps(result,ensure_ascii=False,indent=2),
                file_name='ev_proqnozu.json', mime='application/json')
        st.info('Tarixi elan qiymətləri üzrə tədris demosu. Nəticə cari bazar qiymətləndirməsi deyil.')
        st.caption(f"Bütün testdə: median faiz xətası {reg.median_ape_pct:.2f}% · MAE {money(reg.mae)} · R² {reg.r2:.3f}. Bunlar bu konkret ev üçün zəmanət və ya etibar intervalı deyil.")

with evaluation:
    st.subheader('Görülməmiş test məlumatında nəticələr')
    st.caption('8 724 yaşayış elanı · modellər validation nəticəsinə görə seçilib · test sətirləri çıxarılmayıb')
    a,b,c,d=st.columns(4)
    a.metric('Təsnifat accuracy',f"{100*test.loc['own_tree','accuracy']:.2f}%")
    b.metric('Təsnifat F1',f"{test.loc['own_tree','f1']:.4f}")
    c.metric('Qiymət: median faiz xətası',f'{reg.median_ape_pct:.2f}%')
    d.metric('Qiymət: ±20% daxilində',f'{reg.within_20_pct:.2f}%')
    st.markdown('#### Təsnifat və overfitting yoxlaması')
    table=comparison_table(metrics,search,bundle['config'])
    st.dataframe(table.round(4),hide_index=True,width="stretch")
    st.caption('F1 və accuracy fərqli ölçülərdir. Decision Tree-nin train F1-i daha yüksəkdir; validation və test nəticələri yaxındır. Bu, gələcək dövr üzrə zəmanət vermir.')
    st.image(str(ROOT/'report/figures/final_roc_pr.png'),width="stretch")
    st.markdown('#### Qiymət proqnozunun güclü və zəif tərəfləri')
    a,b,c=st.columns(3)
    a.metric('MAE',money(reg.mae))
    b.metric('RMSE',money(reg.rmse))
    c.metric('R²',f'{reg.r2:.4f}')
    st.warning('Ümumi R² zəifdir. Median faiz xətası aşağı olsa da, model böyük qiymətlərdə ciddi səhvlər edə bilər.')
    groups=pd.read_csv(ROOT/'outputs/final_regression_error_by_category.csv')
    groups=groups[['category','n','mae','median_ape_pct','within_20_pct']].rename(columns={
        'category':'Əmlak növü','n':'Elan sayı','mae':'MAE (AZN)',
        'median_ape_pct':'Median faiz xətası (%)','within_20_pct':'±20% daxilində (%)'})
    st.dataframe(groups.round(2),hide_index=True,width="stretch")
    with st.expander('Ayrı 5-fold CV və SVM optimizasiyası'):
        st.caption('CV ilkin train hissəsində, sabit seçilmiş parametrlərlə aparılıb. Nested CV deyil. 600 milyon AZN-lik yoxlanılmamış qiymət bir fold-un RMSE-sini çox böyüdür.')
        st.image(str(ROOT/'report/figures/final_cv_stability.png'),width="stretch")
        st.image(str(ROOT/'report/figures/final_svm_objective.png'),width="stretch")

with method:
    st.subheader('Məlumat → emal → öz modelimiz → qiymətləndirmə')
    a,b,c=st.columns(3)
    with a:
        st.markdown('#### 01 · Məlumat')
        st.write('100 775 xam sətirdən dublikat tarixçələri təmizlənib; 58 452 yaşayış elanı qalıb.')
        st.write('Train / validation / test: 40 932 / 8 796 / 8 724. Ehtimal olunan dublikat qrupları bölmələr arasında paylaşılmır.')
    with b:
        st.markdown('#### 02 · Modellər')
        st.write('Öz Decision Tree: Gini, entropy və MSE. Öz Pegasos: hinge loss + L2. Öz RFF: qeyri-xətti təsnifat üçün Fourier əlamətləri.')
        st.write('Qiymət: log-qiymət üzərində mənzil və həyət evi üçün ayrı ağaclar, coğrafi məsafələr.')
    with c:
        st.markdown('#### 03 · Dürüst ölçmə')
        st.write('Parametrlər validation-da seçilir. İmputasiya, one-hot kodlama, scaling və coğrafi mərkəzlər train-də öyrənilir.')
        st.write('price, unit_price və total_price giriş feature-ları deyil. Test qiymətləri modeli öyrətmək üçün istifadə olunmur.')
    st.divider()
    st.markdown('#### Model hansı əlamətlərə baxır?')
    importance=pd.read_csv(ROOT/'outputs/final_feature_importance.csv')
    model_choice=st.selectbox('Əlamət əhəmiyyəti', ['Təsnifat ağacı','Qiymət — mənzil','Qiymət — həyət evi'])
    if model_choice=='Təsnifat ağacı':
        chosen=importance[importance.model=='own_tree']
    else:
        route='apartment' if model_choice=='Qiymət — mənzil' else 'house'
        chosen=importance[(importance.model=='own_tree_regression') & (importance.route==route)]
    top=chosen.nlargest(10,'importance')[['feature','importance']].set_index('feature')
    st.bar_chart(top, horizontal=True, color='#127e75')
    st.caption('Əlamət əhəmiyyəti səbəb-nəticə əlaqəsini sübut etmir.')
    with st.expander('Seçilmiş model parametrləri'):
        st.json(bundle['config'])
    a,b=st.columns(2)
    a.download_button('Hesabat PDF', (ROOT/'report/report.pdf').read_bytes(), 'ML_Final_Report.pdf','application/pdf')
    b.download_button('Təqdimat PDF', (ROOT/'presentation/ML_Final_Slides.pdf').read_bytes(), 'ML_Final_Slides.pdf','application/pdf')
    st.caption('Kod, təcrübələr və təqdimat ChatGPT/Codex yardımı ilə hazırlanıb. İnsan müəllifləri və real töhfələr əsas layihədə tamamlanmalıdır.')
