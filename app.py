import json
import math

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Churn Risk Advisor', page_icon='📡',
                   layout='wide')

RAW_COLS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges',
            'TotalCharges']
SERVICES = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies']
PAYMENTS = ['Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)']

# Palette
RED, AMBER, GREEN = '#e4572e', '#f5a524', '#2e9e5b'
TEAL, BLUE, PURPLE = '#00838f', '#2f6fb0', '#7b4fc9'
INK, MUTED, TRACK = '#1f2d3d', '#5b6b7c', '#e3e8ee'
BAND_COLOR = {'HIGH': RED, 'WATCH': AMBER, 'LOW': GREEN}
SENIOR_AGE = 65     # the model only knows SeniorCitizen (0/1), not exact age


# ---------------- Styling ----------------
CSS = """
<style>
.hero {
    background: linear-gradient(120deg, #00838f 0%, #2f6fb0 50%, #7b4fc9 100%);
    padding: 1.4rem 1.8rem; border-radius: 18px; color: white;
    margin-bottom: 1.2rem; box-shadow: 0 6px 18px rgba(47,111,176,.35);
}
.hero h1 { margin: 0; font-size: 2rem; color: white; }
.hero p  { margin: .3rem 0 0 0; opacity: .92; font-size: .95rem; }

.card {
    background: #ffffff; color: #1f2d3d; border-radius: 16px;
    padding: 1rem 1.2rem; border-left: 8px solid var(--c);
    box-shadow: 0 3px 12px rgba(0,0,0,.12); height: 100%;
}
.card .label { font-size: .8rem; text-transform: uppercase;
               letter-spacing: .06em; color: #5b6b7c; }
.card .value { font-size: 2rem; font-weight: 800; color: var(--c);
               line-height: 1.2; }
.card .sub   { font-size: .8rem; color: #5b6b7c; }

.panel {
    background: #ffffff; color: #1f2d3d; border-radius: 16px;
    padding: .8rem 1rem .6rem; box-shadow: 0 3px 12px rgba(0,0,0,.12);
    text-align: center; height: 100%;
}
.panel .ptitle { font-size: .8rem; text-transform: uppercase;
                 letter-spacing: .06em; color: #5b6b7c; }
.panel svg { width: 100%; height: auto; display: block; }
.legend { display: flex; justify-content: center; gap: 1rem;
          font-size: .75rem; color: #5b6b7c; margin-top: .2rem; }
.legend i { display: inline-block; width: 10px; height: 10px;
            border-radius: 50%; margin-right: 4px; }

.section-title { font-size: 1.25rem; font-weight: 700; margin: 1.2rem 0 .4rem;
                 padding-left: .6rem; border-left: 6px solid var(--c); }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(0,131,143,.16),
                                         rgba(123,79,201,.16));
}
[data-testid="stSidebar"] h2 { color: #00838f; }

button[data-baseweb="tab"] { font-weight: 700; font-size: 1rem; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #2f6fb0; border-bottom: 3px solid #7b4fc9;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------- HTML / SVG helpers ----------------
def card(label, value, sub, color):
    return (f'<div class="card" style="--c:{color}">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def section(title, color):
    st.markdown(f'<div class="section-title" style="--c:{color}">{title}</div>',
                unsafe_allow_html=True)


def _pt(cx, cy, r, f):
    """Point on a semicircle; f=0 is the left end, f=1 the right end."""
    a = math.pi * (1 - f)
    return cx + r * math.cos(a), cy - r * math.sin(a)


def _arc(cx, cy, r, f0, f1):
    x0, y0 = _pt(cx, cy, r, f0)
    x1, y1 = _pt(cx, cy, r, f1)
    return f'M {x0:.2f} {y0:.2f} A {r} {r} 0 0 1 {x1:.2f} {y1:.2f}'


def gauge_needle(p, threshold, color, title):
    """Big semicircle gauge with green / amber / red zones and a needle."""
    cx, cy, r = 110, 105, 80
    zones = [(0, threshold / 2, GREEN), (threshold / 2, threshold, AMBER),
             (threshold, 1, RED)]
    arcs = ''.join(
        f'<path d="{_arc(cx, cy, r, a, b)}" stroke="{c}" stroke-width="18" '
        f'fill="none"/>' for a, b, c in zones)
    nx, ny = _pt(cx, cy, r - 14, p)
    tx0, ty0 = _pt(cx, cy, r - 12, threshold)
    tx1, ty1 = _pt(cx, cy, r + 14, threshold)
    svg = (
        f'<svg viewBox="0 0 220 135">{arcs}'
        f'<line x1="{tx0:.1f}" y1="{ty0:.1f}" x2="{tx1:.1f}" y2="{ty1:.1f}" '
        f'stroke="{INK}" stroke-width="3"/>'
        f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" '
        f'stroke="{INK}" stroke-width="4" stroke-linecap="round"/>'
        f'<circle cx="{cx}" cy="{cy}" r="8" fill="{INK}"/>'
        f'<text x="{cx}" y="{cy + 28}" text-anchor="middle" font-size="26" '
        f'font-weight="800" fill="{color}">{p:.0%}</text>'
        f'<text x="{cx - r}" y="{cy + 16}" text-anchor="middle" '
        f'font-size="9" fill="{MUTED}">0%</text>'
        f'<text x="{cx + r}" y="{cy + 16}" text-anchor="middle" '
        f'font-size="9" fill="{MUTED}">100%</text></svg>')
    legend = (f'<div class="legend"><span><i style="background:{GREEN}"></i>'
              f'Low</span><span><i style="background:{AMBER}"></i>Watch</span>'
              f'<span><i style="background:{RED}"></i>High</span>'
              f'<span>| threshold {threshold:.0%}</span></div>')
    return (f'<div class="panel"><div class="ptitle">{title}</div>'
            f'{svg}{legend}</div>')


def gauge_mini(frac, value_text, title, sub, color):
    """Small progress gauge: grey track + coloured fill."""
    frac = min(max(frac, 0.0), 1.0)
    cx, cy, r = 80, 72, 56
    fill = (f'<path d="{_arc(cx, cy, r, 0, frac)}" stroke="{color}" '
            f'stroke-width="12" stroke-linecap="round" fill="none"/>'
            if frac > 0.005 else '')
    svg = (f'<svg viewBox="0 0 160 95">'
           f'<path d="{_arc(cx, cy, r, 0, 1)}" stroke="{TRACK}" '
           f'stroke-width="12" stroke-linecap="round" fill="none"/>{fill}'
           f'<text x="{cx}" y="{cy - 2}" text-anchor="middle" font-size="20" '
           f'font-weight="800" fill="{color}">{value_text}</text>'
           f'<text x="{cx}" y="{cy + 15}" text-anchor="middle" '
           f'font-size="9" fill="{MUTED}">{sub}</text></svg>')
    return (f'<div class="panel"><div class="ptitle">{title}</div>{svg}</div>')


def _mix(c1, c2, t):
    return tuple(round(a + (b - a) * t) for a, b in zip(c1, c2))


def heat_style(col):
    """Green -> amber -> red cell colours for probabilities (no matplotlib)."""
    green, amber, red = (46, 158, 91), (245, 165, 36), (228, 87, 46)
    styles = []
    for v in col:
        v = min(max(float(v), 0.0), 1.0)
        rgb = (_mix(green, amber, v / 0.5) if v < 0.5
               else _mix(amber, red, (v - 0.5) / 0.5))
        text = '#ffffff' if v >= 0.6 else '#1f2d3d'
        styles.append(f'background-color: rgb{rgb}; color: {text}; '
                      f'font-weight: 600')
    return styles


def age_group(age):
    if age < 30:
        return 'Young adult'
    if age < 50:
        return 'Adult'
    if age < SENIOR_AGE:
        return 'Mature'
    return 'Senior'


# ---------------- Model ----------------
@st.cache_resource            # load once per server, not on every click
def load_artifacts():
    model = joblib.load('churn_model.joblib')
    with open('model_meta.json') as f:
        meta = json.load(f)
    return model, meta


def prepare_input(raw, columns):
    X = pd.get_dummies(raw[RAW_COLS])            # no drop_first at serving
    return X.reindex(columns=columns, fill_value=0).astype(float)


model, meta = load_artifacts()
COLS = meta['feature_columns']


def score(raw):
    return model.predict_proba(prepare_input(raw, COLS))[:, 1]


def band_of(p, thr):
    return 'HIGH' if p >= thr else 'WATCH' if p >= thr / 2 else 'LOW'


# ---------------- Sidebar: one customer ----------------
with st.sidebar:
    st.header('🧑 Customer profile')

    age = st.slider('Age (years)', 18, 90, 35)
    senior = age >= SENIOR_AGE
    st.caption(f'Age group: **{age_group(age)}**. The model only uses a '
               f'Senior citizen flag (age {SENIOR_AGE}+).')

    tenure = st.slider('Tenure (months)', 0, 72, 4)
    contract = st.selectbox('Contract',
                            ['Month-to-month', 'One year', 'Two year'])
    monthly = st.number_input('Monthly charges (USD)', 18.0, 120.0, 85.0,
                              step=0.5)
    internet = st.selectbox('Internet service', ['Fiber optic', 'DSL', 'No'])
    services = []
    if internet != 'No':
        services = st.multiselect('Add-on services', SERVICES,
                                  default=['StreamingTV'])
    phone = st.radio('Phone service', ['Yes', 'No'], horizontal=True)
    multi = 'No phone service'
    if phone == 'Yes':
        multi = st.radio('Multiple lines', ['No', 'Yes'], horizontal=True)
    payment = st.selectbox('Payment method', PAYMENTS)
    paperless = st.radio('Paperless billing', ['Yes', 'No'], horizontal=True)
    with st.expander('Demographics'):
        gender = st.radio('Gender', ['Female', 'Male'], horizontal=True)
        partner = st.checkbox('Has partner')
        dependents = st.checkbox('Has dependents')
    st.divider()
    threshold = st.slider('🎯 Contact threshold', 0.05, 0.95,
                          float(meta['threshold']), 0.05)

row = {'gender': gender, 'SeniorCitizen': int(senior),
       'Partner': 'Yes' if partner else 'No',
       'Dependents': 'Yes' if dependents else 'No', 'tenure': tenure,
       'PhoneService': phone, 'MultipleLines': multi,
       'InternetService': internet, 'Contract': contract,
       'PaperlessBilling': paperless, 'PaymentMethod': payment,
       'MonthlyCharges': monthly,
       'TotalCharges': tenure * monthly}           # derived, not typed
for s in SERVICES:
    row[s] = ('No internet service' if internet == 'No'
              else 'Yes' if s in services else 'No')
customer = pd.DataFrame([row])

# ---------------- Main area ----------------
st.markdown(
    f'<div class="hero"><h1>📡 Customer Churn Risk Advisor</h1>'
    f'<p>{meta["model_name"]} &nbsp;|&nbsp; '
    f'CV AUC {meta["cv_auc"]:.3f} &plusmn; {meta["cv_auc_std"]:.3f} '
    f'&nbsp;|&nbsp; Decision support only</p></div>',
    unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(['👤 One customer', '📂 Batch scoring', 'ℹ️ About'])

# ======================= TAB 1: one customer =======================
with tab1:
    p = float(score(customer)[0])
    band = band_of(p, threshold)
    color = BAND_COLOR[band]
    action = {'HIGH': 'Contact now', 'WATCH': 'Monitor',
              'LOW': 'No action'}[band]
    annual_value = monthly * 12
    at_risk = p * annual_value

    # --- Row 1: main gauge + headline cards
    g_col, k_col = st.columns([1.1, 1.9])
    g_col.markdown(gauge_needle(p, threshold, color, 'Churn probability'),
                   unsafe_allow_html=True)
    with k_col:
        k1, k2 = st.columns(2)
        k1.markdown(card('Risk band', band, 'based on your threshold', color),
                    unsafe_allow_html=True)
        k2.markdown(card('Action', action, 'recommended next step', BLUE),
                    unsafe_allow_html=True)
        st.write('')
        k3, k4 = st.columns(2)
        k3.markdown(card('Revenue at risk', f'${at_risk:,.0f}',
                         f'per year (${annual_value:,.0f} x {p:.0%})', RED),
                    unsafe_allow_html=True)
        gap = p - threshold
        k4.markdown(card('Distance to threshold', f'{gap:+.0%}',
                         'above = contact' if gap >= 0 else 'below = safe',
                         PURPLE),
                    unsafe_allow_html=True)

    # --- Row 2: mini gauges describing the profile
    section('📊 Customer health gauges', TEAL)
    m1, m2, m3, m4 = st.columns(4)
    ten_frac = tenure / 72
    ten_col = GREEN if tenure >= 24 else AMBER if tenure >= 12 else RED
    m1.markdown(gauge_mini(ten_frac, f'{tenure} mo', 'Tenure',
                           'of 72 months', ten_col), unsafe_allow_html=True)

    chg_frac = (monthly - 18) / (120 - 18)
    chg_col = GREEN if chg_frac < .35 else AMBER if chg_frac < .65 else RED
    m2.markdown(gauge_mini(chg_frac, f'${monthly:.0f}', 'Monthly bill',
                           'range $18 - $120', chg_col),
                unsafe_allow_html=True)

    lock = {'Month-to-month': 0.15, 'One year': 0.6, 'Two year': 1.0}[contract]
    lock_col = RED if lock < .3 else AMBER if lock < .8 else GREEN
    m3.markdown(gauge_mini(lock, contract.split('-')[0], 'Commitment',
                           'contract lock-in', lock_col),
                unsafe_allow_html=True)

    n_srv = len(services) if internet != 'No' else 0
    srv_col = RED if n_srv == 0 else AMBER if n_srv < 3 else GREEN
    m4.markdown(gauge_mini(n_srv / len(SERVICES), f'{n_srv} / 6',
                           'Add-on services', 'stickiness', srv_col),
                unsafe_allow_html=True)

    # --- Row 3: what-if scenarios
    section('🔀 What would change the risk?', TEAL)
    changes = [('Contract', c) for c in ['One year', 'Two year']
               if c != contract]
    if payment == 'Electronic check':
        changes.append(('PaymentMethod', 'Credit card (automatic)'))
    if internet != 'No' and 'TechSupport' not in services:
        changes.append(('TechSupport', 'Yes'))
    rows = []
    for feature, value in changes:
        alt = customer.copy()
        alt[feature] = value
        q = float(score(alt)[0])
        rows.append({'Change': f'{feature}: {value}',
                     'New probability': round(q, 3),
                     'Difference': round(q - p, 3)})
    if rows:
        table = pd.DataFrame(rows)

        def color_diff(col):
            return [f'color: {GREEN}; font-weight: 700' if v < 0 else
                    f'color: {RED}; font-weight: 700' if v > 0 else ''
                    for v in col]

        w1, w2 = st.columns([1.2, 1])
        w1.dataframe(
            table.style.apply(color_diff, subset=['Difference'])
                       .format({'New probability': '{:.3f}',
                                'Difference': '{:+.3f}'}),
            hide_index=True, use_container_width=True)
        compare = pd.DataFrame(
            {'Churn probability': [p] + [r['New probability'] for r in rows]},
            index=['Current'] + [r['Change'].split(': ')[1] for r in rows])
        w2.bar_chart(compare, color=BLUE, height=220)

        best = table.loc[table['Difference'].idxmin()]
        if best['Difference'] < 0:
            saved = -best['Difference'] * annual_value
            st.success(f"Best lever: **{best['Change']}** cuts risk by "
                       f"{-best['Difference']:.1%} "
                       f"(about ${saved:,.0f} of yearly revenue protected).")
    st.caption('Green = lower risk, red = higher risk. Associations learned '
               'from data, not guaranteed effects.')

    # --- Row 4: explanation
    final = model[-1] if hasattr(model, 'steps') else model
    if hasattr(model, 'steps') and hasattr(final, 'coef_'):
        section('🔍 Why this score? (Logistic Regression)', PURPLE)
        with st.expander('Show the eight biggest contributions', expanded=True):
            z = model[:-1].transform(prepare_input(customer, COLS))[0]
            contrib = pd.Series(final.coef_[0] * z, index=COLS)
            top = contrib.reindex(
                contrib.abs().sort_values(ascending=False).index).head(8)
            chart = pd.DataFrame({
                'Pushes toward churn': top.clip(lower=0),
                'Pushes away from churn': top.clip(upper=0)})
            st.bar_chart(chart, color=[RED, GREEN])
            st.caption('Red bars push toward churn, green bars push away.')

# ======================= TAB 2: batch =======================
with tab2:
    section('📂 Score a customer file', BLUE)
    st.write('Upload a CSV with the original Telco columns.')
    file = st.file_uploader('Customer file', type='csv')
    if file is not None:
        data = pd.read_csv(file)
        missing = [c for c in RAW_COLS if c not in data.columns]
        if missing:
            st.error(f'Missing columns: {missing}')
            st.stop()
        data['TotalCharges'] = pd.to_numeric(
            data['TotalCharges'], errors='coerce').fillna(0)
        data['p_churn'] = score(data).round(3)
        data['band'] = np.where(data['p_churn'] >= threshold, 'HIGH',
                                np.where(data['p_churn'] >= threshold / 2,
                                         'WATCH', 'LOW'))
        data['contact'] = np.where(data['band'] == 'HIGH', 'Yes', '')

        age_pref = st.radio('Age preference', ['All customers',
                                               f'Seniors ({SENIOR_AGE}+) only',
                                               'Non-seniors only'],
                            horizontal=True)
        if age_pref.startswith('Seniors'):
            view = data[data['SeniorCitizen'] == 1]
        elif age_pref.startswith('Non-seniors'):
            view = data[data['SeniorCitizen'] == 0]
        else:
            view = data

        if view.empty:
            st.warning('No customers match this age preference.')
        else:
            n = len(view)
            n_hi = int((view['band'] == 'HIGH').sum())
            avg = float(view['p_churn'].mean())
            rev_risk = float((view['p_churn'] * view['MonthlyCharges'] * 12)
                             [view['band'] == 'HIGH'].sum())

            b1, b2, b3, b4 = st.columns(4)
            b1.markdown(card('Customers shown', f'{n}',
                             f'of {len(data)} uploaded', BLUE),
                        unsafe_allow_html=True)
            b2.markdown(card('Above threshold', f'{n_hi}',
                             f'{n_hi / n:.0%} of shown', RED),
                        unsafe_allow_html=True)
            b3.markdown(card('Average risk', f'{avg:.0%}',
                             'mean churn probability', AMBER),
                        unsafe_allow_html=True)
            b4.markdown(card('Revenue at risk', f'${rev_risk:,.0f}',
                             'per year, high-risk group', PURPLE),
                        unsafe_allow_html=True)

            # Gauges for the whole book
            section('🎛️ Portfolio gauges', TEAL)
            q1, q2, q3 = st.columns(3)
            q1.markdown(gauge_needle(avg, threshold, BAND_COLOR[band_of(
                avg, threshold)], 'Average churn risk'),
                unsafe_allow_html=True)
            share_hi = n_hi / n
            q2.markdown(gauge_mini(share_hi, f'{share_hi:.0%}',
                                   'Share to contact', f'{n_hi} customers',
                                   RED if share_hi > .3 else AMBER
                                   if share_hi > .15 else GREEN),
                        unsafe_allow_html=True)
            m2m = float((view['Contract'] == 'Month-to-month').mean())
            q3.markdown(gauge_mini(m2m, f'{m2m:.0%}', 'Month-to-month',
                                   'share of customers',
                                   RED if m2m > .5 else AMBER
                                   if m2m > .3 else GREEN),
                        unsafe_allow_html=True)

            # Distribution charts
            section('📈 Risk distribution', BLUE)
            d1, d2, d3 = st.columns(3)
            counts = (view['band'].value_counts()
                      .reindex(['LOW', 'WATCH', 'HIGH']).fillna(0))
            d1.caption('Customers per risk band')
            d1.bar_chart(pd.DataFrame({'Customers': counts}), height=240,
                         color=BLUE)
            hist, edges = np.histogram(view['p_churn'], bins=10,
                                       range=(0, 1))
            labels = [f'{int(a * 100)}-{int(b * 100)}%'
                      for a, b in zip(edges[:-1], edges[1:])]
            d2.caption('Churn probability histogram')
            d2.bar_chart(pd.DataFrame({'Customers': hist}, index=labels),
                         height=240, color=PURPLE)
            d3.caption('Average risk by contract')
            d3.bar_chart(view.groupby('Contract')['p_churn'].mean()
                         .rename('Avg risk'), height=240, color=TEAL)

            e1, e2 = st.columns(2)
            e1.caption('Average risk by internet service')
            e1.bar_chart(view.groupby('InternetService')['p_churn'].mean()
                         .rename('Avg risk'), height=220, color=AMBER)
            e2.caption('Average risk by payment method')
            e2.bar_chart(view.groupby('PaymentMethod')['p_churn'].mean()
                         .rename('Avg risk'), height=220, color=RED)

            section('📋 Highest-risk customers (top 50)', RED)
            ranked = view.sort_values('p_churn', ascending=False).head(50)
            st.dataframe(
                ranked.style.apply(heat_style, subset=['p_churn']),
                use_container_width=True)
        st.download_button('⬇️ Download scored CSV', data.to_csv(index=False),
                           'scored_customers.csv', 'text/csv')

# ======================= TAB 3: about =======================
with tab3:
    section('ℹ️ Model details', PURPLE)
    a1, a2, a3 = st.columns(3)
    a1.markdown(gauge_mini(float(meta['cv_auc']), f'{meta["cv_auc"]:.3f}',
                           'CV AUC', 'ranking quality', TEAL),
                unsafe_allow_html=True)
    a2.markdown(gauge_mini(float(meta['threshold']),
                           f'{float(meta["threshold"]):.0%}',
                           'Default threshold', 'contact cut-off', AMBER),
                unsafe_allow_html=True)
    a3.markdown(gauge_mini(min(float(meta['cv_auc_std']) * 10, 1),
                           f'&plusmn;{meta["cv_auc_std"]:.3f}',
                           'AUC spread', 'lower = more stable', PURPLE),
                unsafe_allow_html=True)
    st.write('')
    st.json({k: v for k, v in meta.items() if k != 'feature_columns'})
    st.markdown('**Limitations:** trained on one US telecom dataset; '
                'not validated for other markets. See the model card.')
    st.markdown(f'**About age:** the training data has no exact age column, '
                f'only a Senior citizen flag. The age slider converts age '
                f'{SENIOR_AGE}+ to that flag, so ages below and above the '
                f'cutoff score the same.')
