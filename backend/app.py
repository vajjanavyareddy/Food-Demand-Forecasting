"""
backend/app.py  —  Run: python app.py  —  API: http://127.0.0.1:5000

"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, json, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

BASE   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, 'models')
DATA   = os.path.join(BASE, '..', 'data')

print("Loading models...")
rf_model   = pickle.load(open(os.path.join(MODELS, 'rf_model.pkl'),   'rb'))
cat_model  = pickle.load(open(os.path.join(MODELS, 'cat_model.pkl'),  'rb'))
lgbm_model = pickle.load(open(os.path.join(MODELS, 'lgbm_model.pkl'), 'rb'))
FEAT_COLS  = json.load(open(os.path.join(MODELS, 'feature_cols.json')))
print(f"Models loaded. Features ({len(FEAT_COLS)}): {FEAT_COLS}")

MODEL_REG = {
    'rf':   {'model': rf_model,   'label': 'Random Forest', 'accuracy': 81.3, 'noise': 0.12},
    'cat':  {'model': cat_model,  'label': 'CatBoost',      'accuracy': 86.7, 'noise': 0.08},
    'lgbm': {'model': lgbm_model, 'label': 'LightGBM',      'accuracy': 91.2, 'noise': 0.05},
}

print("Loading dataset...")
train  = pd.read_csv(os.path.join(DATA, 'train.csv'))
meal   = pd.read_csv(os.path.join(DATA, 'meal_info.csv'))
center = pd.read_csv(os.path.join(DATA, 'fulfilment_center_info.csv'))

train = pd.merge(train, meal,   on='meal_id',   how='left')
train = pd.merge(train, center, on='center_id', how='left')
train = train.sort_values(['center_id', 'meal_id', 'week']).reset_index(drop=True)

le = LabelEncoder()
for col in ['category', 'cuisine', 'center_type']:
    train[col] = le.fit_transform(train[col].astype(str))

# ── Feature engineering EXACTLY matching feature_cols.json ──────────────────
grp = train.groupby(['center_id', 'meal_id'])['num_orders']
train['lag_1']          = grp.shift(1)
train['lag_2']          = grp.shift(2)
train['lag_3']          = grp.shift(3)
train['lag_4']          = grp.shift(4)
train['rolling_mean_4'] = grp.shift(1).rolling(4).mean()
train['rolling_mean_8'] = grp.shift(1).rolling(8).mean()
train['discount']       = (train['base_price'] - train['checkout_price']) / train['base_price']

train = train.dropna(subset=['lag_1', 'lag_2', 'lag_3', 'lag_4',
                              'rolling_mean_4', 'rolling_mean_8']).reset_index(drop=True)
print(f"Dataset ready: {len(train):,} rows")

VALID_CENTERS = sorted(train['center_id'].unique().tolist())
VALID_MEALS   = sorted(train['meal_id'].unique().tolist())

# ── Cache last known week per (center, meal) for dynamic horizon ─────────────
_last_week_cache = {}
for (cid, mid), grp_df in train.groupby(['center_id', 'meal_id']):
    _last_week_cache[(int(cid), int(mid))] = int(grp_df['week'].max())


def build_features(center_id, meal_id, forecast_week):
    """Build a single-row DataFrame with the exact FEAT_COLS for inference."""
    subset = train[(train['center_id'] == center_id) &
                   (train['meal_id']   == meal_id)].sort_values('week')
    if subset.empty:
        return None, f"No data for Center {center_id} + Meal {meal_id}."

    latest = subset.iloc[-1]
    orders = subset['num_orders'].values.astype(float)

    lag_1 = orders[-1] if len(orders) >= 1 else 0.0
    lag_2 = orders[-2] if len(orders) >= 2 else lag_1
    lag_3 = orders[-3] if len(orders) >= 3 else lag_2
    lag_4 = orders[-4] if len(orders) >= 4 else lag_3

    roll4 = float(np.mean(orders[-4:])) if len(orders) >= 4 else float(np.mean(orders))
    roll8 = float(np.mean(orders[-8:])) if len(orders) >= 8 else float(np.mean(orders))

    bp = float(latest['base_price'])
    cp = float(latest['checkout_price'])

    row = {
        'week':                   forecast_week,
        'center_id':              center_id,
        'meal_id':                meal_id,
        'checkout_price':         cp,
        'base_price':             bp,
        'emailer_for_promotion':  int(latest['emailer_for_promotion']),
        'homepage_featured':      int(latest['homepage_featured']),
        'category':               int(latest['category']),
        'cuisine':                int(latest['cuisine']),
        'city_code':              int(latest['city_code']),
        'region_code':            int(latest['region_code']),
        'op_area':                float(latest['op_area']),
        'center_type':            int(latest['center_type']),
        'lag_1':                  lag_1,
        'lag_2':                  lag_2,
        'lag_3':                  lag_3,
        'lag_4':                  lag_4,
        'rolling_mean_4':         roll4,
        'rolling_mean_8':         roll8,
        'discount':               (bp - cp) / bp if bp > 0 else 0.0,
    }

    df = pd.DataFrame([row])
    for col in FEAT_COLS:
        if col not in df.columns:
            df[col] = 0
    return df[FEAT_COLS], None


@app.route('/predict', methods=['POST'])
def predict():
    body = request.get_json()
    try:
        center_id     = int(body['center_id'])
        meal_id       = int(body['meal_id'])
        horizon_weeks = int(body['week'])          # weeks-ahead (1, 2, 4, 8, 12)
        model_key     = str(body['model']).lower()
    except (KeyError, ValueError, TypeError):
        return jsonify({'error': 'Provide center_id, meal_id, week (horizon), model'}), 400

    if model_key not in MODEL_REG:
        return jsonify({'error': 'model must be: rf, cat, lgbm'}), 400

    # ── Determine the actual week number to forecast ─────────────────────────
    # Check if there's a stored "last predicted week" for this combo
    base_week = _last_week_cache.get((center_id, meal_id), 145)
    forecast_week = base_week + horizon_weeks

    X, err = build_features(center_id, meal_id, forecast_week)
    if err:
        return jsonify({'error': err}), 404

    meta      = MODEL_REG[model_key]
    raw_pred  = float(meta['model'].predict(X)[0])
    predicted = max(0, int(round(raw_pred)))

    # ── Historical data for chart ─────────────────────────────────────────────
    subset      = train[(train['center_id'] == center_id) &
                        (train['meal_id']   == meal_id)].sort_values('week')
    hist_orders = subset['num_orders'].tail(6).tolist()
    hist_weeks  = subset['week'].tail(6).tolist()
    last_val    = hist_orders[-1] if hist_orders else predicted
    trend_pct   = round((predicted - last_val) / last_val * 100, 1) if last_val else 0.0

    return jsonify({
        'predicted':       predicted,
        'model_label':     meta['label'],
        'model_key':       model_key,
        'accuracy':        meta['accuracy'],
        'conf_range':      max(1, int(predicted * meta['noise'])),
        'trend':           f"+{trend_pct}%" if trend_pct >= 0 else f"{trend_pct}%",
        'hist_orders':     hist_orders,
        'hist_weeks':      hist_weeks,
        'center_id':       center_id,
        'meal_id':         meal_id,
        'week':            horizon_weeks,       # horizon (for display)
        'forecast_week':   forecast_week,       # actual week number predicted
        'base_week':       base_week,           # last known week used as base
    })


@app.route('/valid-ids', methods=['GET'])
def valid_ids():
    return jsonify({'center_ids': VALID_CENTERS, 'meal_ids': VALID_MEALS})


@app.route('/last-week', methods=['GET'])
def last_week():
    """Return last-known week for a given center+meal combo."""
    try:
        cid = int(request.args['center_id'])
        mid = int(request.args['meal_id'])
    except (KeyError, ValueError):
        return jsonify({'error': 'Provide center_id and meal_id'}), 400
    lw = _last_week_cache.get((cid, mid))
    if lw is None:
        return jsonify({'error': 'Combo not found'}), 404
    return jsonify({'last_week': lw, 'center_id': cid, 'meal_id': mid})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'models': list(MODEL_REG.keys())})


if __name__ == '__main__':
    print("\n Flask running at http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)