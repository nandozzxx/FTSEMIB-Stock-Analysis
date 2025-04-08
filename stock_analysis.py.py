import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import grangercausalitytests
import yfinance as yf
from fredapi import Fred
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# PARAMETRI E CONFIGURAZIONE
# ---------------------------
START_DATE = '2010-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')
MIN_TRAIN_SIZE = 36  # minimo numero di mesi per il training nella finestra mobile

# Lista dei ticker FTSEMIB 
tickers = ['BMPS.MI', 'TEN.MI', 'LDO.MI', 'SPM.MI', 'BAMI.MI', 'BPSO.MI', 'FBK.MI', 'G.MI', 'IP.MI', 
           'BC.MI', 'AZM.MI', 'A2A.MI', 'IVG.MI', 'RACE.MI', 'PST.MI', 'PIRC.MI', 'DIA.MI', 'PRY.MI', 
           'SRG.MI', 'STM', 'UNI.MI', 'HER.MI', 'REC.MI', 'TRN.MI', 'BPE.MI', 'ENEL.MI', 'NEXI.MI', 
           'TIT.MI', 'ERG.MI', 'IG.MI', 'BMED.MI', 'CPR.MI', 'MONC.MI', 'INW.MI', 'STLAM.MI', 
           'ENI.MI', 'AMP.MI', 'MB.MI', 'ISP.MI', 'UCG.MI']

# Indicatori macroeconomici da scaricare da FRED
indicatori = {
    'PIL': 'CLVMEURSCAB1GQEA19',
    'Tasso_Disoccupazione': 'LRHUTTTTEZM156S',
    'Produzione_Industriale': 'PRMNTO01EZQ661S',
    'Vendite_Dettaglio': 'EA19SLRTTO02IXOBSAM',
    'Fiducia_Consumatori': 'CSCICP03EZM665S',
    'Tassi_Interesse': 'DFF',
    'Tasso_Cambio_USD_Euro': 'DEXUSEU',
    'Prezzo_Petrolio_Brent': 'DCOILBRENTEU',
    'Prezzo_Gas': 'PNGASEUUSDM',
    'Domanda_Energia': 'DNRGRC1M027SBEA',
    'Prezzo_rame': 'PCOPPUSDM',
    'permessi_costruzioni': 'EA19ODCNPI03GPSAM',
    'domanda_costruzioni': 'EA19PRCNTO01GYSAM',
    'Consumi_beni_primari': '00XEFDEZ19M086NEST',
    'Salute': 'DHLCRX1Q020SBEA',
    'Tassi_Interesse_reali': 'BAMLHE00EHYIEY',
    '%_risparmio': 'PSAVERT',
    'Bilancia_Commerciale': 'BOPGSTB',
    'Inflazione': 'TOTNRGFOODEA20MI15XM'
}

# ---------------------------
# FUNZIONI DI PREPROCESSING
# ---------------------------

def download_indicators(indicators, start_date=START_DATE, end_date=END_DATE):
    """Scarica indicatori da FRED, campionando a frequenza mensile."""
    fred = Fred(api_key='9c7691693c553612eec00fc45b8bc86f')
    data = {}
    for nome, codice in indicators.items():
        try:
            serie = fred.get_series(codice, start_date, end_date)
            data[nome] = serie
        except Exception as e:
            print(f"Errore nel download di {nome}: {e}")
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    df = df.resample('MS').last() 
    return df.ffill()

def get_monthly_returns(ticker):
    """Scarica i dati mensili da yfinance e calcola il rendimento mensile."""
    try:
        data = yf.download(ticker, start=START_DATE, end=END_DATE, interval='1mo', 
                           auto_adjust=True, progress=False)
        data['Monthly_Return'] = (data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1)
        return data['Monthly_Return'].dropna()
    except Exception as e:
        print(f"Errore ottenendo i rendimenti per {ticker}: {e}")
        return None

def compute_RSI(series, window=14):
    """Calcola l'indicatore di forza relativa (RSI)."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=window, min_periods=window).mean()
    ma_down = down.rolling(window=window, min_periods=window).mean()
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    return rsi

def get_technical_indicators(ticker):
    """Scarica dati giornalieri e calcola alcuni indicatori tecnici, campionandoli mensilmente."""
    try:
        data = yf.download(ticker, start=START_DATE, end=END_DATE, interval='1d', 
                           auto_adjust=True, progress=False)
        data['SMA_50'] = data['Close'].rolling(window=50, min_periods=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200, min_periods=200).mean()
        data['Volatility'] = data['Close'].pct_change().rolling(window=21).std()  # volatilità mensile
        data['RSI_14'] = compute_RSI(data['Close'], window=14)
        tech_features = data[['SMA_50', 'SMA_200', 'Volatility', 'RSI_14']].resample('MS').last()
        return tech_features
    except Exception as e:
        print(f"Errore indicatori tecnici per {ticker}: {e}")
        return pd.DataFrame()

def test_granger_causality(df, returns, max_lag=4, significance_level=0.05):
    """Esegue il test di Granger tra ciascun indicatore e i rendimenti."""
    results = {}
    for col in df.columns:
        data = pd.concat([df[col], returns], axis=1).dropna()
        if len(data) <= max_lag:
            continue
        try:
            test_result = grangercausalitytests(data, max_lag, verbose=False)
            best_p_value = float('inf')
            best_lag = None
            for lag in range(1, max_lag + 1):
                p_value = test_result[lag][0]['ssr_chi2test'][1]
                if p_value < significance_level and p_value < best_p_value:
                    best_p_value = p_value
                    best_lag = lag
            if best_lag is not None:
                results[col] = {'lag': best_lag, 'p_value': best_p_value}
        except Exception as e:
            continue
    return results

def create_features_df(granger_results, df):
    """Crea feature aggiuntive shiftate in base al test di Granger."""
    features_df = pd.DataFrame(index=df.index)
    for col, result in granger_results.items():
        lag = result['lag']
        features_df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return features_df

df_indicators = download_indicators(indicatori)

# ---------------------------
# FUNZIONI PER IL CALCOLO DELLA VOLATILITÀ GARCH
# ---------------------------
def calculate_garch_forecast(ticker, train_end_date):
    """
    Calcola la volatilità mensile prevista utilizzando un modello GARCH(1,1)
    sui rendimenti giornalieri (utilizzando 1 anno di dati precedenti).
    Restituisce la volatilità prevista (float) per il mese successivo.
    """
    try:
        train_end = pd.to_datetime(train_end_date)
        start_for_garch = (train_end - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
        daily_data = yf.download(
            ticker, 
            start=start_for_garch, 
            end=train_end_date, 
            interval='1d', 
            auto_adjust=True, 
            progress=False
        )
        daily_returns = daily_data['Close'].pct_change().dropna()
        if len(daily_returns) < 30:
            return np.nan
        # Creazione del modello GARCH: rendimenti in percentuale
        model = arch_model(daily_returns * 100, vol='Garch', p=1, q=1, rescale=False)
        model_fit = model.fit(disp="off")
        forecast = model_fit.forecast(horizon=21)
        # Estrai la varianza prevista per l'ultimo giorno del forecast
        predicted_variance = forecast.variance.iloc[-1, -1]
        monthly_vol = (np.sqrt(predicted_variance) / 100) * np.sqrt(21)
        # Forza la conversione in float
        monthly_vol = float(monthly_vol)
        return monthly_vol
    except Exception as e:
        return np.nan

# ---------------------------
# BACKTEST ROLLING CON STRATEGIA LONG/SHORT
# ---------------------------

def backtest_ticker(ticker):
    """
    Per il ticker indicato:
     - Scarica rendimenti mensili e indicatori tecnici.
     - Applica il test di Granger sugli indicatori macro.
     - Crea un DataFrame 'data' con tutte le feature e la variabile 'Return'.
     - Per ogni periodo rolling (minimo MIN_TRAIN_SIZE mesi di training) addestra il modello tramite GridSearchCV,
       effettua la previsione per il mese successivo e stima la volatilità attesa tramite GARCH.
     - Combina la previsione (70%) e la volatilità stimata (30%) per ottenere un segnale combinato.
       Il segnale determina se assumere una posizione long (se positivo) o short (se negativo).
     - Aggiorna il portafoglio investendo il 100% del capitale in base al segnale.
    """
    # --- Preparazione dati ---
    monthly_returns = get_monthly_returns(ticker)
    if monthly_returns is None or len(monthly_returns) < MIN_TRAIN_SIZE:
        print(f"Dati insufficienti per {ticker}")
        return None

    tech_indicators = get_technical_indicators(ticker)
    df_features_full = pd.concat([df_indicators, tech_indicators], axis=1)
    
    granger_results = test_granger_causality(df_indicators, monthly_returns)
    if not granger_results:
        print(f"Nessuna causalità di Granger significativa per {ticker}")
        return None
    features_granger = create_features_df(granger_results, df_indicators)
    
    features = pd.concat([features_granger, tech_indicators], axis=1)
    data = pd.concat([features, monthly_returns.rename('Return')], axis=1).dropna()
    if len(data) < MIN_TRAIN_SIZE:
        print(f"Dati allineati insufficienti per {ticker}")
        return None

    data = data.sort_index()
    data.columns = data.columns.astype(str)
    
    # --- Inizializzazione portafoglio e lista delle predizioni ---
    portfolio = pd.DataFrame(index=data.index, columns=['Strategy_Return'])
    portfolio['Strategy_Return'] = 0
    predictions = []
    
    # Rolling backtest
    for i in range(MIN_TRAIN_SIZE, len(data)-1):
        train = data.iloc[:i]
        test = data.iloc[i:i+1]
        
        X_train = train.drop(columns=['Return'])
        y_train = train['Return']
        X_test = test.drop(columns=['Return'])
        
        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)
        
        # Standardizzazione delle feature
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Ottimizziamo il modello con GridSearchCV
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, None],
            'min_samples_split': [2, 5]
        }
        rf = RandomForestRegressor(random_state=42)
        grid = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_
        
        # Previsione del rendimento per il mese successivo
        pred = best_model.predict(X_test_scaled)[0]
        predictions.append(pred)
        
        # --- Calcolo della volatilità attesa tramite GARCH ---
        train_end_date = train.index[-1].strftime('%Y-%m-%d')
        vol_forecast = calculate_garch_forecast(ticker, train_end_date)
        # Se non è stato calcolato correttamente, lo sostituiamo con il valore medio della volatilità dei rendimenti giornalieri
        try:
            daily_data = yf.download(
                ticker, 
                start=(pd.to_datetime(train_end_date) - pd.DateOffset(years=1)).strftime('%Y-%m-%d'),
                end=train_end_date, 
                interval='1d', 
                auto_adjust=True, 
                progress=False
            )
            train_daily_returns = daily_data['Close'].pct_change().dropna()
            avg_vol = float(train_daily_returns.std() * np.sqrt(21))
        except:
            avg_vol = 0.01
        if vol_forecast is None or np.isnan(vol_forecast) or avg_vol == 0:
            print(f"Skipped due to invalid vol forecast for {ticker}")
        return None

        
        # --- Calcolo del segnale combinato ---
        # Se la previsione è positiva, riduciamo l'esposizione se la volatilità è elevata.
        # Se la previsione è negativa, aumenta l'esposizione short se la volatilità è elevata.
        if pred >= 0:
            combined_signal = 0.7 * pred - 0.3 * normalized_vol
        else:
            combined_signal = 0.7 * pred + 0.3 * normalized_vol
        # Decidiamo la posizione: se il segnale combinato è positivo, posizione long (1), se negativo, short (-1).
        position = 1 if combined_signal > 0 else -1 if combined_signal < 0 else 0
        
        # La strategia utilizza il 100% del capitale: il rendimento del mese è il rendimento reale moltiplicato per la posizione
        trade_return = test['Return'].iloc[0] * position
        portfolio.iloc[i+1, portfolio.columns.get_loc('Strategy_Return')] = trade_return

    # Calcolo rendimento cumulato della strategia
    portfolio['Cumulative_Return'] = (1 + portfolio['Strategy_Return'].fillna(0)).cumprod()
    market = (1 + data['Return']).cumprod()
    
    # Plot di confronto tra strategia e benchmark
    plt.figure(figsize=(10,6))
    plt.plot(portfolio.index, portfolio['Cumulative_Return'], label='Strategia Long/Short')
    plt.plot(market.index, market, label='Benchmark (Mercato)')
    plt.title(f"Backtest Rolling per {ticker}")
    plt.xlabel("Data")
    plt.ylabel("Rendimento cumulato")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return {
        'portfolio': portfolio,
        'market': market,
        'predictions': predictions
    }

# ---------------------------
# ESECUZIONE DEL BACKTEST SU OGNI TICKER
# ---------------------------
results = {}
for ticker in tickers:
    print(f"\nEsecuzione backtest per {ticker}")
    res = backtest_ticker(ticker)
    if res is not None:
        results[ticker] = res

print("Backtest completato!")
