# FTSEMIB Stock Analysis and Forecasting

Questo progetto implementa un algoritmo di analisi e previsione per le azioni dell'indice FTSEMIB (Borsa Italiana), utilizzando indicatori macroeconomici, indicatori tecnici e modelli di previsione. L'obiettivo è sviluppare un modello che combina analisi tecnica, analisi fondamentale e previsioni di volatilità per prendere decisioni di investimento.

## Funzionalità

- **Scaricamento dei dati finanziari**: Il codice scarica i dati di prezzo mensili per le azioni appartenenti all'indice FTSEMIB tramite `yfinance`.
- **Indicatori Macroeconomici**: Utilizza dati macroeconomici scaricati tramite FRED (Federal Reserve Economic Data).
- **Analisi tecnica**: Calcola indicatori tecnici come SMA (Moving Averages), RSI (Relative Strength Index), e volatilità.
- **Test di Granger**: Applica il test di causalità di Granger per identificare le relazioni tra variabili macroeconomiche e rendimenti azionari.
- **Previsioni tramite Random Forest**: Utilizza un modello Random Forest per prevedere i rendimenti azionari.
- **Volatilità GARCH**: Calcola la volatilità futura utilizzando modelli GARCH (Generalized Autoregressive Conditional Heteroskedasticity).
- **Backtest**: Esegue un backtest rolling basato su segnali derivanti dalla previsione dei rendimenti e dalla volatilità.

## Requisiti

- `pandas`
- `numpy`
- `matplotlib`
- `sklearn`
- `yfinance`
- `statsmodels`
- `arch`
- `fredapi`
- `datetime`

Puoi installare le dipendenze richieste con:
```bash
pip install pandas numpy matplotlib sklearn yfinance statsmodels arch fredapi

