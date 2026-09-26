# Baku 2026 winner likelihood

This is a hobby project. I am a Ferrari and Formula 1 fan, and I built it for fun after watching Mariana Antaya's race predictions.

The model is inspired by those predictions and by her 2026 scripts:

https://github.com/mar-antaya/2026_f1_predictions

This version trains an XGBoost regressor on the 2026 season only, using FastF1, then runs 25,000 simulated races for the Azerbaijan Grand Prix. It is a personal experiment, not a betting tool and not an official prediction.

## What it uses

- 2026 rounds 1–15 only. 2024 and 2025 are not used.
- Qualifying position and the gap to pole
- Practice pace, recent driver and team form, pit stops, and how hard it is to pass at each circuit
- Qualifying weather: air temperature and wet or dry
- An XGBoost regressor that predicts a finishing position

Rounds 1–10 train a test model. Rounds 11–14 check it. The final model trains on rounds 1–14 and predicts Baku, round 15.

## Requirements

- Python 3.11 or newer
- The packages in `requirements.txt`:

```
fastf1>=3.4,<4
xgboost>=2.1,<4
scikit-learn>=1.4,<2
pandas>=2.2
numpy>=1.26
requests>=2.31
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python predict_baku_like_mar.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The script needs a network connection the first time it asks FastF1 for the 2026 calendar. Race data for rounds 1–15 is already saved in `cache/weekends_2026.json`, so the table below should match if you keep that file.

Do not upload or copy the `.venv` folder. Each machine should create its own.

## Result

Qualifying weather used for Baku: 24.5°C, dry. 25,000 simulations.

| Place | Driver | Team | Qualifying | Simulations won | Likelihood |
| --- | --- | --- | --- | --- | --- |
| 1 | George Russell | Mercedes | P1 | 9,307 | 37.2% |
| 2 | Max Verstappen | Red Bull Racing | P8 | 5,103 | 20.4% |
| 3 | Isack Hadjar | Red Bull Racing | P4 | 3,012 | 12.0% |
| 4 | Charles Leclerc | Ferrari | P2 | 1,754 | 7.0% |
| 5 | Lando Norris | McLaren | P5 | 1,591 | 6.4% |
| 6 | Oscar Piastri | McLaren | P3 | 1,297 | 5.2% |
| 7 | Lewis Hamilton | Ferrari | P6 | 823 | 3.3% |
| 8 | Kimi Antonelli | Mercedes | P16 | 502 | 2.0% |
| 9 | Pierre Gasly | Alpine | P7 | 474 | 1.9% |
| 10 | Franco Colapinto | Alpine | P10 | 392 | 1.6% |
| 11 | Liam Lawson | Racing Bulls | P12 | 188 | 0.8% |
| 12 | Arvid Lindblad | Racing Bulls | P15 | 139 | 0.6% |
| 13 | Oliver Bearman | Haas F1 Team | P11 | 110 | 0.4% |
| 14 | Esteban Ocon | Haas F1 Team | P14 | 83 | 0.3% |
| 15 | Alexander Albon | Williams | P13 | 78 | 0.3% |
| 16 | Carlos Sainz | Williams | P9 | 74 | 0.3% |
| 17 | Gabriel Bortoleto | Audi | P17 | 26 | 0.1% |
| 18 | Nico Hulkenberg | Audi | P18 | 20 | 0.1% |
| 19 | Fernando Alonso | Aston Martin | P19 | 14 | 0.1% |
| 20 | Sergio Perez | Cadillac | P20 | 7 | 0.0% |
| 21 | Lance Stroll | Aston Martin | P21 | 3 | 0.0% |
| 22 | Valtteri Bottas | Cadillac | P22 | 3 | 0.0% |

Highest likelihood: George Russell (Mercedes), 9,307 of 25,000 simulations (37.2%).

The same table is saved to `baku_likelihoods.csv`. Training rows go to `training_2026.csv`. The fitted model is saved as `baku_2026_regressor.json`.

These percentages are the model's count of simulated wins. They are not a promise of the race result.
