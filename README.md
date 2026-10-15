# rent-estimator-ml

Estimates monthly rent in Brazilian reais (BRL) from property size, bedrooms, location, and other features.

## Data

Deterministic synthetic property data is generated locally in `pipeline.py`. No private data or external download is required.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib
```

## Evaluation results

With seed 42, a random holdout of 550 properties had an MAE of BRL 517.51, an RMSE of BRL 648.05, and an R² of 0.9175. The generator uses artificial relationships between city, floor area, and rent; these errors do not describe the Brazilian rental market.

## Limitations

The synthetic property data is deliberately simple. Real estimates would require appropriately sourced listings, careful handling of location and listing dates, and validation on a separate market sample.
