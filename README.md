# rent-estimator-ml

Estima aluguel mensal em BRL usando área, quartos, localização e características do imóvel em um conjunto sintético.

## Dados

O pipeline gera dados sintéticos determinísticos localmente, identificados como sintéticos em `pipeline.py`. Nenhum dado privado ou download externo é necessário. A divisão e a semente de avaliação são descritas junto às métricas abaixo.

## Instalação e execução

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib
```

## Métricas obtidas

Seed 42: holdout aleatório de 550 imóveis — MAE **R$ 517,51**, RMSE **R$ 648,05**, R² **0,9175**. O gerador usa relações artificiais entre cidade, área e aluguel; os erros não são estimativas para o mercado imobiliário brasileiro.

Limitação: dados e regras de geração simplificados; antes de uso real, substitua-os por dados com licença e proveniência adequadas e valide fora da amostra.
