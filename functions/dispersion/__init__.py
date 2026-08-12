# scaffolding repo de test — NE PAS recopier dans le vrai repo.
# Shim vide : le vrai repo re-exporte ici optimize/backtest/solve/price et les
# modèles publics (la page fait `from functions.dispersion import optimize, ...`).
# Ce shim reste vide pour ne pas deviner la surface réelle ; les tests
# importent les sous-modules directement (functions.dispersion.models, etc.).
