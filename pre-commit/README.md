# Pre-commit Configuration

Ce dossier contient tous les fichiers de configuration et scripts pour le système pre-commit.

## Structure

```
pre-commit/
├── README.md                    # Ce fichier
├── .pre-commit-config.yaml      # Configuration active
├── hooks/
│   └── pre-commit              # Hook git actif
└── scripts/
    └── help-message.sh         # Script d'aide en cas d'erreur
```

## Fonctionnement

Le pre-commit s'exécute automatiquement avant chaque `git commit` et vérifie :

1. **Black** : Formatage automatique du code Python
2. **isort** : Tri des imports
3. **flake8** : Vérification du style de code
4. **Hooks basiques** : Trailing whitespace, EOF, YAML, etc.

## Configuration

Tous les fichiers actifs sont maintenant centralisés dans le dossier `pre-commit/` :
- `/pre-commit/.pre-commit-config.yaml` - Configuration active
- `/pre-commit/hooks/pre-commit` - Hook git actif
- Git est configuré avec `core.hooksPath = pre-commit/hooks`

## Commandes utiles

```bash
# Lancer manuellement tous les checks
pre-commit run --all-files --config pre-commit/.pre-commit-config.yaml

# Réinstaller les hooks
pre-commit install --config pre-commit/.pre-commit-config.yaml

# Skip le pre-commit (pas recommandé)
git commit --no-verify
```

## Documentation

https://pre-commit.com/
