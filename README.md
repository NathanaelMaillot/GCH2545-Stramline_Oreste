# GCH2545 – Écoulement potentiel autour d’un cylindre

Ce dépôt contient un projet pédagogique de simulation 2D d’un écoulement potentiel autour d’un cylindre. Il illustre la résolution de l’équation de Laplace en coordonnées polaires, la conversion en vitesses cartésiennes, le calcul de grandeurs aérodynamiques et l’analyse de convergence du maillage.

---

## Prérequis

- Python ≥ 3.10  
- Bibliothèques : numpy, scipy, matplotlib

---

## Structure du dépôt

```
.
├── main.py              # Script principal
├── cylinder_flow.py     # Fonctions de calcul et de visualisation
└── README.md            # Présent document
```

**`main.py`** orchestre la simulation : génération du maillage, assemblage du système, résolution, post-traitements graphiques et analyse de convergence.

**`cylinder_flow.py`** regroupe les fonctions nécessaires, de la création du maillage jusqu’aux tracés de résultats.

---

## Concepts clés et fonctions

| Étape               | Fonction                         | Description |
|---------------------|----------------------------------|-------------|
| Maillage polaire    | `creer_maillage`                 | Génère le maillage en coordonnées (r, θ) |
| Assemblage système  | `construire_matrice_systeme`     | Discrétise l’équation de Laplace en matrice creuse avec conditions de Dirichlet |
| Résolution          | `resoudre_laplace`               | Résout le système linéaire pour obtenir le potentiel ψ |
| Champ de vitesses   | `calculer_vitesses`              | Dérive ψ pour obtenir (u, v) cartésiens et vitesses polaires |
| Référence analytique| `solution_analytique` + `erreur_L2` | Compare la solution numérique à l’expression analytique via la norme L2 |
| Coefficients aérodynamiques | `calculer_coefficients_pression` | Calcule Cp, Cd et Cl sur la surface du cylindre |
| Visualisations      | `tracer_lignes_courant` & `tracer_champ_vitesse` | Lignes de courant et champ vectoriel de vitesse |
| Analyse de convergence | `analyse_convergence` & `tracer_convergence` | Étudie l’évolution de l’erreur L2 et du temps de calcul selon le maillage |

---

## Prise en main pour faire rouler le code

1. **Cloner le dépôt**

   ```bash
   git clone https://github.com/votre-organisation/GCH2545-Stramline_Oreste.git
   cd GCH2545-Stramline_Oreste
   ```

2. **Créer et activer un environnement**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Sous Windows : .venv\Scripts\activate
   pip install numpy scipy matplotlib
   ```

3. **Exécuter le script principal**

   ```bash
   python main.py
   ```

   Les graphiques des lignes de courant, du champ de vitesse et les courbes de convergence s’affichent automatiquement.

---

## Travailler sur le projet

### Modifier ou ajouter des fichiers directement sur GitHub

1. Ouvrir le dépôt sur github.com et naviguer jusqu’au fichier.
2. Cliquer sur l’icône **crayon** pour l’éditer ou sur **Add file** → *Create new file*.
3. Saisir les modifications.
4. Dans *Commit changes*, rédiger un court message puis choisir :
   - *Commit directly to the `main` branch* pour un changement simple ;
   - *Create a new branch* pour proposer une Pull Request.
5. Cliquer sur **Commit changes**.

### Utiliser Git en local (optionnel)

1. **Créer une branche de travail**

   ```bash
   git checkout -b feature/ma-fonction
   ```

2. **Enregistrer vos modifications**

   ```bash
   git add .
   git commit -m "Ajoute la fonction X et ses tests"
   git push origin feature/ma-fonction
   ```

3. **Ouvrir une Pull Request** sur GitHub et demander une revue.

---

## Ressources utiles

- [Guide GitHub débutant](https://docs.github.com/fr/get-started)
- [Rappels Git en français](https://git-scm.com/book/fr/v2)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)

---

## Pour aller plus loin

- Étendre le code à d’autres géométries (aile, obstacles multiples)
- Tester différentes conditions aux limites ou méthodes numériques
- Intégrer des tests automatisés ou un CI/CD simple (GitHub Actions)

---

Bon courage et bienvenue dans le projet !

