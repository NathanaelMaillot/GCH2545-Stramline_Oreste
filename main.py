import numpy as np
from cylinder_flow import (
    creer_maillage,
    construire_matrice_systeme,
    resoudre_laplace,
    calculer_vitesses,
    solution_analytique,
    erreur_L2,
    calculer_coefficients_pression,
    tracer_lignes_courant,
    tracer_champ_vitesse,
    tracer_cartes_vx_vy,
    analyse_convergence,
    tracer_convergence,
    seq_maillages,
)


def main():
    """Exécute la simulation d'écoulement autour d'un cylindre.

    Parameters
    ----------
    None

    Returns
    -------
    None
        La fonction ne renvoie aucune valeur mais génère des figures
        et affiche les résultats numériques.

    Notes
    -----
    Les paramètres physiques ``U_inf`` (m/s), ``R`` (m) et ``R_ext`` (m)
    sont définis directement dans la fonction.
    """

    # Paramètres du problème donnés
    U_inf = 10  # vitesse uniforme à l'infini (m/s)
    R = 3       # rayon du cylindre (m)
    R_ext = 10  # rayon extérieur du domaine (m)

    # Résolution pour le maillage choisi
    print("=== Résolution du problème ===")
    # création du maillage polaire
    r, theta, dr, dtheta = creer_maillage(R, R_ext, 50, 80)
    A, b = construire_matrice_systeme(r, theta, dr, dtheta, U_inf, R, R_ext)
    psi = resoudre_laplace(A, b, len(r), len(theta))
    vr, vtheta, u, v = calculer_vitesses(psi, r, theta, dr, dtheta)
    u_n, v_n = u / U_inf, v / U_inf  # vitesses normalisées
    tracer_cartes_vx_vy(u_n, v_n, r, theta, R)

    # Vérification avec la solution exacte
    psi_exact = solution_analytique(U_inf, r, theta, R)
    erreur = erreur_L2(psi, psi_exact)
    print(f"\nErreur L2 : {erreur:.2e}")

    # Calcul des coefficients aérodynamiques
    Cp, Cd, Cl = calculer_coefficients_pression(vr, vtheta, theta, U_inf)
    print(f"Coefficient de traînée Cd : {Cd:.6f}")
    print(f"Coefficient de portance Cl : {Cl:.6f}")
    tracer_lignes_courant(psi, r, theta, R)
    tracer_champ_vitesse(u, v, r, theta, R)

    # Analyse de la convergence
    print("\n=== Analyse de convergence ===")
    tailles_maillage = seq_maillages()
    print(f"Taille de maillage : {tailles_maillage}")
    # calcul L2, temps total et nombre de points
    nb_points, erreurs, temps = analyse_convergence(tailles_maillage, U_inf, R, R_ext)

    eoc = np.log(erreurs[1:] / erreurs[:-1]) / np.log(nb_points[1:] / nb_points[:-1])
    print("EOC attendu ≈ -0.50 :", eoc)
    tracer_convergence(nb_points, erreurs, temps)


if __name__ == "__main__":
    main()
