import os
import numpy as np
from time import time
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve




#question pour le prof: 
'préferer creer classe pr parametre ou peut directement mettre comme valeur '


def creer_maillage(R, R_ext, nr, ntheta):
    """Crée une grille polaire uniforme.

    Parameters
    ----------
    R : float
        Rayon du cylindre intérieur en mètres.
    R_ext : float
        Rayon extérieur du domaine en mètres.
    nr : int
        Nombre de nœuds dans la direction radiale.
    ntheta : int
        Nombre de nœuds dans la direction angulaire.

    Returns
    -------
    r : numpy.ndarray
        Coordonnées radiales (m).
    theta : numpy.ndarray
        Coordonnées angulaires (rad).
    dr : float
        Pas radial (m).
    dtheta : float
        Pas angulaire (rad).

    Notes
    -----
    La grille sert de support à la discrétisation de l'équation de Laplace
    par la méthode des différences finies.
    """
    r = np.linspace(R, R_ext, nr)
    theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False) # genere point espacés également
    dr = r[1] - r[0]
    dtheta = theta[1] - theta[0]
    return r, theta, dr, dtheta


def obtenir_indice(i, j, ntheta):
    """Convertit un couple d'indices 2D en indice 1D.

    Parameters
    ----------
    i : int
        Indice radial.
    j : int
        Indice angulaire.
    ntheta : int
        Nombre total de points angulaires.

    Returns
    -------
    int
        Indice aplati correspondant à ``(i, j)``.
    """
    return i * ntheta + j


def construire_matrice_systeme(r, theta, dr, dtheta, U_inf, R, R_ext):
    """Assemble la matrice du système pour l'équation de Laplace.

    Parameters
    ----------
    r : numpy.ndarray
        Coordonnées radiales (m).
    theta : numpy.ndarray
        Coordonnées angulaires (rad).
    dr : float
        Pas radial (m).
    dtheta : float
        Pas angulaire (rad).
    U_inf : float
        Vitesse à l'infini (m/s).
    R : float
        Rayon du cylindre (m).
    R_ext : float
        Rayon extérieur du domaine (m).

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Matrice creuse du système linéaire.
    b : numpy.ndarray
        Vecteur second membre.

    Notes
    -----
    La discrétisation est effectuée par différences finies d'ordre deux en
    coordonnées polaires avec conditions de Dirichlet imposées aux
    frontières interne et externe.
    """
    nr = len(r)
    ntheta = len(theta)
    N = nr * ntheta
    A = lil_matrix((N, N))
    b = np.zeros(N)
    
    for i in range(nr):
        for j in range(ntheta):
            idx = obtenir_indice(i, j, ntheta)
            r_i = r[i]
            
            if i == 0:
                A[idx, idx] = 1.0
                b[idx] = 0.0
            elif i == nr - 1:
                A[idx, idx] = 1.0
                b[idx] = U_inf * R_ext * np.sin(theta[j]) * (1 - R**2 / R_ext**2)
            else:
                rp = obtenir_indice(i+1, j, ntheta)
                rm = obtenir_indice(i-1, j, ntheta)
                jp = obtenir_indice(i, (j+1)%ntheta, ntheta)
                jm = obtenir_indice(i, (j-1)%ntheta, ntheta)

                A[idx, rm] = 1/dr**2 - 1/(2*r_i*dr)
                A[idx, rp] = 1/dr**2 + 1/(2*r_i*dr)
                A[idx, jm] = 1/(r_i**2 * dtheta**2)
                A[idx, jp] = 1/(r_i**2 * dtheta**2)
                A[idx, idx] = -2/dr**2 - 2/(r_i**2 * dtheta**2)
    return A.tocsr(), b


def resoudre_laplace(A, b, nr, ntheta):
    """Résout l'équation de Laplace discrétisée.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        Matrice du système linéaire.
    b : numpy.ndarray
        Vecteur second membre.
    nr : int
        Nombre de nœuds radiaux.
    ntheta : int
        Nombre de nœuds angulaires.

    Returns
    -------
    numpy.ndarray
        Potentiel de vitesse ``ψ`` au format ``(nr, ntheta)``.
    """
    psi_aplati = spsolve(A, b)
    return psi_aplati.reshape((nr, ntheta))


def calculer_vitesses(psi, r, theta, dr, dtheta):
    """Calcule les composantes de vitesse à partir du potentiel.

    Parameters
    ----------
    psi : numpy.ndarray
        Potentiel de vitesse calculé (m²/s).
    r : numpy.ndarray
        Coordonnées radiales (m).
    theta : numpy.ndarray
        Coordonnées angulaires (rad).
    dr : float
        Pas radial (m).
    dtheta : float
        Pas angulaire (rad).

    Returns
    -------
    vr : numpy.ndarray
        Composante radiale de la vitesse (m/s).
    vtheta : numpy.ndarray
        Composante angulaire de la vitesse (m/s).
    u : numpy.ndarray
        Composante cartésienne ``x`` de la vitesse (m/s).
    v : numpy.ndarray
        Composante cartésienne ``y`` de la vitesse (m/s).
    """
    nr, ntheta = psi.shape
    vr = np.zeros_like(psi)
    vtheta = np.zeros_like(psi)

    for i in range(nr):
        for j in range(ntheta):
            jm = (j - 1) % ntheta
            jp = (j + 1) % ntheta
            vr[i, j] = (psi[i, jp] - psi[i, jm]) / (2 * dtheta * r[i])

    for i in range(1, nr-1):
        for j in range(ntheta):
            vtheta[i, j] = -(psi[i+1, j] - psi[i-1, j]) / (2 * dr)

    vtheta[0, :] = vtheta[-1, :] = 0

    # Conversion vers un systeme cartésien pour représentation dans un graphique
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    for i in range(nr):
        for j in range(ntheta):
            u[i, j] = vr[i, j] * np.cos(theta[j]) - vtheta[i, j] * np.sin(theta[j])
            v[i, j] = vr[i, j] * np.sin(theta[j]) + vtheta[i, j] * np.cos(theta[j])

    return vr, vtheta, u, v


def tracer_cartes_vx_vy(u, v, r, theta, R, niveaux=200, cmap='coolwarm',
                        meme_echelle=False, fichier=None):
    """
    Affiche deux cartes: v_x (à gauche) et v_y (à droite) sur la grille polaire
    (r, theta) reprojetée en (x, y). La zone r < R est masquée (cercle blanc).
    """
    # Grille XY à partir de la grille polaire
    Rg, Tg = np.meshgrid(r, theta, indexing='ij')  # mêmes shapes que u, v
    X = Rg * np.cos(Tg)
    Y = Rg * np.sin(Tg)

    # Masquer l'intérieur du cylindre
    masque = Rg <= (R + 1e-12)
    u_m = np.ma.masked_where(masque, u)
    v_m = np.ma.masked_where(masque, v)

    # Échelles de couleurs (symétriques autour de 0)
    if meme_echelle:
        vmax = np.nanmax([np.nanmax(np.abs(u_m)), np.nanmax(np.abs(v_m))])
        vlims_u = vlims_v = (-vmax, vmax)
    else:
        vmax_u = np.nanmax(np.abs(u_m))
        vmax_v = np.nanmax(np.abs(v_m))
        vlims_u = (-vmax_u, vmax_u)
        vlims_v = (-vmax_v, vmax_v)

    # Figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im0 = axs[0].contourf(X, Y, u_m, levels=niveaux, cmap=cmap,vmin=vlims_u[0], vmax=vlims_u[1])
    axs[0].set_aspect('equal'); axs[0].set_title(r"$v_x$ Component")
    axs[0].set_xlabel("Position X"); axs[0].set_ylabel("Position Y")
    axs[0].set_xlim(-r.max(), r.max()); axs[0].set_ylim(-r.max(), r.max())
    c0 = fig.colorbar(im0, ax=axs[0]); c0.set_label(r"$v_x$")

    im1 = axs[1].contourf(X, Y, v_m, levels=niveaux, cmap=cmap,vmin=vlims_v[0], vmax=vlims_v[1])
    axs[1].set_aspect('equal'); axs[1].set_title(r"$v_y$ Component")
    axs[1].set_xlabel("Position X"); axs[1].set_ylabel("Position Y")
    axs[1].set_xlim(-r.max(), r.max()); axs[1].set_ylim(-r.max(), r.max())
    c1 = fig.colorbar(im1, ax=axs[1]); c1.set_label(r"$v_y$")

    os.makedirs("Figures", exist_ok=True)
    if fichier:
        plt.savefig(fichier, dpi=300, bbox_inches="tight")
    plt.show()


def tracer_champ_vitesse(u, v, r, theta, R, saut=2):
    """Affiche un champ de vecteurs et son intensité.

    Parameters
    ----------
    u : numpy.ndarray
        Composante cartésienne ``x`` de la vitesse (m/s).
    v : numpy.ndarray
        Composante cartésienne ``y`` de la vitesse (m/s).
    r : numpy.ndarray
        Coordonnées radiales (m).
    theta : numpy.ndarray
        Coordonnées angulaires (rad).
    R : float
        Rayon du cylindre (m).
    saut : int, optional
        Sous-échantillonnage du champ de vecteurs, par défaut ``2``.

    Returns
    -------
    None
        La fonction produit un graphique et ne renvoie aucune valeur.
    """
    R_grille, Theta_grille = np.meshgrid(r, theta, indexing='ij')
    X = R_grille * np.cos(Theta_grille)
    Y = R_grille * np.sin(Theta_grille)

    plt.figure(figsize=(8, 6))
    plt.quiver(X[::saut, ::saut], Y[::saut, ::saut], u[::saut, ::saut], v[::saut, ::saut], scale=100)
    plt.contour(X, Y, u**2 + v**2, levels=10, cmap='coolwarm', alpha=0.5)
    plt.gca().add_patch(plt.Circle((0, 0), R, color='red', fill=True, alpha=0.3))
    plt.axis('equal')
    plt.title("Champ de vitesse")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    plt.savefig("Figures/Tracee champs de vitesse.png", dpi=300)
    plt.show()


def solution_analytique(U_inf, r, theta, R):
    """Évalue la solution analytique du potentiel d'écoulement.

    Parameters
    ----------
    U_inf : float
        Vitesse uniforme à l'infini (m/s).
    r : numpy.ndarray
        Coordonnées radiales (m).
    theta : numpy.ndarray
        Coordonnées angulaires (rad).
    R : float
        Rayon du cylindre (m).

    Returns
    -------
    numpy.ndarray
        Potentiel analytique ``ψ`` (m²/s) sur la grille ``(len(r), len(theta))``.
    """
    R_grid, Theta_grid = np.meshgrid(r, theta, indexing="ij")
    psi_exact = U_inf * R_grid * np.sin(Theta_grid) * (1 - R**2 / R_grid**2)
    return psi_exact


def erreur_L2(psi, psi_exact):
    """Calcule l'erreur quadratique moyenne entre deux champs.

    Parameters
    ----------
    psi : numpy.ndarray
        Champ numérique obtenu.
    psi_exact : numpy.ndarray
        Champ de référence exact.

    Returns
    -------
    float
        Norme ``L2`` de la différence entre ``psi`` et ``psi_exact``.
    """
    return np.sqrt(np.sum((psi - psi_exact)**2))


def calculer_coefficients_pression(vr, vtheta, theta, U_inf):
    """Évalue les coefficients aérodynamiques à la paroi du cylindre.

    Parameters
    ----------
    vr : numpy.ndarray
        Composante radiale de la vitesse (m/s).
    vtheta : numpy.ndarray
        Composante angulaire de la vitesse (m/s).
    theta : numpy.ndarray
        Coordonnées angulaires à la surface (rad).
    U_inf : float
        Vitesse uniforme à l'infini (m/s).

    Returns
    -------
    Cp : numpy.ndarray
        Coefficient de pression le long de la surface.
    Cd : float
        Coefficient de traînée.
    Cl : float
        Coefficient de portance.
    """
    V_surface = np.sqrt(vr[0, :]**2 + vtheta[0, :]**2)
    Cp = 1 - (V_surface / U_inf)**2
    Cd = -0.5 * np.trapz(Cp * np.cos(theta), theta)
    Cl = -0.5 * np.trapz(Cp * np.sin(theta), theta)
    return Cp, Cd, Cl


def tracer_lignes_courant(psi, r, theta, R):
    """Affiche les lignes de courant de l'écoulement.

    Parameters
    ----------
    psi : numpy.ndarray
        Potentiel de vitesse (m²/s).
    r : numpy.ndarray
        Coordonnées radiales (m).
    theta : numpy.ndarray
        Coordonnées angulaires (rad).
    R : float
        Rayon du cylindre (m).

    Returns
    -------
    None
        La fonction trace un graphique et ne renvoie rien.
    """
    R_grille, Theta_grille = np.meshgrid(r, theta, indexing='ij')
    X = R_grille * np.cos(Theta_grille)
    Y = R_grille * np.sin(Theta_grille)
    niveaux = np.linspace(np.min(psi), np.max(psi), 20)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contour(X, Y, psi, levels=niveaux, cmap='viridis')
    ax.add_patch(plt.Circle((0, 0), R, color='red', fill=False, linewidth=2))
    ax.set_aspect('equal')
    ax.set_title("Lignes de courant")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("Figures", exist_ok=True)
    plt.savefig("Figures/Tracee ligne courants.png", dpi=300)
    plt.show()
    

def seq_maillages(R=3, R_ext=10, k=1.0, nr_list=(24, 30, 36, 45)):
    """Génère des tailles de maillage isotropes successives.

    Parameters
    ----------
    R : float, optional
        Rayon du cylindre (m), par défaut ``3``.
    R_ext : float, optional
        Rayon extérieur du domaine (m), par défaut ``10``.
    k : float, optional
        Facteur d'isotropie ``R dθ ≈ k·dr``, par défaut ``1.0``.
    nr_list : tuple of int, optional
        Valeurs possibles pour le nombre de nœuds radiaux.

    Returns
    -------
    list of tuple
        Chaque élément ``(nr, ntheta)`` correspond à une paire de tailles de
        maillage radiale et angulaire.
    """
    tailles = []
    for nr in nr_list:
        dr = (R_ext - R) / (nr - 1)
        ntheta = int(round((2 * np.pi * R) / (k * dr)))  # isotropie: R dθ ≈ k·dr
        tailles.append((nr, ntheta))
    return tailles


def analyse_convergence(tailles_maillage, U_inf, R, R_ext):
    """Évalue l'erreur et le coût pour plusieurs maillages.

    Parameters
    ----------
    tailles_maillage : list of tuple
        Liste de couples ``(nr, ntheta)`` définissant les maillages.
    U_inf : float
        Vitesse à l'infini (m/s).
    R : float
        Rayon du cylindre (m).
    R_ext : float
        Rayon extérieur du domaine (m).

    Returns
    -------
    nb_points : numpy.ndarray
        Nombre total de nœuds pour chaque maillage.
    erreurs : numpy.ndarray
        Norme ``L2`` de l'erreur associée à chaque maillage.
    temps : numpy.ndarray
        Durée de résolution pour chaque maillage (s).
    """

    L = len(tailles_maillage)
    nb_points = np.zeros(L, dtype=int)
    erreurs   = np.zeros(L, dtype=float)
    temps     = np.zeros(L, dtype=float)

    for i, couple in enumerate(tailles_maillage):
        nr, ntheta = int(couple[0]), int(couple[1])

        # --- Maillage
        r, theta, dr, dtheta = creer_maillage(R, R_ext, nr, ntheta)

        # --- Assemblage + résolution (temps mesuré)
        t0 = time()
        A, b = construire_matrice_systeme(r, theta, dr, dtheta, U_inf, R, R_ext)
        psi = resoudre_laplace(A, b, nr, ntheta)
        temps[i] = time() - t0

        # --- Référence + erreur
        psi_ref = solution_analytique(U_inf, r, theta, R)
        erreurs[i] = erreur_L2(psi, psi_ref)
        nb_points[i] = nr * ntheta

#       print(f"Erreur L2 : {erreur:.2e} | Temps : {duree:.2f}s")


    return nb_points, erreurs, temps


def tracer_convergence(nb_points, erreurs, temps):
    """Trace les courbes de convergence et de performance.

    Parameters
    ----------
    nb_points : numpy.ndarray
        Nombre de nœuds pour chaque maillage.
    erreurs : numpy.ndarray
        Erreurs ``L2`` correspondantes.
    temps : numpy.ndarray
        Temps de calcul pour chaque maillage (s).

    Returns
    -------
    None
        La fonction produit des graphiques et ne renvoie rien.
    """
    # Référence ~ N^{-1/2}, ancrée au 2e point si possible (sinon au 1er)
    j = 1 if len(erreurs) >= 2 else 0
    ref = erreurs[j] * (nb_points / nb_points[j])**(-0.5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Erreur
    ax1.loglog(nb_points, erreurs, 'o-', label='Erreur')
    ax1.loglog(nb_points, ref, '--', label='Référence ∝ N$^{-1/2}$')
    ax1.set_xlabel('Nombre de nœuds $N$')
    ax1.set_ylabel('Erreur')
    ax1.set_title("Convergence de l'erreur")
    ax1.grid(True)
    ax1.legend()

    # Performance
    ax2.loglog(nb_points, temps, 's-', label='Temps de calcul')
    ax2.set_xlabel('Nombre de nœuds $N$')
    ax2.set_ylabel('Temps (s)')
    ax2.set_title('Performance')
    ax2.grid(True)
    ax2.legend()
    os.makedirs("Figures", exist_ok=True)
    plt.savefig("Figures/Evolution convergence et performance vs nombre point.png", dpi=300)
    plt.tight_layout()
    plt.show()