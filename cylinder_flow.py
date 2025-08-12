import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from time import time


#question pour le prof: 
'préferer creer classe pr parametre ou peut directement mettre comme valeur '


def creer_maillage(R, R_ext, nr, ntheta):
    """ 
    Crée le maillage de base et discrétise l'équation de Laplace par les MDFs.
    """
    r = np.linspace(R, R_ext, nr)
    theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False) # genere point espacés également 
    dr = r[1] - r[0]
    dtheta = theta[1] - theta[0]
    return r, theta, dr, dtheta


def obtenir_indice(i, j, ntheta):
    """
    Convertit indice 2D en 1D.
    """
    return i * ntheta + j


def construire_matrice_systeme(r, theta, dr, dtheta, U_inf, R, R_ext):
    """
    Construit systeme de matrice, discrétise l'équation polaire en Laplace 
    et applique les conditions dirichlets aux frontières par les MDFs à l'ordre 2 pour les dérivées.
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
    """
    Résout syst. linéaire avcec spsolve et donne la réponse en 2D.
    """
    psi_aplati = spsolve(A, b)
    return psi_aplati.reshape((nr, ntheta))


def calculer_vitesses(psi, r, theta, dr, dtheta):
    """
    Calcul le champ de vitesse à partir réponse de 'resoudre_laplace()'
    et convertit les vitesses en cartesienne.
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


def tracer_champ_vitesse(u, v, r, theta, R, saut=2):
    """
    Trace champs de vitesse/intensité.
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
    plt.savefig("Figures/Tracee champs de vitesse.png", dpi=300)
    plt.show()


def solution_analytique(U_inf, r, theta, R):
    """
    Calcul la solution analytique du probleme.
    """
    psi_exact = np.zeros((len(r), len(theta)))
    for i in range(len(r)):
        for j in range(len(theta)):
            psi_exact[i, j] = U_inf * r[i] * np.sin(theta[j]) * (1 - R**2 / r[i]**2)
    return psi_exact


def erreur_L2(psi, psi_exact):
    """
    Calcul erreur L2.
    """
    return np.sqrt(np.sum((psi - psi_exact)**2))


def calculer_coefficients_pression(vr, vtheta, theta, U_inf):
    """
    Calcul le coeff. de pression Cp et les coeffs. aérodynamique pour la portance et trainée."""
    V_surface = np.sqrt(vr[0, :]**2 + vtheta[0, :]**2)
    Cp = 1 - (V_surface / U_inf)**2
    Cd = -0.5 * np.trapz(Cp * np.cos(theta), theta)
    Cl = -0.5 * np.trapz(Cp * np.sin(theta), theta)
    return Cp, Cd, Cl


def tracer_lignes_courant(psi, r, theta, R):
    """
    Trace les lignes de courants
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
    plt.savefig("Figures/Tracee ligne courants.png", dpi=300)
    plt.show()
    

def seq_maillages(R=3, R_ext=10, k=1.0, nr_list=(24,30,36,45)):
    tailles = []
    for nr in nr_list:
        dr = (R_ext - R)/(nr - 1)
        ntheta = int(round((2*np.pi*R)/(k*dr)))  # isotropie: R dθ ≈ k·dr
        tailles.append((nr, ntheta))
    return tailles


def analyse_convergence(tailles_maillage, U_inf, R, R_ext):
    """
    Calcule nb_points, erreurs et temps pour chaque maillage.
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
        erreurs[i]   = np.sqrt(np.sum((psi - psi_ref)**2))   # √Σ e^2
        nb_points[i] = nr * ntheta

#       print(f"Erreur L2 : {erreur:.2e} | Temps : {duree:.2f}s")


    return nb_points, erreurs, temps


def tracer_convergence(nb_points, erreurs, temps):
    """
    Trace (1) erreur vs nb_points avec référence ∝ N^{-1/2}
          (2) temps de calcul vs nb_points (courbe de performance).
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

    plt.savefig("Figures/Evolution convergence et performance vs nombre point.png", dpi=300)
    plt.tight_layout()
    plt.show()