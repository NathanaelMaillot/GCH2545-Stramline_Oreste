import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from time import time
#question pour le prof: 
'possible avoir réponse?'
'préferer creer classe pr parametre ou peut directement mettre comme valeur '
'prefere une page analyse et et une page fonxction ou divide'
'ideal de combiner fonction ou ok si plusieurs'


#Fonction élémentaire afin de créer le maillage de base de notre systeme 
#qui nous permet de discrétiser l'équation de Laplaxe avec la methode des MDFs
def creer_maillage(R, R_ext, nr, ntheta):
    r = np.linspace(R, R_ext, nr)
    theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False) # genere point espacés également 
    dr = r[1] - r[0]
    dtheta = theta[1] - theta[0]
    return r, theta, dr, dtheta

'Ya moyen de combiner selon vous???'
#Convertit indice 2D en 1D pour la suite de notre matrice creuse
def obtenir_indice(i, j, ntheta):
    return i * ntheta + j

#Fonction qui permet de construire notre systeme de matrice. Discretise l'équation polaire en Laplace
#On applique condition dirichlet aux frontières ert MDFs ordre2 pour nos dérivées
def construire_matrice_systeme(r, theta, dr, dtheta, U_inf, R, R_ext):
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

#Résoude syst. linéaire avcec spsolve, donne réponse en 2D
def resoudre_laplace(A, b, nr, ntheta):
    psi_aplati = spsolve(A, b)
    return psi_aplati.reshape((nr, ntheta))



#Fonction qui va servir à déterminer le champ de vitesse à partir réponse de "def resoudre_laplace"
#On convertit les vitesses en cartesienne pour les graphiques
def calculer_vitesses(psi, r, theta, dr, dtheta):
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

    # Conversion vers un systeme cartésien poyr représentation dans un graphique
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    for i in range(nr):
        for j in range(ntheta):
            u[i, j] = vr[i, j] * np.cos(theta[j]) - vtheta[i, j] * np.sin(theta[j])
            v[i, j] = vr[i, j] * np.sin(theta[j]) + vtheta[i, j] * np.cos(theta[j])

    return vr, vtheta, u, v


#Solution analytique du probleme pour comparer avec solution numéerique et va servir à la validation
def solution_analytique(U_inf, r, theta, R):
    psi_exact = np.zeros((len(r), len(theta)))
    for i in range(len(r)):
        for j in range(len(theta)):
            psi_exact[i, j] = U_inf * r[i] * np.sin(theta[j]) * (1 - R**2 / r[i]**2)
    return psi_exact





#Trouver erreur L2 pour analyse
def erreur_L2(psi, psi_exact):
    return np.sqrt(np.sum((psi - psi_exact)**2))

#calcul le coeff. de pression Cp pour la surface du cylindre ainsi que les coeffs. aérodynamique
#utilisé ppour la portance et trainée des graphiques
def calculer_coefficients_pression(vr, vtheta, theta, U_inf):
    V_surface = np.sqrt(vr[0, :]**2 + vtheta[0, :]**2)
    Cp = 1 - (V_surface / U_inf)**2
    Cd = -0.5 * np.trapz(Cp * np.cos(theta), theta)
    Cl = -0.5 * np.trapz(Cp * np.sin(theta), theta)
    return Cp, Cd, Cl


'Est-ce mieux de laisser dans le fichier fonction ou on devrait mettre ds analyse?!'
#Sert à tracer les lignes de courants (trajectoire du fluide)  
def tracer_lignes_courant(psi, r, theta, R):
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
    plt.show()
    
    
'Est-ce mieux de laisser dans le fichier fonction ou on devrait mettre ds analyse?!'
'combiner les deux ?!?!?'
#Sert à tracer champs de vitesse/intensité pour graphique
def tracer_champ_vitesse(u, v, r, theta, R, saut=2):
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
    plt.show()

def analyse_convergence(maillages, U_inf=10, R=3, R_ext=10):
    erreurs = []
    temps = []
    nb_points = []

    for nr, ntheta in maillages:
        print(f"\n=== Maillage {nr} x {ntheta} ===")
        r, theta, dr, dtheta = creer_maillage(R, R_ext, nr, ntheta)

        debut = time()
        A, b = construire_matrice_systeme(r, theta, dr, dtheta, U_inf, R, R_ext)
        psi = resoudre_laplace(A, b, nr, ntheta)
        vr, vtheta, u, v = calculer_vitesses(psi, r, theta, dr, dtheta)
        duree = time() - debut

        psi_exact = solution_analytique(U_inf, r, theta, R)
        erreur = erreur_L2(psi, psi_exact)

        erreurs.append(erreur)
        temps.append(duree)
        nb_points.append(nr * ntheta)

        print(f"Erreur L2 : {erreur:.2e} | Temps : {duree:.2f}s")

    return np.array(nb_points), np.array(erreurs), np.array(temps)


#Analyser la convergencee et donc, évolution erreur L2 selon le rafinement du maillage, ettemps de calcul
def tracer_convergence(nb_points, erreurs, temps):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Erreur L2 vs nombre de points
    ax1.loglog(nb_points, erreurs, 'o-', label='Erreur L2')
    h = 1 / np.sqrt(nb_points)
    ax1.loglog(nb_points, erreurs[0] * (h / h[0])**2, '--', label='Référence O(h²)')
    ax1.set_title("Convergence de l'erreur")
    ax1.set_xlabel("Nombre de points")
    ax1.set_ylabel("Erreur L2")
    ax1.legend()
    ax1.grid(True)

    # Temps vs nombre de points
    ax2.loglog(nb_points, temps, 's-r', label='Temps de calcul')
    ax2.set_title("Performance")
    ax2.set_xlabel("Nombre de points")
    ax2.set_ylabel("Temps (s)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
