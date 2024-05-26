import matplotlib.pyplot as plt
from NMc import NMc, Secao, Concreto, AcoCA, ConjuntoBarrasCA


# Dados de entrada iniciais. Valores da geometria.
concreto = Concreto("C25")
aco = AcoCA("CA50")
cobrimento = 3.0
diametro_barra_1 = 2.0
diametro_estribo = 0.63
a = cobrimento + diametro_barra_1 / 2 + diametro_estribo
cj_1 = ConjuntoBarrasCA([(6, diametro_barra_1)])
largura_1 = 50
h_total = 19

# Criação dos objetos NMc e Secao que representam o algoritmo N-M-c e a seção de concreto armado, respectivamente
nmc = NMc()
secao = Secao(
    concreto,
    aco,
    [(largura_1, 0), (largura_1, h_total)],
    [(cj_1, a), (cj_1, h_total - a)],
    nmc,
)

# Cálculo das curvas N-M-c da seção para ELU e ELS para posterior plotagem pelo matplotlib
x, y = nmc.calc_curva_NMc(secao, ELU=True, N_sd=1203.0)
x2, y2 = nmc.calc_curva_NMc(secao, ELU=False, N_sd=1203.0)
plt.rcParams.update(
    {
        "text.usetex": True,
        "axes.labelsize": 14,
        "axes.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
)
plt.title("Momento curvatura")
plt.xlabel("Curvatura")
plt.ylabel("Momento fletor ($kNcm$)")
plt.plot(x2, y2, "k--", label=r"1,10$fcd$")
plt.plot(x, y, "k-", label=r"0,85$fcd$")
plt.grid(which="major", color="k", linestyle=":", linewidth=0.8, alpha=0.2)
plt.legend(loc="best", framealpha=1.0, edgecolor="black")
plt.tight_layout(pad=0.5)
plt.show()

# Também é possível, por meio do objeto de seção, obter os valores de Momento resistente de cálculo e Rigidez secante da seção
M_rd = secao.calc_M_rd(N_sd=1203.0)
EI_sec = secao.calc_EI_secante(N_sd=1203.0)
print(M_rd, EI_sec)
