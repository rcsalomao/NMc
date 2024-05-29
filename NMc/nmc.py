from __future__ import annotations
import math
import numpy as np
from numba import jit
from typing import List, Tuple


def calcula_area_aco_conjunto(tupla_n_diametro_i: List[Tuple[int, float]]):
    """
    Calcula a área de aço total para um determinado conjunto de barras
    """
    return sum([n * math.pi * d**2 / 4 for n, d in tupla_n_diametro_i])


@jit(nopython=True)
def calcula_sigma_c_classe_I(fck, epsilon, beta):
    """
    fck: Tensão característica do concreto. (kN/cm²).
    beta: Coeficiente relacionado à ELU ou gamma_f3. (0.85 para ELU ou 1.1 caso contrário).
    """
    fcd = fck / 1.4
    epsilon_c2 = 2.0 / 1000.0
    n = 2
    if epsilon <= epsilon_c2:
        return beta * fcd * (1 - (1 - epsilon / epsilon_c2) ** n)
    else:
        return beta * fcd


@jit(nopython=True)
def calcula_epsilon_c2_classeII(fck):
    """
    fck: Tensão característica do concreto. (kN/cm²).
    """
    fck_MPa = fck * 10.0
    return 2 / 1000.0 + (0.085 / 1000.0) * ((fck_MPa - 50.0) ** 0.53)


@jit(nopython=True)
def calcula_epsilon_cu_classeII(fck):
    """
    fck: Tensão característica do concreto. (kN/cm²).
    """
    fck_MPa = fck * 10.0
    return 2.6 / 1000.0 + (35.0 / 1000.0) * ((90.0 - fck_MPa) / 100.0) ** 4


@jit(nopython=True)
def calcula_sigma_c_classe_II(fck, epsilon, beta):
    """
    fck: Tensão característica do concreto. (kN/cm²).
    beta: Coeficiente relacionado à ELU ou gamma_f3. (0.85 para ELU ou 1.1 caso contrário).
    """
    fcd = fck / 1.4
    fck_MPa = fck * 10.0
    epsilon_c2 = calcula_epsilon_c2_classeII(fck)
    n = 1.4 + 23.4 * (((90.0 - fck_MPa) / 100.0) ** 4)
    if epsilon <= epsilon_c2:
        return beta * fcd * (1 - (1 - epsilon / epsilon_c2) ** n)
    else:
        return beta * fcd


@jit(nopython=True)
def calcula_epsilon_c2(fck):
    """
    fck: Tensão característica do concreto. (kN/cm²).
    """
    if fck <= 5.0:
        return 2.0 / 1000.0
    else:
        return calcula_epsilon_c2_classeII(fck)


@jit(nopython=True)
def calcula_epsilon_cu(fck):
    """
    fck: Tensão característica do concreto. (kN/cm²).
    """
    if fck <= 5.0:
        return 3.5 / 1000.0
    else:
        return calcula_epsilon_cu_classeII(fck)


@jit(nopython=True)
def calcula_sigma_c(fck, epsilon, beta):
    """
    fck: Tensão característica do concreto. (kN/cm²).
    beta: Coeficiente relacionado à ELU ou gamma_f3. (0.85 para ELU ou 1.1 caso contrário).
    """
    if fck <= 5.0:
        return calcula_sigma_c_classe_I(fck, epsilon, beta)
    else:
        return calcula_sigma_c_classe_II(fck, epsilon, beta)


@jit(nopython=True)
def calcula_largura(x, b_i, h_i):
    for i in range(len(b_i)):
        if h_i[i] > x:
            dh = h_i[i] - h_i[i - 1]
            dx = x - h_i[i - 1]
            db = b_i[i] - b_i[i - 1]
            return b_i[i - 1] + db * (dx / dh)


@jit(
    "(float64, float64[:], float64[:], float64, float64, float64, int64)",
    nopython=True,
)
def calcula_centroide_Normal_concreto(fck, b_i, h_i, curvatura, x, beta, n_xi):
    """
    fck: Tensão característica do concreto. (kN/cm²).
    beta: Coeficiente relacionado à ELU ou gamma_f3. (0.85 para ELU ou 1.1 caso contrário).
    """
    dx = min(x, h_i[-1]) / n_xi
    N_cd = 0.0
    Na_cd = 0.0
    for i in range(n_xi):
        x_i_medio = dx * i + dx / 2.0
        epsilon_i_medio = (x - x_i_medio) * curvatura
        sigma_i_medio = calcula_sigma_c(fck, epsilon_i_medio, beta)
        N_i = calcula_largura(x_i_medio, b_i, h_i) * dx * sigma_i_medio
        N_cd += N_i
        Na_cd += N_i * (x - x_i_medio)
    alavanca_cd = Na_cd / N_cd
    centroide_N_cd = x - alavanca_cd
    return (N_cd, centroide_N_cd)


@jit(nopython=True)
def calcula_epsilon_s(d, curvatura, x):
    return curvatura * (x - d)


@jit(nopython=True)
def calcula_sigma_aco(fy, epsilon_s):
    """
    fy: Tensão de escoamento do aço. (kN/cm²)
    """
    fyd = fy / 1.15
    E_s = 21000.0
    return max(min(E_s * epsilon_s, fyd), -fyd)


@jit(nopython=True)
def calcula_Normal_resultante(
    fck, b_i, h_i, Asi, di, curvatura, x, beta, N_sd, fy, n_xi
):
    """
    fck: Tensão característica do concreto. (kN/cm²).
    beta: Coeficiente relacionado à ELU ou gamma_f3. (0.85 para ELU ou 1.1 caso contrário).
    """
    Rs = 0.0
    for i in range(len(di)):
        Rs += Asi[i] * calcula_sigma_aco(fy, calcula_epsilon_s(di[i], curvatura, x))
    N_cd, _ = calcula_centroide_Normal_concreto(fck, b_i, h_i, curvatura, x, beta, n_xi)
    return N_cd + Rs - N_sd


@jit(nopython=True)
def calcula_area_trapezio(h, a, b):
    return h * (a + b) / 2.0


@jit(nopython=True)
def calcula_centroide_trapezio(h, a, b):
    return h * ((b + 2 * a) / (3 * (a + b)))


@jit(nopython=True)
def calcula_centroide_b_i_h_i(b_i, h_i):
    area_total = 0.0
    area_i_cg_i = 0.0
    for i in range(len(b_i) - 1):
        b = b_i[i]
        a = b_i[i + 1]
        dh = h_i[i + 1] - h_i[i]
        area_i = calcula_area_trapezio(dh, a, b)
        cg_i = calcula_centroide_trapezio(dh, a, b) + h_i[i]
        area_i_cg_i += area_i * cg_i
        area_total += area_i
    return area_i_cg_i / area_total


@jit(nopython=True)
def calcula_Momento_resultante(
    fck, b_i, h_i, Asi, di, curvatura, x, beta, M_sd, fy, n_xi
):
    """
    fck: Tensão característica do concreto. (kN/cm²).
    beta: Coeficiente relacionado à ELU ou gamma_f3. (0.85 para ELU ou 1.1 caso contrário).
    """
    Ms = 0.0
    centroide_secao = calcula_centroide_b_i_h_i(b_i, h_i)
    for i in range(len(di)):
        Ms += (
            Asi[i]
            * calcula_sigma_aco(fy, calcula_epsilon_s(di[i], curvatura, x))
            * (centroide_secao - di[i])
        )
    N_cd, centroide_N_cd = calcula_centroide_Normal_concreto(
        fck, b_i, h_i, curvatura, x, beta, n_xi
    )
    return N_cd * (centroide_secao - centroide_N_cd) + Ms - M_sd


def calcula_linha_neutra(
    fck,
    b_i,
    h_i,
    Asi,
    di,
    curvatura,
    beta,
    N_sd,
    fy,
    n_xi,
    x_min=None,
    x_max=None,
    dx=None,
    tol=1e-5,
    max_iter=1000,
    x_min_absoluto=1e-3,
):
    h_secao = h_i[-1]
    if x_min is None:
        x_min = h_secao * 0.2
    if x_max is None:
        x_max = h_secao * 0.8
    if dx is None:
        dx = h_secao / 2.0
    N_x_min = calcula_Normal_resultante(
        fck, b_i, h_i, Asi, di, curvatura, x_min, beta, N_sd, fy, n_xi
    )
    N_x_max = calcula_Normal_resultante(
        fck, b_i, h_i, Asi, di, curvatura, x_max, beta, N_sd, fy, n_xi
    )
    N_x_med = 1.0
    i = 0
    while math.fabs(N_x_med) > tol:
        if N_x_min < 0.0 and N_x_max < 0.0:
            x_min = x_max
            N_x_min = N_x_max
            x_max = x_max + dx
            N_x_max = calcula_Normal_resultante(
                fck, b_i, h_i, Asi, di, curvatura, x_max, beta, N_sd, fy, n_xi
            )
        elif N_x_min > 0.0 and N_x_max > 0.0:
            x_max = x_min
            N_x_max = N_x_min
            x_min = max(x_min - dx, x_min_absoluto)
            N_x_min = calcula_Normal_resultante(
                fck, b_i, h_i, Asi, di, curvatura, x_min, beta, N_sd, fy, n_xi
            )
        elif N_x_min < 0.0 and N_x_max > 0.0:
            x_med = (x_max + x_min) / 2.0
            N_x_med = calcula_Normal_resultante(
                fck, b_i, h_i, Asi, di, curvatura, x_med, beta, N_sd, fy, n_xi
            )
            if N_x_med > 0.0:
                x_max = x_med
                N_x_max = N_x_med
            else:
                x_min = x_med
                N_x_min = N_x_med
        i += 1
        if i > max_iter:
            print("Número máximo de iterações alcançado :c")
            exit(2)
    return x_med


def calcula_deformacoes_verificacao(h_encurtamento, di, curvatura, x_LN):
    epsilon_c_top = curvatura * x_LN
    epsilon_c_encurtamento = curvatura * (x_LN - h_encurtamento)
    epsilon_si = [calcula_epsilon_s(di[i], curvatura, x_LN) for i in range(len(di))]
    epsilon_s_max = max(epsilon_si)
    epsilon_s_min = min(epsilon_si)
    epsilon_s_max_absoluto = max(math.fabs(epsilon_s_min), math.fabs(epsilon_s_max))
    return (
        epsilon_c_top,
        epsilon_c_encurtamento,
        epsilon_s_max_absoluto,
        epsilon_s_min,
        epsilon_s_max,
    )


def calcula_verificacao_ELU(
    epsilon_cu, epsilon_c2, h_encurtamento, di, curvatura, x_LN
):
    (
        epsilon_c_top,
        epsilon_c_encurtamento,
        epsilon_s_max_absoluto,
        _,
        _,
    ) = calcula_deformacoes_verificacao(h_encurtamento, di, curvatura, x_LN)
    epsilon_s_limite = 10.0 / 1000.0
    verificacoes = [
        (epsilon_c_top <= epsilon_cu),
        (epsilon_c_encurtamento <= epsilon_c2),
        (epsilon_s_max_absoluto <= epsilon_s_limite),
    ]
    if False in verificacoes:
        return False
    else:
        return True


def calcula_historico_momento_curvatura(
    fck,
    b_i,
    h_i,
    Asi,
    di,
    ELU=True,
    N_sd=0.0,
    M_sd=0.0,
    fy=50.0,
    n_xi=400,
    x_min=None,
    x_max=None,
    x_min_absoluto=1e-3,
    dx=None,
    dtheta_curvatura=0.02,
    n_dtheta_curvatura=6,
    max_iter_linha_neutra=1000,
    tol_linha_neutra=1e-5,
):
    epsilon_cu = calcula_epsilon_cu(fck)
    epsilon_c2 = calcula_epsilon_c2(fck)
    h_secao = h_i[-1]
    h_encurtamento = h_secao * ((epsilon_cu - epsilon_c2) / epsilon_cu)
    curvatura_base = epsilon_cu / h_secao
    theta_curvatura = 0.0
    theta_curvatura_verificacao = 0.0
    verificacao_ELU = True
    historico_curvatura = [0]
    historico_M_rd = [0]
    if ELU:
        if (b_i[0] + 1e-6) >= b_i[1]:
            beta = 0.85
        else:
            beta = 0.80
    else:
        beta = 1.1
        N_sd /= 1.1
        M_sd /= 1.1
    while verificacao_ELU:
        theta_curvatura += dtheta_curvatura
        curvatura = theta_curvatura * curvatura_base
        x_LN = calcula_linha_neutra(
            fck,
            b_i,
            h_i,
            Asi,
            di,
            curvatura,
            beta,
            N_sd,
            fy,
            n_xi,
            x_min,
            x_max,
            dx,
            tol_linha_neutra,
            max_iter_linha_neutra,
            x_min_absoluto,
        )
        M_rd = calcula_Momento_resultante(
            fck, b_i, h_i, Asi, di, curvatura, x_LN, beta, M_sd, fy, n_xi
        )
        verificacao_ELU = calcula_verificacao_ELU(
            epsilon_cu, epsilon_c2, h_encurtamento, di, curvatura, x_LN
        )
        if verificacao_ELU:
            theta_curvatura_verificacao = theta_curvatura
            historico_curvatura.append(curvatura)
            historico_M_rd.append(M_rd)
    verificacao_ELU = True
    theta_curvatura = theta_curvatura_verificacao
    while verificacao_ELU:
        theta_curvatura += dtheta_curvatura / n_dtheta_curvatura
        curvatura = theta_curvatura * curvatura_base
        x_LN = calcula_linha_neutra(
            fck,
            b_i,
            h_i,
            Asi,
            di,
            curvatura,
            beta,
            N_sd,
            fy,
            n_xi,
            x_min,
            x_max,
            dx,
            tol_linha_neutra,
            max_iter_linha_neutra,
            x_min_absoluto,
        )
        M_rd = calcula_Momento_resultante(
            fck, b_i, h_i, Asi, di, curvatura, x_LN, beta, M_sd, fy, n_xi
        )
        verificacao_ELU = calcula_verificacao_ELU(
            epsilon_cu, epsilon_c2, h_encurtamento, di, curvatura, x_LN
        )
        if verificacao_ELU:
            theta_curvatura_verificacao = theta_curvatura
            historico_curvatura.append(curvatura)
            historico_M_rd.append(M_rd)
    return (np.array(historico_curvatura), np.array(historico_M_rd))


@jit(nopython=True)
def interpola_sequencia_x(x, y, y_alvo):
    if y_alvo <= y[0]:
        return x[0]
    if y_alvo >= y[-1]:
        return x[-1]
    i = 0
    while y_alvo > y[i]:
        i += 1
    dy = y[i] - y[i - 1]
    dy_alvo = y_alvo - y[i - 1]
    assert dy_alvo <= dy
    razao = dy_alvo / dy
    dx = x[i] - x[i - 1]
    return x[i - 1] + razao * dx


class Concreto(object):
    _classe_fck = {
        "C" + str(classe): float(classe / 10)
        for classe in list(range(20, 55, 5)) + list(range(50, 100, 10))
    }

    def __init__(self, classe: str):
        """
        classe: Classe do concreto, segundo a norma NBR6118.
        """
        self.fck = self._classe_fck[classe]


class AcoCA(object):
    _classe_fy = {"CA25": 25, "CA50": 50, "CA60": 60}

    def __init__(self, classe: str):
        """
        classe: Classe do aço, segundo a norma NBR6118.
        """
        self.fy = self._classe_fy[classe]


class ConjuntoBarrasCA(object):
    def __init__(self, tupla_n_diametro_i: List[Tuple[int, float]]):
        """
        tupla_n_diametro_i: lista de tuplas com quantidade e diametro em 'cm' das barras que compõem cada conjunto de barras de mesma profundidade.
        """
        self.tupla_n_diametro_i = tupla_n_diametro_i

    def calc_As(self):
        return calcula_area_aco_conjunto(self.tupla_n_diametro_i)


class NMc(object):
    """
    Objeto representativo do algoritmo responsável por
    calcular e gerar a curva Normal, Momento, curvatura.
    """

    def __init__(
        self,
        n_xi=400,
        x_min=None,
        x_max=None,
        x_min_absoluto=1e-3,
        dx=None,
        dtheta_curvatura=0.02,
        n_dtheta_curvatura=6,
        max_iter_linha_neutra=1000,
        tol_linha_neutra=1e-5,
    ):
        self.n_xi = n_xi
        self.x_min = x_min
        self.x_max = x_max
        self.x_min_absoluto = x_min_absoluto
        self.dx = dx
        self.dtheta_curvatura = dtheta_curvatura
        self.n_dtheta_curvatura = n_dtheta_curvatura
        self.max_iter_linha_neutra = max_iter_linha_neutra
        self.tol_linha_neutra = tol_linha_neutra

    def calc_curva_NMc(self, secao: Secao, ELU: bool, N_sd: float = 0, M_sd: float = 0):
        return calcula_historico_momento_curvatura(
            secao.concreto.fck,
            secao.b_i,
            secao.h_i,
            secao.As_i,
            secao.d_i,
            ELU,
            N_sd,
            M_sd,
            secao.aco.fy,
            self.n_xi,
            self.x_min,
            self.x_max,
            self.x_min_absoluto,
            self.dx,
            self.dtheta_curvatura,
            self.n_dtheta_curvatura,
            self.max_iter_linha_neutra,
            self.tol_linha_neutra,
        )


class Secao(object):
    def __init__(
        self,
        concreto: Concreto,
        aco: AcoCA,
        tupla_bh_i: List[Tuple[float, float]],
        tupla_cj_barras_d_i: List[Tuple[ConjuntoBarrasCA, float]],
        nmc: NMc | None = None,
    ):
        """
        tupla_bh_i: lista de tuplas, detalhando a largura e profundidade de um perfil de concreto armardo.
        tupla_cj_barras_d_i: lista de tuplas, detalhando um conjunto de barras para cada profundidade d_i.
        """
        self.concreto = concreto
        self.aco = aco
        b_i = []
        h_i = []
        for b, h in tupla_bh_i:
            b_i.append(b)
            h_i.append(h)
        self.b_i = np.array(b_i, dtype="float64")
        self.h_i = np.array(h_i, dtype="float64")
        cj_barras_i = []
        d_i = []
        for cj_barras, d in tupla_cj_barras_d_i:
            cj_barras_i.append(cj_barras)
            d_i.append(d)
        self.d_i = np.array(d_i, dtype="float64")
        self.As_i = np.array([cj.calc_As() for cj in cj_barras_i], dtype="float64")
        self.nmc = nmc
        assert all(
            h_i[i] <= h_i[i + 1] for i in range(len(h_i) - 1)
        ), "As profundidades informadas devem estar obrigatoriamente em ordem crescente"

    def calc_M_rd(self, N_sd: float = 0, M_sd: float = 0):
        assert self.nmc is not None
        historico_curvatura, historico_momento = self.nmc.calc_curva_NMc(
            self, True, N_sd, M_sd
        )
        return historico_momento[-1]

    def calc_EI_secante(self, N_sd: float = 0, M_sd: float = 0):
        M_rd = self.calc_M_rd(N_sd, M_sd)
        historico_curvatura_servico, historico_momento_servico = (
            self.nmc.calc_curva_NMc(self, False, N_sd, M_sd)
        )
        M_rd_secante = M_rd / 1.1
        valor_curvatura_secante = interpola_sequencia_x(
            historico_curvatura_servico, historico_momento_servico, M_rd_secante
        )
        return (M_rd_secante / valor_curvatura_secante) / 10000


# fck = 4
# epsilon_i = np.linspace(0, calcula_epsilon_cu(fck), 1000)
# sigma_i = [calcula_sigma_c(fck, x) for x in epsilon_i]
# plt.plot(epsilon_i, sigma_i, 'k')
# plt.show()

# fck = 2.5
# b_i: np.array = np.array([50, 50], dtype="float64")
# h_i: np.array = np.array([0, 19], dtype="float64")
# assert all(h_i[i] <= h_i[i + 1] for i in range(len(h_i) - 1))
# h_secao = h_i[-1]
# epsilon_cu = calcula_epsilon_cu(fck)
# epsilon_c2 = calcula_epsilon_c2(fck)
# h_encurtamento = h_secao*((epsilon_cu-epsilon_c2)/epsilon_cu)
# curvatura_base = epsilon_cu/h_secao
# theta_curvatura = 1.0
# curvatura = theta_curvatura*curvatura_base
# x = 5.35
# print(calcula_centroide_Normal_concreto(fck, b_i, h_i, curvatura, x, 0.85, 400))

# As = 37.7 / 2
# cobrimento = 3.0
# a = cobrimento + 2 / 2 + 0.63
# di = np.array([a, h_secao - a], dtype="float64")
# Asi = np.array([As for _ in range(len(di))], dtype="float64")
# x_LN = calcula_linha_neutra(fck, b_i, h_i, Asi, di, curvatura, 0.85, 0.0, 50, 400)
# print(x_LN, curvatura)
# print(calcula_centroide_Normal_concreto(fck, b_i, h_i, curvatura, x_LN, 0.85, 400))
# print(calcula_Normal_resultante(fck, b_i, h_i, Asi, di, curvatura, x_LN, 0.85, 0, 50, 400))
# print(calcula_Momento_resultante(fck, b_i, h_i, Asi, di, curvatura, x_LN, 0.85, 0, 50, 400))
# print(calcula_deformacoes_verificacao(h_encurtamento, di, curvatura, x_LN))
