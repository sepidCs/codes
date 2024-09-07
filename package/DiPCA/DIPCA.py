import numpy as np
import scipy.stats

# def autos(X):
#     m = X.shape[0]
#     n = X.shape[1]
#     X_m = np.zeros((m, n))
#     mu = np.mean(X, axis=0)
#     sigma = np.std(X, axis=0, ddof=1)
#     for i in range(n):
#         a = np.ones(m) * mu[i]
#         X_m[:, i] = (X[:, i]-a) / sigma[i]
#     return X_m

# def DiPCA(X, s, a):
    
#     n = X.shape[0]
#     m = X.shape[1]
#     N = n - s
#     Xe = X[s:N+s, :]
#     alpha = 0.01
#     level = 1 - alpha
#     P = np.zeros((m, a))
#     W = np.zeros((m, a))
#     T = np.zeros((n, a))
#     w = np.ones(m)
#     w = w / np.linalg.norm(w, ord=2)

#     l = 0
#     while l < a:
#         iterr = 1000
#         temp = np.dot(X, w)
#         while iterr > 0.00001:
#             t = np.dot(X, w)
#             beta = np.ones((s))
#             for i in range(s):
#                 beta[i] = np.dot(t[i:N+i-1].T, t[s:N+s-1])
#             beta = beta / np.linalg.norm(beta, ord=2)
#             w = np.zeros(m)

#             for i in range(s):
#                 w = w + beta[i] * (np.dot(X[s:N+s-1, :].T, t[i:N+i-1]) +
#                                    np.dot(X[i:N+i-1].T, t[s:N+s-1]))
#             w = w / np.linalg.norm(w, ord=2)
#             t = np.dot(X, w)
#             iterr = np.linalg.norm((t - temp), ord=2)

#             temp = t
#         p = np.dot(X.T, t) / np.dot(t.T, t)
#         p = X.T @ t / (t.T @ t)

#         t = np.array([t]).T
#         p = np.array([p]).T
#         X = X - np.dot(t, p.T)
#         P[:, l] = p[:, 0]
#         W[:, l] = w
#         T[:, l] = t[:, 0]
#         l = l + 1

#     # Dynamic Inner Modeling
#     TT = T[0:N, :]
#     j = 1
#     while j < s:
#         TT = np.c_[TT, T[j:(N+j), :]]
#         j = j + 1
#     Theta = np.dot(np.dot(np.linalg.inv(np.dot(TT.T, TT)), TT.T), T[s:N+s, :])

#     V = T[s:N+s, :] - np.dot(TT, Theta)
#     # #
#     # epsilon = 1e-10  # Small regularization constant
#     # V += epsilon * np.eye(V.shape[0])
#     # #
#     # Always return 'a' components
#     _, Sv, Pv = np.linalg.svd(V)
#     Pv = Pv.T
#     Pv = Pv[:, 0:a]
#     lambda_v = 1 / (N - 1) * np.diag(Sv[0:a] ** 2)

#     if a != a:
#         gv = 1 / (N - 1) * sum(Sv[a:a] ** 4) / sum(Sv[a:a] ** 2)
#         hv = (sum(Sv[a:a] ** 2) ** 2) / sum(Sv[a:a] ** 4)
#         Tv2_lim = a * (N ** 2 - 1) / (N * (N - a)) * scipy.stats.f.ppf(level, a, N - a)
#         Qv_lim = gv * scipy.stats.chi2.ppf(level, hv)
#         PHI_v = np.dot(np.dot(Pv, np.linalg.inv(lambda_v)), Pv.T) / Tv2_lim + (np.identity(len(Pv @ Pv.T)) - Pv @ Pv.T) / Qv_lim
#         SS_v = 1 / (N - 1) * V.T @ V
#         g_phi_v = np.trace((SS_v @ PHI_v) @ (SS_v @ PHI_v)) / (np.trace(SS_v @ PHI_v))
#         h_phi_v = (np.trace(SS_v @ SS_v) ** 2) / np.trace((SS_v @ PHI_v) @ (SS_v @ PHI_v))
#         phi_v_lim = g_phi_v * scipy.stats.chi2.ppf(level, h_phi_v)
#     else:
#         Tv2_lim = a * (N ** 2 - 1) / (N * (N - a)) * scipy.stats.f.ppf(level, a, N - a)
#         PHI_v = np.dot(np.dot(Pv, np.linalg.inv(lambda_v)), Pv.T)
#         phi_v_lim = Tv2_lim

#     Xe = Xe - np.dot(np.dot(TT, Theta), P.T)

#     # Fixed number of components in the second SVD
#     _, Ss, Ps = np.linalg.svd(Xe)
#     Ps = Ps.T
#     Ps = Ps[:, 0:a]
#     Ts = np.dot(Xe, Ps)
#     lambda_s = 1 / (N - 1) * np.diag(Ss[0:a] ** 2)
#     m = Ss.shape[0]
#     gs = 1 / (N - 1) * sum(Ss[a:m] ** 4) / sum(Ss[a:m] ** 2)
#     hs = (sum(Ss[a:m] ** 2) ** 2) / sum(Ss[a:m] ** 4)

#     Ts2_lim = scipy.stats.chi2.ppf(level, a)
#     Qs_lim = gs * scipy.stats.chi2.ppf(level, hs)

#     return P, W, Theta, Ps, lambda_s, PHI_v, phi_v_lim, Ts2_lim, Qs_lim
def transform_data(X, Ps):
    X_reduced = np.dot(X, Ps)
    return X_reduced

def autos(X):
    m = X.shape[0]
    n = X.shape[1]
    X_m = np.zeros((m, n))
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    for i in range(n):
        a = np.ones(m) * mu[i]
        X_m[:, i] = (X[:, i]-a) / sigma[i]
    return X_m, mu, sigma

def pc_number(X):
    U, S, V = np.linalg.svd(X)
    if S.shape[0] == 1:
        i = 1
    else:
        i = 0
        var = 0
        while var < 1*sum(S*S): #0.85
            var = var+S[i]*S[i]
            i = i + 1
    return i

def DiPCA1(X, s, a):
    n = X.shape[0]
    m = X.shape[1]
    N = n - s
    Xe = X[s:N+s, :]
    alpha = 0.01
    level = 1-alpha
    P = np.zeros((m, a))
    W = np.zeros((m, a))
    T = np.zeros((n, a))
    w = np.ones(m)
    w = w / np.linalg.norm(w, ord=2)

    if s > 0:
        l = 0
        while l < a:
            iterr = 1000
            temp = np.dot(X, w)
            while iterr > 0.00001:
                t = np.dot(X, w)
                beta = np.ones((s))
                for i in range(s):
                    beta[i] = np.dot(t[i:N+i-1].T, t[s:N+s-1])
                beta = beta / np.linalg.norm(beta, ord=2)
                w = np.zeros(m)

                for i in range(s):
                    w = w + beta[i]*(np.dot(X[s:N+s-1, :].T, t[i:N+i-1]) +
                                     np.dot(X[i:N+i-1].T, t[s:N+s-1]))
                w = w / np.linalg.norm(w, ord=2)
                t = np.dot(X, w)
                iterr = np.linalg.norm((t-temp), ord=2)

                temp = t
            p = np.dot(X.T, t)/np.dot(t.T, t)
            p = X.T@ t/(t.T@t)

            t = np.array([t]).T

            p = np.array([p]).T
            X = X - np.dot(t, p.T)
            P[:, l] = p[:, 0]
            W[:, l] = w
            T[:, l] = t[:, 0]
            l = l+1

        # Dynamic Inner Modeling
        TT = T[0:N, :]
        j = 1
        while j < s:
            TT = np.c_[TT, T[j:(N+j), :]]
            j = j+1
        Theta = np.dot(np.dot(np.linalg.inv(np.dot(TT.T, TT)), TT.T), T[s:N+s, :])

        V = T[s:N+s, :] - np.dot(TT, Theta)
        a_v = pc_number(V)
        _, Sv, Pv = np.linalg.svd(V)
        Pv = Pv.T
        Pv = Pv[:, 0:a_v]
        lambda_v = 1/(N-1)*np.diag(Sv[0:a_v]**2)
        if a_v!=a: # Ensure both T^2 and Q exist
            gv = 1/(N-1)*sum(Sv[a_v:a]**4)/sum(Sv[a_v:a]**2)
            hv = (sum(Sv[a_v:a]**2)**2)/sum(Sv[a_v:a]**4)
            Tv2_lim = a_v * (N ** 2 - 1) / (N * (N - a_v))* scipy.stats.f.ppf(level, a_v, N-a_v)
            Qv_lim = gv*scipy.stats.chi2.ppf(level, hv)
            PHI_v = np.dot(np.dot(Pv, np.linalg.inv(lambda_v)), Pv.T)/Tv2_lim + (np.identity(len(Pv@Pv.T))-Pv@Pv.T)/Qv_lim;
            SS_v=1/(N-1)*V.T@V
            g_phi_v=np.trace((SS_v@PHI_v)@(SS_v@PHI_v))/(np.trace(SS_v@PHI_v))
            h_phi_v=(np.trace(SS_v@SS_v)**2)/np.trace((SS_v@PHI_v)@(SS_v@PHI_v))
            phi_v_lim = g_phi_v*scipy.stats.chi2.ppf(level, h_phi_v)
        else:
            Tv2_lim = a_v * (N ** 2 - 1) / (N * (N - a_v))* scipy.stats.f.ppf(level, a_v, N-a_v)
            PHI_v = np.dot(np.dot(Pv, np.linalg.inv(lambda_v)), Pv.T)
            phi_v_lim=Tv2_lim
        Xe = Xe-np.dot(np.dot(TT, Theta), P.T)
    a_s = pc_number(Xe)
    _, Ss, Ps = np.linalg.svd(Xe)
    Ps = Ps.T
    Ps = Ps[:,0:a_s]
    Ts = np.dot(Xe, Ps)
    lambda_s = 1 / (N - 1) * np.diag(Ss[0:a_s] ** 2)
    m = Ss.shape[0]
    gs = 1 / (N - 1) * sum(Ss[a_s:m] ** 4) / sum(Ss[a_s:m] ** 2)
    hs = (sum(Ss[a_s:m] ** 2) ** 2) / sum(Ss[a_s:m] ** 4)

    Ts2_lim = scipy.stats.chi2.ppf(level,a_s)
    Qs_lim = gs*scipy.stats.chi2.ppf(level,hs)
    return P,W,Theta,Ps,lambda_s,PHI_v,phi_v_lim,Ts2_lim ,Qs_lim
