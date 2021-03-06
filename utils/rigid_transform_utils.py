"""
Parts of the code are adapted from https://github.com/akanazawa/hmr
"""
import numpy as np

def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (R, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 3. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 4. Recover translation.
    t = mu2 - (R.dot(mu1))

    # 5. Error:
    S1_hat = R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat