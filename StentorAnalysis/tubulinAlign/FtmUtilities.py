import numpy as np

# Find orientation distribution of a 2D FFT spectrum
# Applies a band-pass filter and normalizes the spectrum in 1 degree wedges
def FFT_orientation(mat, r_lo=5, r_hi=30):
    r, c = mat.shape

    assert(r_hi > r_lo), "Inner radius must be smaller than outer radius"
    assert(r_lo >= 0), "Inner radius must be non-negative"
    assert(r_hi <= r/2 and r_hi <= c/2), "Outer radius must be within dimensions of image"

    # Radius mesh
    x = np.arange(c) - np.floor(c/2)
    y = (np.arange(r) - np.floor(r/2)) * -1

    X, Y = np.meshgrid(x, y)
    radius_grid = np.sqrt(X ** 2 + Y ** 2)

    # Angle mesh in degrees
    angle_grid = np.arctan2(Y, X)
    angle_grid[angle_grid < 0] += 2 * np.pi
    angle_grid = np.round(angle_grid * 180 / np.pi)

    # print('center angles', angle_grid[int(r/2)-2:int(r/2) + 3, int(c/2)-2: int(c/2)+3])

    # Iterate 1 degree wedges

    ftm = np.zeros((360,))

    for q in range(360):
        # Number average by radius
        mask1 = angle_grid == q         # Angle mask
        mask2 = radius_grid > r_lo      # High pass mask
        mask3 = radius_grid < r_hi      # Low pass mask

        current_wedge_mask = mask1 & mask2 & mask3

        n = np.count_nonzero(current_wedge_mask)

        scores = mat[current_wedge_mask]

        # Radius list for use in normalizing
        radius_list = radius_grid[current_wedge_mask]
        unique_rad = np.unique(radius_list)

        # to_add = 0
        #
        # for r_bin in unique_rad:
        #     cur_vals = scores[radius_list == r_bin]
        #     cur_n = len(cur_vals)
        #
        #     to_add += np.sum(cur_vals) / cur_n
        #
        # if len(unique_rad) == 0:
        #     ftm[q] = ftm[q-1]
        #     continue
        #
        # to_add /= len(unique_rad)
        if n > 0:
            to_add = np.sum(scores) / n
        else:
            to_add = -1

        ftm[q] = to_add

    # Fix missing points due to angle-pixel resolution
    for mp in range(360):
        if ftm[mp] == -1:
            if mp == 0:
                ftm[mp] = (ftm[359] + ftm[1]) / 2
            if mp == 359:
                ftm[mp] = (ftm[358] + ftm[0]) / 2
            else:
                ftm[mp] = (ftm[mp-1] + ftm[mp + 1]) / 2


    # Normalize to [0, 1]
    # area = np.trapz(0.5 * ftm ** 2, dx=dq)
    # ftm /= 2 * area
    ftm = (ftm - np.min(ftm)) / (np.max(ftm) - np.min(ftm))

    # Rotated angle vector
    theta = np.arange(360) * np.pi / 180

    theta += np.pi / 2

    # Fix ends
    theta = theta.reshape((len(theta), 1))
    ftm = ftm.reshape((len(ftm), 1))

    theta = np.vstack((theta, theta[0, 0]))
    ftm = np.vstack((ftm, ftm[0, 0]))

    return theta, ftm


# Get anisotropy index from an orientation distribution function
def anisotropyIndex(theta, F):
    # Convert to a 2D second rank orientation tensor (Sander 2008)

    # Trim theta and F (repeated)
    # TODO: NOTE THIS IS HARD CODED TO ALIGN WITH OUTPUT OF FFTorientation i.e. F is ordered from 0 to 360 degrees
    F_mod = F[0:180]
    F_mod = F_mod.reshape((180, ))

    Fhat = np.zeros((2,2))

    # Sweep q for integration
    angles = np.arange(0, 180, 1) * np.pi / 180
    dq = 1 * np.pi / 180

    r11 = np.cos(angles) ** 2 * F_mod
    r22 = np.sin(angles) ** 2 * F_mod
    r12 = np.cos(angles) * np.sin(angles) * F_mod

    F11 = np.trapz(r11, dx=dq)
    F22 = np.trapz(r22, dx=dq)
    F12 = np.trapz(r12, dx=dq)

    Fhat2 = np.array([[F11, F12],
                      [F12, F22]])

    for i in range(len(angles)):
        a = angles[i]
        # idx1 = np.abs(theta_mod - a) < dq * 0.01
        # idx2 = np.abs(theta_mod + 2 * np.pi - a) < dq * 0.01
        # idx3 = np.abs(theta_mod - 2 * np.pi - a) < dq * 0.01
        #
        # idx = idx1 | idx2 | idx3
        # F_q = F_mod[idx]

        # print('cur angle ', a, 'found angle ', theta_mod[i])

        #
        # if len(F_q) > 1:
        #     print("ERROR: Found multiple indices matching angle for integration")
        #     print(F_q)
        #     return -1

        F_q = F_mod[i]

        # Integrated r_tensor
        r_tensor = np.array([[np.cos(a) ** 2, np.cos(a) * np.sin(a)],
                             [np.cos(a) * np.sin(a), np.sin(a) ** 2]])
        # r_tensor = np.array([[0.5 * (a + np.sin(a) * np.cos(a)), -0.5 * np.cos(a) ** 2],
        #                      [-0.5 * np.cos(a) ** 2, 0.5 * (a - np.sin(a) * np.cos(a))]])

        d_Fhat = r_tensor * F_q * dq

        Fhat += d_Fhat

    # print(Fhat, Fhat2) # Compare integration methods

    # Find eigenvalues, then ansiotropy index
    lambdas, v = np.linalg.eig(Fhat2)
    lambdas = np.sort(lambdas)

    # print(v)
    # print('eigens', lambdas)

    if lambdas[0] > lambdas[1]:
        first = 0
        second = 1
    else:
        first = 1
        second = 0

    temp = np.arctan2(v[1, first], v[0, first]) * 180 / np.pi
    temp2 = np.arctan2(v[1, second], v[0, second]) * 180 / np.pi
    print('principal directions', temp, temp2)

    alpha = 1 - lambdas[second] / lambdas[first]
    return alpha

