import numpy as np
import termplotlib as tpl


def print_stats(mesh, extra_cols=None):
    extra_cols = [] if extra_cols is None else extra_cols

    angles = mesh.angles / np.pi * 180
    angles_hist, angles_bin_edges = np.histogram(
        angles, bins=np.linspace(0.0, 180.0, num=73, endpoint=True)
    )

    q = mesh.q_radius_ratio
    q_hist, q_bin_edges = np.histogram(
        q, bins=np.linspace(0.0, 1.0, num=41, endpoint=True)
    )

    grid = tpl.subplot_grid(
        (1, 4 + len(extra_cols)), column_widths=None, border_style=None
    )
    grid[0, 0].hist(angles_hist, angles_bin_edges, grid=[24], bar_width=1, strip=True)
    grid[0, 1].aprint("min angle:     {:7.3f}".format(np.min(angles)))
    grid[0, 1].aprint("avg angle:     {:7.3f}".format(60))
    grid[0, 1].aprint("max angle:     {:7.3f}".format(np.max(angles)))
    grid[0, 1].aprint("std dev angle: {:7.3f}".format(np.std(angles)))
    grid[0, 2].hist(q_hist, q_bin_edges, bar_width=1, strip=True)
    grid[0, 3].aprint("min quality: {:5.3f}".format(np.min(q)))
    grid[0, 3].aprint("avg quality: {:5.3f}".format(np.average(q)))
    grid[0, 3].aprint("max quality: {:5.3f}".format(np.max(q)))

    for k, col in enumerate(extra_cols):
        grid[0, 4 + k].aprint(col)

    grid.show()


# def stepsize_till_flat(x, v):
#     """Given triangles and directions, compute the minimum stepsize t at which the area
#     of at least one of the new triangles `x + t*v` is zero.
#     """
#     # <https://math.stackexchange.com/a/3242740/36678>
#     x1x0 = x[:, 1] - x[:, 0]
#     x2x0 = x[:, 2] - x[:, 0]
#     #
#     v1v0 = v[:, 1] - v[:, 0]
#     v2v0 = v[:, 2] - v[:, 0]
#     #
#     a = v1v0[:, 0] * v2v0[:, 1] - v1v0[:, 1] * v2v0[:, 0]
#     b = (
#         v1v0[:, 0] * x2x0[:, 1]
#         + x1x0[:, 0] * v2v0[:, 1]
#         - v1v0[:, 1] * x2x0[:, 0]
#         - x1x0[:, 1] * v2v0[:, 0]
#     )
#     c = x1x0[:, 0] * x2x0[:, 1] - x1x0[:, 1] * x2x0[:, 0]
#     #
#     alpha = b ** 2 - 4 * a * c
#     i = (alpha >= 0) & (a != 0.0)
#     sqrt_alpha = np.sqrt(alpha[i])
#     t0 = (-b[i] + sqrt_alpha) / (2 * a[i])
#     t1 = (-b[i] - sqrt_alpha) / (2 * a[i])
#     return min(np.min(t0[t0 > 0]), np.min(t1[t1 > 0]))


def get_new_points_averaged(mesh, reference_points, weights=None):
    """Provided reference points for each cell (e.g., the barycenter), for each point
    this method returns the (weighted) average of the reference points of all adjacent
    cells. The weights could for example be the volumes of the cells.
    """
    if weights is None:
        scaled_rp = reference_points.T
    else:
        scaled_rp = reference_points.T * weights

    # new_points = np.zeros(mesh.points.shape)
    # for i in mesh.cells["points"].T:
    #     np.add.at(new_points, i, scaled_rp)
    # omega = np.zeros(len(mesh.points))
    # for i in mesh.cells["points"].T:
    #     np.add.at(omega, i, mesh.cell_volumes)

    n = mesh.points.shape[0]
    new_points = np.zeros(mesh.points.shape)
    for i in mesh.cells["points"].T:
        new_points += np.array(
            [np.bincount(i, vals, minlength=n) for vals in scaled_rp]
        ).T

    if weights is None:
        omega = np.bincount(mesh.cells["points"].reshape(-1), minlength=n)
    else:
        omega = np.zeros(n)
        for i in mesh.cells["points"].T:
            omega += np.bincount(i, weights, minlength=n)

    new_points /= omega[:, None]
    return new_points
