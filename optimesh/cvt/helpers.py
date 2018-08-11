# -*- coding: utf-8 -*-
#


def jac_uniform(mesh):
    # create Jacobian
    centroids = mesh.control_volume_centroids
    X = mesh.node_coords
    jac = 2 * (X - centroids) * mesh.control_volumes[:, None]
    return jac
