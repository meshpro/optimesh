def jac_uniform(mesh, mask):
    # create Jacobian
    cv = mesh.get_control_volumes(idx=mask)
    cvc = mesh.get_control_volume_centroids(idx=mask)
    return 2 * (mesh.points - cvc) * cv[:, None]
