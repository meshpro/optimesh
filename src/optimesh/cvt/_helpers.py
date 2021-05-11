def jac_uniform(mesh, mask):
    # create Jacobian
    cv = mesh.get_control_volumes(cell_mask=mask)
    cvc = mesh.get_control_volume_centroids(cell_mask=mask)
    return 2 * (mesh.points - cvc) * cv[:, None]
