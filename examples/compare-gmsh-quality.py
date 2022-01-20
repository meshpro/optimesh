import time

import create_circle
import matplotlib.pyplot as plt
import meshio
import meshplex
import numpy as np
import scipy.sparse
from dolfin import (
    Constant,
    DirichletBC,
    Function,
    FunctionSpace,
    KrylovSolver,
    Mesh,
    TestFunction,
    TrialFunction,
    XDMFFile,
    as_backend_type,
    assemble,
    dx,
    grad,
    inner,
)

import optimesh


def get_poisson_condition(pts, cells):
    # Still can't initialize a mesh from points, cells
    filename = "mesh.xdmf"
    meshio.write_points_cells(filename, pts, {"triangle": cells})
    mesh = Mesh()
    with XDMFFile(filename) as f:
        f.read(mesh)

    # build Laplace matrix with Dirichlet boundary using dolfin
    V = FunctionSpace(mesh, "Lagrange", 1)

    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, "on_boundary")
    f = Constant(1.0)
    L = f * v * dx

    A = assemble(a)
    b = assemble(L)
    bc.apply(A, b)

    # solve(A, x, b, "cg")
    solver = KrylovSolver("cg", "none")

    x = Function(V)
    x_vec = x.vector()
    num_steps = solver.solve(A, x_vec, b)

    # convert to scipy matrix
    A = as_backend_type(A).mat()
    ai, aj, av = A.getValuesCSR()
    A = scipy.sparse.csr_matrix(
        (av, aj, ai), shape=(A.getLocalSize()[0], A.getSize()[1])
    )

    # ev = eigvals(A.todense())
    ev_max = scipy.sparse.linalg.eigs(A, k=1, which="LM")[0][0]
    assert np.abs(ev_max.imag) < 1.0e-15
    ev_max = ev_max.real
    ev_min = scipy.sparse.linalg.eigs(A, k=1, which="SM")[0][0]
    assert np.abs(ev_min.imag) < 1.0e-15
    ev_min = ev_min.real
    cond = ev_max / ev_min

    # solve poisson system, count num steps
    # b = np.ones(A.shape[0])
    # out, info  = krylov.gmres(A, b)
    # num_steps = len(info.numsteps)
    return cond, num_steps


def process(name, t):
    cond, num_steps = get_poisson_condition(pts, cells)
    data[name]["n"].append(len(pts))
    data[name]["cond"].append(cond)
    data[name]["cg"].append(num_steps)
    mesh = meshplex.MeshTri(pts, cells)
    avg_q = np.sum(mesh.q_radius_ratio) / len(mesh.q_radius_ratio)
    data[name]["q"].append(avg_q)
    print(f"{cond:.2e}", num_steps, f"{avg_q:.2f}", f"({t:.2f}s)")
    # mesh.show()


# draw test meshes
num_points = 200
kwargs = {
    "show_coedges": False,
    "cell_quality_coloring": ("viridis", 0.7, 1.0, False),
    "show_axes": False,
}
pts, cells = create_circle.gmsh(num_points)
mesh = meshplex.MeshTri(pts, cells)
mesh.save("out0.png", **kwargs)
#
pts, cells = optimesh.cvt.quasi_newton_uniform_blocks(
    pts, cells, tol=1.0e-6, max_num_steps=np.inf
)
pts, cells = optimesh.cvt.quasi_newton_uniform_full(
    pts, cells, tol=1.0e-4, max_num_steps=100
)
mesh = meshplex.MeshTri(pts, cells)
mesh.save("out1.png", **kwargs)
#
pts, cells = create_circle.dmsh(num_points)
pts, cells = optimesh.cvt.quasi_newton_uniform_blocks(
    pts, cells, tol=1.0e-6, max_num_steps=np.inf
)
# pts, cells = optimesh.cvt.quasi_newton_uniform_full(
#     pts, cells, tol=1.0e-4, max_num_steps=100
# )
mesh = meshplex.MeshTri(pts, cells)
mesh.show(**kwargs)
mesh.save("out2.png", **kwargs)

exit(1)


data = {
    "gmsh": {"n": [], "cond": [], "cg": [], "q": []},
    "gmsh + optimesh": {"n": [], "cond": [], "cg": [], "q": []},
    "dmsh": {"n": [], "cond": [], "cg": [], "q": []},
}
for num_points in range(1000, 10000, 1000):
    t = time.time()
    pts, cells = create_circle.gmsh(num_points)
    t = time.time() - t
    process("gmsh", t)

    # pts, cells = create_circle.random(num_points)
    # ev_random = get_poisson_spectrum(pts, cells)
    # print(ev_random[-1] / ev_random[0])
    t = time.time()
    pts, cells = optimesh.cvt.quasi_newton_uniform_blocks(
        pts, cells, tol=1.0e-4, max_num_steps=np.inf
    )
    # pts, cells = optimesh.cvt.quasi_newton_uniform_full(
    #     pts, cells, tol=1.0e-4, max_num_steps=200
    # )
    t = time.time() - t
    process("gmsh + optimesh", t)

    t = time.time()
    pts, cells = create_circle.dmsh(num_points)
    # pts, cells = optimesh.cvt.quasi_newton_uniform_blocks(
    #     pts, cells, tol=1.0e-4, max_num_steps=np.inf
    # )
    # pts, cells = optimesh.cvt.quasi_newton_uniform_full(
    #     pts, cells, tol=1.0e-4, max_num_steps=100
    # )
    t = time.time() - t
    process("dmsh", t)
    print()

# plot condition number
for key, value in data.items():
    plt.plot(value["n"], value["cond"], "-o", label=key)
plt.xlabel("size matrix/matrix")
plt.title("Laplacian L2-condition number")
plt.grid()
plt.legend()
plt.ylim(bottom=0)
# plt.show()
plt.savefig("cond.svg", transparent=True, bbox_inches="tight")
plt.close()

# plot CG iterations number
for key, value in data.items():
    plt.plot(value["n"], value["cg"], "-o", label=key)
plt.xlabel("size mesh/matrix")
plt.title("number of CG iterations")
plt.ylim(bottom=0)
plt.grid()
plt.legend()
# plt.show()
plt.savefig("cg.svg", transparent=True, bbox_inches="tight")
plt.close()

# plot average mesh quality
for key, value in data.items():
    plt.plot(value["n"], value["q"], "-o", label=key)
plt.xlabel("size mesh")
plt.title("average cell quality")
plt.ylim(top=1.0)
plt.grid()
plt.legend()
# plt.show()
plt.savefig("q.svg", transparent=True, bbox_inches="tight")
plt.close()
