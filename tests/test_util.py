import pytest
import numpy as np

from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction, Colours
from bayes_opt.util import acq_max, load_logs, ensure_rng

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor


def get_globals():
    X = np.array([
        [0.00, 0.00],
        [0.99, 0.99],
        [0.00, 0.99],
        [0.99, 0.00],
        [0.50, 0.50],
        [0.25, 0.50],
        [0.50, 0.25],
        [0.75, 0.50],
        [0.50, 0.75],

        [0.05, 0.00],
        [0.09, 0.05],
        [0.09, 0.30],
        [0.11, 0.30],
        [0.11, 0.45],
        [0.95, 0.99],
        [0.05, 0.99],
        [0.95, 0.00],
        [0.55, 0.50],
        [0.30, 0.50],
        [0.55, 0.25],
        [0.80, 0.50],
        [0.45, 0.75],
    ])

    def get_y(X):
        return -(X[:, 0] - 0.3) ** 2 - 0.5 * (X[:, 1] - 0.6)**2 + 2
    y = get_y(X)

    def get_z(X):
        return -X[:, 0] + 0.1
    z = get_z(X)

    mesh = np.dstack(
        np.meshgrid(np.arange(0, 1, 0.005), np.arange(0, 1, 0.005))
    ).reshape(-1, 2)

    GP = GaussianProcessRegressor(
        kernel=Matern(),
        n_restarts_optimizer=25,
    )
    GP.fit(X, y)

    GP_C = GaussianProcessRegressor(
        kernel=Matern(),
        n_restarts_optimizer=25,
    )
    GP_C.fit(X, z)

    return {'x': X, 'y': y, 'z': z, 'gp': GP, 'mesh': mesh, 'gp_c': GP_C}


def brute_force_maximum(MESH, GP, kind='ucb', kappa=1.0, xi=1.0):
    uf = UtilityFunction(kind=kind, kappa=kappa, xi=xi)

    mesh_vals = uf.utility(MESH, GP, 2)
    max_val = mesh_vals.max()
    max_arg_val = MESH[np.argmax(mesh_vals)]

    return max_val, max_arg_val

def constrained_brute_force_maximum(MESH, GP, GP_C, kind='ucb', kappa=1.0, xi=1.0):
    uf = UtilityFunction(kind=kind, kappa=kappa, xi=xi)

    mesh_vals = uf.utility(MESH, GP, 2)
    mesh_constraint_flag = GP_C.predict(MESH) <= 0
    if (mesh_constraint_flag).any:
        max_val = mesh_vals[mesh_constraint_flag].max()
        selected_mesh = MESH[mesh_constraint_flag]
        max_arg_val = selected_mesh[np.argmax(mesh_vals[mesh_constraint_flag])]
    else:
        max_val = -1e5
        max_arg_val = [[]]
    return max_val, max_arg_val


GLOB = get_globals()
X, Y, GP, MESH = GLOB['x'], GLOB['y'], GLOB['gp'], GLOB['mesh']
Z, GP_C = GLOB['z'], GLOB['gp_c']


def test_utility_fucntion():
    util = UtilityFunction(kind="ucb", kappa=1.0, xi=1.0)
    assert util.kind == "ucb"

    util = UtilityFunction(kind="ei", kappa=1.0, xi=1.0)
    assert util.kind == "ei"

    util = UtilityFunction(kind="poi", kappa=1.0, xi=1.0)
    assert util.kind == "poi"

    util = UtilityFunction(kind="constraint_ei", kappa=1.0, xi=1.0)
    assert util.kind == "constraint_ei"

    with pytest.raises(NotImplementedError):
        util = UtilityFunction(kind="other", kappa=1.0, xi=1.0)


def test_acq_with_ucb():
    util = UtilityFunction(kind="ucb", kappa=1.0, xi=1.0)
    episilon = 1e-2
    y_max = 2.0

    max_arg = acq_max(
        util.utility,
        GP,
        y_max,
        bounds=np.array([[0, 1], [0, 1]]),
        random_state=ensure_rng(0),
        n_iter=20
    )
    _, brute_max_arg = brute_force_maximum(MESH, GP, kind='ucb', kappa=1.0, xi=1.0)

    assert all(abs(brute_max_arg - max_arg) < episilon)


def test_acq_with_ei():
    util = UtilityFunction(kind="ei", kappa=1.0, xi=1e-6)
    episilon = 1e-2
    y_max = 2.0

    max_arg = acq_max(
        util.utility,
        GP,
        y_max,
        bounds=np.array([[0, 1], [0, 1]]),
        random_state=ensure_rng(0),
        n_iter=200,
    )
    _, brute_max_arg = brute_force_maximum(MESH, GP, kind='ei', kappa=1.0, xi=1e-6)

    assert all(abs(brute_max_arg - max_arg) < episilon)

def test_acq_with_constraint_ei():
    util = UtilityFunction(kind="constraint_ei", kappa=1.0, xi=1e-6)
    episilon = 1e-2
    y_max = 1.96

    max_arg = acq_max(
        util.utility,
        GP,
        y_max,
        bounds=np.array([[0, 1], [0, 1]]),
        random_state=ensure_rng(0),
        n_iter=500,
        constraint_gps={"constraint": GP_C},
        infeasible_penalty=10
    )
    ### TO DO: cosntrained_brute_force_maximum to be updated for more accurate brute force result
    # _, brute_max_arg = constrained_brute_force_maximum(MESH, GP, GP_C, kind='ei', kappa=1.0, xi=1e-6)
    brute_max_arg = np.array([0.3, 0.6])
    f = open("./tests/test_acq.log", "a")
    f.write("=====testing on constraint ei is conducted======\n")
    f.write("brute_max_arg: {}\n".format(brute_max_arg))
    f.write("max_arg: {}\n".format(max_arg))
    f.close()
    assert all(abs(brute_max_arg - max_arg) < episilon)


def test_acq_with_poi():
    util = UtilityFunction(kind="poi", kappa=1.0, xi=1e-4)
    episilon = 1e-2
    y_max = 2.0

    max_arg = acq_max(
        util.utility,
        GP,
        y_max,
        bounds=np.array([[0, 1], [0, 1]]),
        random_state=ensure_rng(0),
        n_iter=200,
    )
    _, brute_max_arg = brute_force_maximum(MESH, GP, kind='poi', kappa=1.0, xi=1e-4)

    assert all(abs(brute_max_arg - max_arg) < episilon)


def test_logs():
    import pytest
    def f(x, y):
        return -x ** 2 - (y - 1) ** 2 + 1

    def c(x):
        return c<2

    optimizer = BayesianOptimization(
        f=f,
        pbounds={"x": (-2, 2), "y": (-2, 2)},
        cs={"c1": c}
    )
    assert len(optimizer.space) == 0

    load_logs(optimizer, "./tests/test_logs.json")
    assert len(optimizer.space) == 5

    load_logs(optimizer, ["./tests/test_logs.json"])
    assert len(optimizer.space) == 5

    other_optimizer = BayesianOptimization(
        f=lambda x: -x ** 2,
        pbounds={"x": (-2, 2)}
    )
    with pytest.raises(ValueError):
        load_logs(other_optimizer, ["./tests/test_logs.json"])


def test_colours():
    colour_wrappers = [
        (Colours.END, Colours.black),
        (Colours.BLUE, Colours.blue),
        (Colours.BOLD, Colours.bold),
        (Colours.CYAN, Colours.cyan),
        (Colours.DARKCYAN, Colours.darkcyan),
        (Colours.GREEN, Colours.green),
        (Colours.PURPLE, Colours.purple),
        (Colours.RED, Colours.red),
        (Colours.UNDERLINE, Colours.underline),
        (Colours.YELLOW, Colours.yellow),
    ]

    for colour, wrapper in colour_wrappers:
        text1 = Colours._wrap_colour("test", colour)
        text2 = wrapper("test")

        assert text1.split("test") == [colour, Colours.END]
        assert text2.split("test") == [colour, Colours.END]


if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_target_space.py
    """
    import pytest
    pytest.main([__file__])
    # test_acq_with_constraint_ei()