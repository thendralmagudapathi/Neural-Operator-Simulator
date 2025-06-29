import deepxde as dde
import numpy as np
from deepxde.backend import tf
from deepxde.data import TimePDE

# 1. Material to alpha mapping
material_alpha = {
    "aluminum": 9.7e-5,
    "steel": 1.2e-5,
    "iron": 2.3e-5,
}

import torch

def material_lookup_tensor(mat_id):
    # Input: mat_id is a Tensor of shape (N, 1)
    alpha_values = tf.constant([1.2e-5, 2.3e-5, 9.7e-5], dtype=tf.float32)  # steel, iron, aluminum
    mat_id_int = tf.cast(tf.squeeze(mat_id), tf.int32)
    return tf.gather(alpha_values, mat_id_int)[:, None]


# 2. PDE definition
def heat_pde(x, u):
    x_coord = x[:, 0:1]
    y_coord = x[:, 1:2]
    t = x[:, 2:3]
    hx = x[:, 3:4]
    hy = x[:, 4:5]
    mat = x[:, 5:6]

    alpha = material_lookup_tensor(mat)

    u_t = dde.grad.jacobian(u, x, j=2)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, i=1, j=1)

    return u_t - alpha * (u_xx + u_yy)



def make_ic_general():
    def func(x):
        # Extract coordinates and hotspot location from the full input
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        hx = x[:, 3:4]
        hy = x[:, 4:5]
        # Gaussian centered at (hx, hy)
        dist_sq = (x_coord - hx) ** 2 + (y_coord - hy) ** 2
        return np.exp(-100 * dist_sq)
    return func



# 3. Geometry and Time domain
geom = dde.geometry.Rectangle([0, 0], [1, 1])
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 5. Dirichlet boundary condition (fixed temperature)
def boundary(_, on_boundary):
    return on_boundary

bc = dde.DirichletBC(geomtime, lambda x: 0, boundary)
ic = dde.IC(geomtime, make_ic_general, lambda _, on_initial: on_initial)



# Override the geometry sampler
def custom_sampler(N):
    x = np.random.rand(N, 1)
    y = np.random.rand(N, 1)
    t = np.random.rand(N, 1)
    hx = np.random.rand(N, 1)
    hy = np.random.rand(N, 1)
    mat_id = np.random.randint(0, 3, size=(N, 1))  # 0: steel, 1: iron, 2: aluminum

    return np.hstack((x, y, t, hx, hy, mat_id))

# Set the sampler
data = dde.data.TimePDE(geomtime, heat_pde, [bc, ic], num_domain=10000, num_boundary=200, num_initial=200)


# Sample and append extra variables
def create_custom_data(hx_val, hy_val, material_id_val):
    def mapping(X):
        hx = np.full((X.shape[0], 1), hx_val)
        hy = np.full((X.shape[0], 1), hy_val)
        material_id = np.full((X.shape[0], 1), material_id_val)
        return np.hstack((X, hx, hy, material_id))

    def heat_pde(X_aug, y):
        x, y_spatial, t = X_aug[:, 0:1], X_aug[:, 1:2], X_aug[:, 2:3]
        hx = X_aug[:, 3:4]
        hy = X_aug[:, 4:5]
        material_id = X_aug[:, 5:6]
        return (
            dde.grad.jacobian(y, X_aug, i=0, j=2)
            - 0.01 * (dde.grad.hessian(y, X_aug, i=0, j=0) + dde.grad.hessian(y, X_aug, i=0, j=1))
        )

    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    data = MappedTimePDE(
        geomtime,
        pde=heat_pde,
        ic_bcs=[
            dde.IC(
                geomtime,
                lambda x: np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2]),
                lambda _, on_initial: on_initial,
            )
        ],
        num_domain=10000,
        num_boundary=0,
        num_initial=1000,
        mapping = mapping,
    )

    return data


# 6. Model training function
def train_general_heat_model():
    # Geometry and time
    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # Initial and boundary conditions
    ic = dde.IC(geomtime, make_ic_general(), lambda _, on_initial: on_initial)
    bc = dde.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)

    hx = 0.5
    hy = 0.5
    material_id = 1  # e.g. aluminum

    data = create_custom_data(hx, hy, material_id)

    # Custom input: [x, y, t, hx, hy, mat_id]
    net = dde.nn.FNN([6] + [128] * 3 + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    model.train(epochs=20)

    model.save("app/models/general_heat_model")
    return model


class MappedTimePDE(dde.data.TimePDE):
    def __init__(self, *args, **kwargs):
        self.mapping = kwargs.pop("mapping", None)
        super().__init__(*args, **kwargs)

    def train_next_batch(self, batch_size=None):
        inputs, outputs, auxiliary_var = super().train_next_batch(batch_size)
        if self.mapping is not None:
            inputs = self.mapping(inputs)
        return inputs, outputs, auxiliary_var



def predict_and_plot(model, t_val=1.0, resolution=100):
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    XYT = np.vstack((X.flatten(), Y.flatten(), np.full_like(X.flatten(), t_val))).T
    u = model.predict(XYT).reshape(resolution, resolution)

    import matplotlib.pyplot as plt
    plt.imshow(u, extent=[0, 1, 0, 1], origin="lower", cmap="hot")
    plt.colorbar(label="Temperature")
    plt.title(f"Heat Distribution at t = {t_val}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()



model = train_general_heat_model()
predict_and_plot(model, t_val=1.0)


"""LOAD A SAVED MODEL LATER"""

"""def load_heat_model(path):
    # Create dummy data object (required to init model before loading weights)
    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    data = dde.data.TimePDE(geomtime, lambda x, y: 0, [], num_domain=10)  # minimal dummy data

    net = dde.maps.FNN([3] + [64] * 3 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)

    model.restore(path)
    return model

# Save during training
model = train_heat_model(0.5, 0.5, "steel")

# Later, reload it
model = load_heat_model("models/heat_model_steel_hx0.5_hy0.5")
predict_and_plot(model)"""
