import numpy as np

def generate_heat_diffusion_sequence(
    shape=(64,64), 
    loc=(32,32), 
    radius=3, 
    alpha=1,  # <<< was 1e-3, increase by 10x or 100x
    timesteps=20, 
    dt=1.0
):
    grid = np.zeros(shape, dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (i - loc[0])**2 + (j - loc[1])**2 <= radius**2:
                grid[i,j] = 1.0

    u = grid.copy()
    seq = [u.copy()]
    dx = 1.0
    dy = 1.0

    for t in range(timesteps-1):
        u_new = u.copy()
        laplacian_sum = 0.0
        max_laplacian = -1e10
        for i in range(1, shape[0]-1):
            for j in range(1, shape[1]-1):
                laplacian = (
                    (u[i+1,j] - 2*u[i,j] + u[i-1,j]) +
                    (u[i,j+1] - 2*u[i,j] + u[i,j-1])
                )
                laplacian_sum += np.abs(laplacian)
                max_laplacian = max(max_laplacian, np.abs(laplacian))
                u_new[i,j] = u[i,j] + alpha * dt * laplacian
                print(f"After update: max={u.max():.4f}, min={u.min():.4f}")

        u = u_new
        seq.append(u.copy())
        print(f"Timestep {t+1}: mean abs laplacian: {laplacian_sum/(shape[0]*shape[1]):.6f}, max abs laplacian: {max_laplacian:.6f}")


    return np.stack(seq, axis=0)



if __name__ == "__main__":
    seq = generate_heat_diffusion_sequence(alpha=1e-2)
    for t, frame in enumerate(seq):
        print(f"Timestep {t} - min: {frame.min():.4f}, max: {frame.max():.4f}")
    num_samples = 100
    data = []
    for k in range(num_samples):
        # Random hotspot
        x = np.random.randint(10,54)
        y = np.random.randint(10,54)
        radius = np.random.randint(2,5)
        seq = generate_heat_diffusion_sequence(loc=(y,x), radius=radius)
        data.append(seq)

    data = np.stack(data, axis=0)  # (N,T,H,W)
    print("Generated data shape:", data.shape)
    np.save("heat_train_data.npy", data)
