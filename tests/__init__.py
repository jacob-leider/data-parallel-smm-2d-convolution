import numpy as np

def compare_matrices(cn_mat, sp_mat, name, atol=1e-7):
    elementwise_comparison = np.isclose(cn_mat, sp_mat, atol=atol)
    if np.all(elementwise_comparison):
        print(f"Pass: {name}")
    else:
        print(f"Fail: {name}")
        print("Differences:")
        diff_indices = np.where(~elementwise_comparison)
        for i, j in zip(*diff_indices):
            sp_val = sp_mat[i, j]
            cn_val = cn_mat[i, j]
            print(
                f"Different at ({i}, {j}): scipy: {sp_val}, cn: {cn_val}")
