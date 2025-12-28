import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score



# 1. Încărcare dataset ORL

def load_orl_dataset(root_dir, n_persons=40, n_train=5, n_test=5):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for person_id in range(1, n_persons + 1):
        # s01, s02, ..., s40
        person_folder = os.path.join(root_dir, f"s{person_id:02d}")

        # primele n_train imagini -> train
        for img_id in range(1, n_train + 1):
            global_id = (person_id - 1) * 10 + img_id   # 001..400
            filename = f"{global_id:03d}.bmp"
            img_path = os.path.join(person_folder, filename)

            img = imageio.imread(img_path)
            img_vec = img.flatten().astype(np.float32)
            train_images.append(img_vec)
            train_labels.append(person_id)

        # următoarele n_test imagini -> test
        for img_id in range(n_train + 1, n_train + n_test + 1):
            global_id = (person_id - 1) * 10 + img_id
            filename = f"{global_id:03d}.bmp"
            img_path = os.path.join(person_folder, filename)

            img = imageio.imread(img_path)
            img_vec = img.flatten().astype(np.float32)
            test_images.append(img_vec)
            test_labels.append(person_id)

    X_train = np.vstack(train_images)
    y_train = np.array(train_labels)

    X_test = np.vstack(test_images)
    y_test = np.array(test_labels)

    return X_train, y_train, X_test, y_test



# 2. Bayes gaussian cu covarianță comună

def train_bayes_gaussian(Z_train, y_train, reg=1e-3):
    """
    Antrenează un clasificator Bayes gaussian cu:
    - medii separate pe clasă
    - covarianță comună pentru toate clasele
    """
    classes = np.unique(y_train)
    n_classes = len(classes)
    n_samples, n_features = Z_train.shape

    means = []
    cov_sum = np.zeros((n_features, n_features), dtype=np.float64)
    priors = []

    for c in classes:
        Xc = Z_train[y_train == c]
        mu_c = Xc.mean(axis=0)
        means.append(mu_c)

        # covarianța ne-scalată (sumă (x - mu)(x - mu)^T)
        Xc_centered = Xc - mu_c
        cov_sum += Xc_centered.T @ Xc_centered

        priors.append(len(Xc) / n_samples)

    # covarianță comună (împărțim la N - K)
    cov = cov_sum / (n_samples - n_classes)

    # regularizare pentru stabilitate numerică
    cov_reg = cov + reg * np.eye(n_features)

    # inversa și log-det (opțional)
    cov_inv = np.linalg.inv(cov_reg)
    _, logdet = np.linalg.slogdet(cov_reg)

    means = np.vstack(means)             # shape (n_classes, n_features)
    priors = np.array(priors)            # shape (n_classes,)
    log_priors = np.log(priors + 1e-12)  # să nu avem log(0)

    params = {
        "classes": classes,
        "means": means,
        "cov_inv": cov_inv,
        "logdet": logdet,
        "log_priors": log_priors,
    }
    return params


def predict_bayes_gaussian(Z_test, params):
    classes = params["classes"]
    means = params["means"]
    cov_inv = params["cov_inv"]
    log_priors = params["log_priors"]

    y_pred = []

    for z in Z_test:
        # z: (n_features,)
        scores = []
        for mu_c, log_prior in zip(means, log_priors):
            diff = z - mu_c
            # -0.5 * (z - mu)^T Σ^{-1} (z - mu) + log P(c)
            quad = diff.T @ cov_inv @ diff
            score = -0.5 * quad + log_prior
            scores.append(score)

        best_idx = np.argmax(scores)
        y_pred.append(classes[best_idx])

    return np.array(y_pred)

# 3. Rulare experiment

def run_experiment(n_train, n_test, n_components=50):
    print(f"\n=== Experiment {n_train}/{n_test} (train/test imagini per persoană) ===")

    X_train, y_train, X_test, y_test = load_orl_dataset(
        "orl_faces",
        n_persons=40,
        n_train=n_train,
        n_test=n_test,
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)

    # PCA
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(X_train)

    Z_train = pca.transform(X_train)
    Z_test = pca.transform(X_test)

    print("Z_train shape:", Z_train.shape)
    print("Z_test shape :", Z_test.shape)

    # Bayes gaussian 
    params = train_bayes_gaussian(Z_train, y_train, reg=1e-2)
    y_pred = predict_bayes_gaussian(Z_test, params)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acuratețe pe test: {acc * 100:.2f}%")

def evaluate_for_m(n_train, n_test, m):
    X_train, y_train, X_test, y_test = load_orl_dataset(
        "orl_faces", n_persons=40, n_train=n_train, n_test=n_test
    )

    pca = PCA(n_components=m, whiten=True)
    pca.fit(X_train)
    Z_train = pca.transform(X_train)
    Z_test = pca.transform(X_test)

    params = train_bayes_gaussian(Z_train, y_train, reg=1e-2)
    y_pred = predict_bayes_gaussian(Z_test, params)

    acc = accuracy_score(y_test, y_pred)
    return acc

def plot_accuracy_vs_components():
    components_list = list(range(5, 105, 5))  # 5,10,15,...,100

    acc_55 = []
    acc_73 = []

    for m in components_list:
        a1 = evaluate_for_m(n_train=5, n_test=5, m=m)
        a2 = evaluate_for_m(n_train=7, n_test=3, m=m)

        acc_55.append(a1 * 100)
        acc_73.append(a2 * 100)

        print(f"m={m:3d} -> acc 5/5={a1*100:.2f}% | acc 7/3={a2*100:.2f}%")

    plt.figure()
    plt.plot(components_list, acc_55, marker="o")
    plt.plot(components_list, acc_73, marker="o")
    plt.xlabel("Numar componente PCA (m)")
    plt.ylabel("Scor de succes / Acuratete (%)")
    plt.title("Acuratete recunoastere vs numar componente PCA")
    plt.legend(["Split 5/5 (50/50)", "Split 7/3 (70/30)"])
    plt.grid(True)
    plt.savefig("accuracy_vs_m.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: accuracy_vs_m.png")


def reconstruct_and_plot_rsz(n_train=5, n_test=5, ms=None, img_global_id=1):
    if ms is None:
        ms = [5, 10, 20, 30, 40, 50, 75, 100]

    # folosim setul de train doar ca să învățăm PCA
    X_train, y_train, X_test, y_test = load_orl_dataset(
        "orl_faces", n_persons=40, n_train=n_train, n_test=n_test
    )

    # determin folderul imaginii
    person_id = (img_global_id - 1) // 10 + 1
    person_folder = os.path.join("orl_faces", f"s{person_id:02d}")
    img_path = os.path.join(person_folder, f"{img_global_id:03d}.bmp")

    img_orig = imageio.imread(img_path).astype(np.float32)
    h, w = img_orig.shape
    x_orig = img_orig.flatten()

    rsz_values = []
    reconstructions = []

    for m in ms:
        pca = PCA(n_components=m, whiten=True)
        pca.fit(X_train)

        z = pca.transform(x_orig.reshape(1, -1))
        x_hat = pca.inverse_transform(z).reshape(-1)

        # RMSE
        e = np.sqrt(np.mean((x_hat - x_orig) ** 2))

        # RSZ document
        rsz = -20 * np.log10(255.0 / (e + 1e-12))
        rsz_values.append(rsz)

        reconstructions.append(x_hat.reshape(h, w))

        print(f"m={m:3d} | RMSE={e:.4f} | RSZ={rsz:.2f} dB")

    # --- Afișare imagini ---
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 4, 1)
    plt.imshow(img_orig, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    for i, (m, rec) in enumerate(zip(ms, reconstructions), start=2):
        plt.subplot(3, 4, i)
        plt.imshow(rec, cmap="gray")
        plt.title(f"m={m}")
        plt.axis("off")

    plt.suptitle("Reconstrucția imaginilor folosind PCA")
    plt.tight_layout()
    plt.savefig("reconstructii_pca.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: reconstructii_pca.png")

    # --- Grafic RSZ ---
    plt.figure()
    plt.plot(ms, rsz_values, marker="o")
    plt.xlabel("Numar componente PCA (m)")
    plt.ylabel("RSZ (dB)")
    plt.title("RSZ in funcție de numarul de componente PCA")
    plt.grid(True)
    plt.savefig("rsz_vs_m.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: rsz_vs_m.png")

def plot_energy_vs_components(max_m=100, split_train=5, split_test=5):          #GRAFIC ENERGIE F(m)
    """
    Calculează energia cumulata F(m) = sum_{i=1..m} explained_variance_ratio_i
    și o salvează ca energy_vs_m.png
    """
    X_train, y_train, X_test, y_test = load_orl_dataset(
        "orl_faces", n_persons=40, n_train=split_train, n_test=split_test
    )

    # PCA cu numar mare de componente ca să putem calcula cumulativ
    pca_full = PCA(n_components=max_m, whiten=False)
    pca_full.fit(X_train)

    evr = pca_full.explained_variance_ratio_           # energie pe fiecare componentă
    F = np.cumsum(evr)                                 # energie cumulata

    ms = np.arange(1, len(F) + 1)

    plt.figure()
    plt.plot(ms, F, marker="o")
    plt.xlabel("Numar componente PCA (m)")
    plt.ylabel("Energie cumulata F(m)")
    plt.title("Factorul de conservare a energiei in funcție de m")
    plt.grid(True)
    plt.ylim(0, 1.01)

    plt.savefig("energy_vs_m.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: energy_vs_m.png")

    # optional: raportează m pentru 90%, 95%, 98%
    for target in [0.90, 0.95, 0.98]:
        m_needed = int(np.argmax(F >= target) + 1)
        print(f"Pentru F={target:.2f} ai nevoie de m={m_needed} componente")

def save_success_table(n_train, n_test, components_list, filename):
    with open(filename, "w") as f:
        f.write("m,accuracy_percent\n")
        for m in components_list:
            acc = evaluate_for_m(n_train=n_train, n_test=n_test, m=m) * 100
            f.write(f"{m},{acc:.2f}\n")
    print(f"Saved: {filename}")


def main():
    print("START main")

    run_experiment(n_train=5, n_test=5, n_components=50)
    run_experiment(n_train=7, n_test=3, n_components=50)

    print("START plot_accuracy_vs_components")
    plot_accuracy_vs_components()
    print("DONE plot_accuracy_vs_components")

    print("START reconstruct_and_plot_rsz")
    reconstruct_and_plot_rsz(
        n_train=5, n_test=5,
        ms=[5, 10, 15, 20, 30, 40, 50, 75, 100],
        img_global_id=1
    )
    print("DONE reconstruct_and_plot_rsz")

    print("START plot_energy_vs_components")
    plot_energy_vs_components(max_m=100, split_train=5, split_test=5)
    print("DONE plot_energy_vs_components")

    print("START save_success_table")
    components_list = list(range(5, 105, 5))  # 5,10,...,100
    save_success_table(5, 5, components_list, "tabel_scor_succes_50_50.csv")
    save_success_table(7, 3, components_list, "tabel_scor_succes_70_30.csv")
    print("DONE save_success_table")


if __name__ == "__main__":
    main()
