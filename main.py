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

    # Bayes gaussian (implementat de tine)
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

