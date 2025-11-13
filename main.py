import os
import numpy as np
import imageio.v2 as imageio

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score


def load_orl_dataset(root_dir, n_persons=40, n_train=5, n_test=5):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for person_id in range(1, n_persons + 1):
        # s01, s02, ..., s40
        person_folder = os.path.join(root_dir, f"s{person_id:02d}")

        # primele 5 imagini -> train
        for img_id in range(1, n_train + 1):
            # global: s01 -> 001..010, s02 -> 011..020, etc.
            global_id = (person_id - 1) * 10 + img_id
            filename = f"{global_id:03d}.bmp"
            img_path = os.path.join(person_folder, filename)

            img = imageio.imread(img_path)
            img_vec = img.flatten().astype(np.float32)
            train_images.append(img_vec)
            train_labels.append(person_id)

        # următoarele 5 imagini -> test
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


def main():
    # 1. încărcăm datele
    dataset_dir = "orl_faces"
    X_train, y_train, X_test, y_test = load_orl_dataset(dataset_dir)

    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)

    # 2. PCA
    n_components = 50
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(X_train)

    Z_train = pca.transform(X_train)
    Z_test = pca.transform(X_test)

    print("Z_train shape:", Z_train.shape)
    print("Z_test shape :", Z_test.shape)

    # 3. Bayes cu repartiții normale (QDA)
    clf = LinearDiscriminantAnalysis()
    clf.fit(Z_train, y_train)

    # 4. Predicții
    y_pred = clf.predict(Z_test)

    # 5. Acuratețe
    acc = accuracy_score(y_test, y_pred)
    print(f"Acuratețe: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
