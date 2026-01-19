import os
import torch
from Generators.ArchitecturesGenerator.utils.dataset import get_data_loader

# ============================================================
# Funzione per generare file VNNLIB (robustezza locale)
# ============================================================
def generate_local_robustness_property(
    input_sample,
    noise_level,
    correct_label,
    property_path,
    num_classes=10,
    total_properties=None,
    sample_idx=None
):
    """
    Genera un file .vnnlib per la robustezza locale.
    Ogni assert di output è separato per ABCROWN.
    """
    os.makedirs(os.path.dirname(property_path), exist_ok=True)
    flat_input = input_sample.flatten()

    with open(property_path, "w") as f:
        # ----------------------------------------------------
        # Commenti informativi
        # ----------------------------------------------------
        if total_properties is not None:
            f.write(f"; total properties: {total_properties}\n")
        if sample_idx is not None:
            f.write(f"; property index: {sample_idx}\n")
        f.write(f"; correct label: {correct_label}\n")
        f.write(f"; epsilon (L_inf): {noise_level}\n\n")

        # ----------------------------------------------------
        # Variabili input
        # ----------------------------------------------------
        for i in range(flat_input.shape[0]):
            f.write(f"(declare-const X_{i} Real)\n")

        # ----------------------------------------------------
        # Variabili output
        # ----------------------------------------------------
        for i in range(num_classes):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # ----------------------------------------------------
        # Vincoli sugli input (L_inf ball)
        # NOTA: input non normalizzato
        # ----------------------------------------------------
        for i, val in enumerate(flat_input):
            lower = val - noise_level
            upper = val + noise_level

            f.write(f"(assert (>= X_{i} {lower:.10f}))\n")
            f.write(f"(assert (<= X_{i} {upper:.10f}))\n")

        f.write("\n")

        # ----------------------------------------------------
        # Negazione della robustezza locale (output constraints)
        # ∃ i ≠ y : Y_i ≥ Y_y
        # ----------------------------------------------------
        for i in range(num_classes):
            if i != correct_label:
                f.write(f"(assert (>= Y_{i} Y_{correct_label}))\n")


# ============================================================
# Script principale
# ============================================================
if __name__ == "__main__":
    # Parametri principali
    epsilon = 0.17
    property_folder = "./results/CUSTOM_CIFAR10"
    num_classes = 10
    total_properties = 100
    batch_size = 32
    shuffle_size = 64

    # ============================================================
    # Caricamento dataset train/test e dummy input
    # ============================================================
    train_set, test_set, dummy_input, input_dim, output_dim = get_data_loader(
        "CUSTOM_CIFAR10", batch_size, shuffle_size, input_flattened=False
    )

    os.makedirs(property_folder, exist_ok=True)

    # ============================================================
    # Generazione proprietà VNNLIB
    # ============================================================
    for idx in range(total_properties):
        input_np, label = test_set.dataset[idx]

        prop_path = os.path.join(
            property_folder,
            f"sample_{idx:04d}_label_{label}_eps_{epsilon:.4f}.vnnlib"
        )

        generate_local_robustness_property(
            input_np,
            epsilon,
            label,
            prop_path,
            num_classes=num_classes,
            total_properties=total_properties,
            sample_idx=idx
        )

    print(f"✅ Generated {total_properties} properties in {property_folder}")
