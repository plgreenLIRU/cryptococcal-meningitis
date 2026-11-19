"""
Build the final TIFF figures in ``figures/manuscript/``.

Run these source-figure scripts before this one:

1. ``uv run python test_samples.py``
   generates Figure 1: ``figures/tests/population_noise.png``
2. ``uv run python post_process.py``
   generates Figure 2: ``figures/posteriors/threshold_only.png``
3. ``uv run python Bayes.py``
   generates patient diagnostic figures in ``figures/diagnostics/``
4. ``uv run python Predict.py``
   generates Figures 4 and 5 source plots in ``figures/predictions/``
5. ``Rscript posterior_estimates.R``
   generates Figure 6: ``figures/posteriors/population_posterior_plot.png``

Then run:

``uv run python make_manuscript_figures.py``

This script combines or converts those source figures into journal-compliant
TIFF files for Figures 1--6.
"""

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_ROOT / "figures"
MANUSCRIPT_DIR = FIGURES_DIR / "manuscript"
MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
for obsolete_format in ("*.png", "*.pdf"):
    for obsolete_path in MANUSCRIPT_DIR.glob(obsolete_format):
        obsolete_path.unlink()

CM_TO_INCH = 1 / 2.54
SINGLE_COLUMN_CM = 8.4
DOUBLE_COLUMN_CM = 17.4
MAX_HEIGHT_CM = 23.0


def save_journal_tiff(fig, destination: Path, width_cm: float,
                      height_cm: float, dpi: int, mode: str = "RGB") -> None:
    """Save an exact-size, TIFF-format journal figure."""
    if width_cm not in (SINGLE_COLUMN_CM, DOUBLE_COLUMN_CM):
        raise ValueError("Width must be 8.4 cm or 17.4 cm")
    if height_cm > MAX_HEIGHT_CM:
        raise ValueError("Figure height exceeds 23 cm")

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="white")
    buffer.seek(0)

    image = Image.open(buffer).convert(mode)
    target_width = round(width_cm * CM_TO_INCH * dpi)
    target_height = round(height_cm * CM_TO_INCH * dpi)
    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    image.save(destination, format="TIFF", dpi=(dpi, dpi), compression="tiff_lzw")


def copy_journal_tiff(source: Path, destination: Path, width_cm: float,
                      height_cm: float, dpi: int, mode: str = "L") -> None:
    """Resize and convert an existing PNG/PDF source into TIFF."""
    image = Image.open(source).convert(mode)
    target_width = round(width_cm * CM_TO_INCH * dpi)
    target_height = round(height_cm * CM_TO_INCH * dpi)
    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    image.save(destination, format="TIFF", dpi=(dpi, dpi), compression="tiff_lzw")


# Figure 1: population noise (greyscale, 600 dpi)
copy_journal_tiff(
    FIGURES_DIR / "tests" / "population_noise.png",
    MANUSCRIPT_DIR / "figure_1_population_noise.tiff",
    DOUBLE_COLUMN_CM, 14.0, 600, "L",
)

# Figure 2: threshold only (black-and-white, 1200 dpi)
copy_journal_tiff(
    FIGURES_DIR / "posteriors" / "threshold_only.png",
    MANUSCRIPT_DIR / "figure_2_threshold_only.tiff",
    SINGLE_COLUMN_CM, 8.4, 1200, "1",
)

# Figure 3: four gold-standard patient panels (colour, RGB, 300 dpi)
patient_ids = (10003, 10007, 10009, 10011)
patient_images = []
for index, patient_id in enumerate(patient_ids):
    source = (FIGURES_DIR / "diagnostics" / f"{patient_id}_0" /
              f"posterior_diagnostic_{index}.png")
    if not source.exists():
        raise FileNotFoundError(f"Missing diagnostic figure: {source}")
    patient_images.append(source)

fig, axes = plt.subplots(2, 2, figsize=(17.4 * CM_TO_INCH, 18.0 * CM_TO_INCH))
for ax, panel_label, image_path in zip(axes.flat, ("A", "B", "C", "D"), patient_images):
    ax.imshow(mpimg.imread(image_path))
    ax.set_title(panel_label, loc="left", fontsize=12, fontweight="bold")
    ax.axis("off")
fig.tight_layout()
save_journal_tiff(fig, MANUSCRIPT_DIR / "figure_3_patient_parameter_estimates.tiff",
                  DOUBLE_COLUMN_CM, 18.0, 300, "RGB")
plt.close(fig)

lp_labels = (
    "Two LP data points",
    "Three LP data points",
    "Four LP data points",
    "Five LP data points",
)

# Figure 4: declining observations (colour, RGB, 300 dpi)
declining_images = [
    FIGURES_DIR / "predictions" / f"declining_observations_{n}.png"
    for n in range(2, 6)
]
fig, axes = plt.subplots(2, 2, figsize=(17.4 * CM_TO_INCH, 18.0 * CM_TO_INCH))
for ax, label, image_path in zip(axes.flat, lp_labels, declining_images):
    if not image_path.exists():
        raise FileNotFoundError(f"Missing declining observation figure: {image_path}")
    ax.imshow(mpimg.imread(image_path))
    ax.set_title(label, fontsize=12)
    ax.axis("off")
fig.suptitle(r"Patient A $(p_d = 0.2,\ p_b = 0.1875)$", fontsize=14, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.97))
save_journal_tiff(fig, MANUSCRIPT_DIR / "figure_4_declining_observations.tiff",
                  DOUBLE_COLUMN_CM, 18.0, 300, "RGB")
plt.close(fig)

# Figure 5: growth observations (colour, RGB, 300 dpi)
growth_images = [
    FIGURES_DIR / "predictions" / f"growth_observations_{n}.png"
    for n in range(2, 6)
]
fig, axes = plt.subplots(2, 2, figsize=(17.4 * CM_TO_INCH, 18.0 * CM_TO_INCH))
for ax, label, image_path in zip(axes.flat, lp_labels, growth_images):
    if not image_path.exists():
        raise FileNotFoundError(f"Missing growth observation figure: {image_path}")
    ax.imshow(mpimg.imread(image_path))
    ax.set_title(label, fontsize=12)
    ax.axis("off")
fig.suptitle(r"Patient B $(p_d = 0.2,\ p_b = 0.25)$", fontsize=14, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.97))
save_journal_tiff(fig, MANUSCRIPT_DIR / "figure_5_growth_observations.tiff",
                  DOUBLE_COLUMN_CM, 18.0, 300, "RGB")
plt.close(fig)

# Figure 6: population posterior overlay (colour, RGB, 300 dpi)
copy_journal_tiff(
    FIGURES_DIR / "posteriors" / "population_posterior_plot.png",
    MANUSCRIPT_DIR / "figure_6_population_posterior.tiff",
    DOUBLE_COLUMN_CM, 14.0, 300, "RGB",
)

print(f"Journal TIFF figures written to: {MANUSCRIPT_DIR}")
