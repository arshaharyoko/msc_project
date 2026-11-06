"""Okabe-Ito palette and the color and marker assignments shared by every figure script."""

BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
YELLOW = "#F0E442"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#999999"
DARK_GRAY = "#333333"

ORANGE_DARK = "#A36F00"
GREEN_LIGHT = "#80CFB8"
VERMILLION_LIGHT = "#EAAA80"
BLUE_DARK = "#004A73"

OKABE_ITO = (ORANGE, SKY, GREEN, YELLOW, BLUE, VERMILLION, PURPLE, BLACK)

VARIANT = {"MMD + tSVD": BLUE, "MMD": SKY, "KL + tSVD": ORANGE_DARK, "KL": ORANGE}
VARIANT_MARKER = {"MMD + tSVD": "o", "MMD": "o", "KL + tSVD": "^", "KL": "^"}
DIVERGENCE = {"MMD": BLUE, "KL": ORANGE}
DIVERGENCE_MARKER = {"MMD": "o", "KL": "^"}

FACTOR = {"svd": GREEN, "mmd": BLUE, "rgcn": VERMILLION, "gate": PURPLE, "cost": ORANGE}
NEGATIVES = {"graded": BLUE, "legacy": BLACK}
OPT_ARM = {"raw": PURPLE, "anch": BLUE, "prot": ORANGE}

LEVELS = (BLUE, SKY, GREEN, ORANGE, VERMILLION, PURPLE)

THRESHOLD = VERMILLION
ANNOTATION = DARK_GRAY

CLUSTER_STYLES = (
    (BLUE, "o"), (ORANGE, "s"), (GREEN, "^"), (VERMILLION, "D"),
    (PURPLE, "v"), (SKY, "P"), (BLACK, "X"), (GRAY, "h"),
    (BLUE_DARK, "<"), (ORANGE_DARK, ">"), (GREEN_LIGHT, "p"), (VERMILLION_LIGHT, "*"),
)

SEQUENTIAL = "cividis"
