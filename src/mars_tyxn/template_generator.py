import argparse
import os
from typing import List

import cv2
import numpy as np


def draw_line_from_center(img, center, angle_deg, length, thickness=3):
    """Draw a black arm from center at a given angle (degrees)."""
    angle_rad = np.deg2rad(angle_deg)
    end_x = int(round(center[0] + length * np.cos(angle_rad)))
    end_y = int(round(center[1] + length * np.sin(angle_rad)))
    cv2.line(img, center, (end_x, end_y), 0, thickness)


def create_base_img(size, gray_level):
    """Create a square image filled with the configured gray level."""
    return np.full((size, size), gray_level, dtype=np.uint8)


def create_V_template(size=51, spread_angle=60, thickness=3, gray_level=128):
    """Legacy 2-branch template."""
    img = create_base_img(size, gray_level)
    center = (size // 2, size // 2)
    length = size // 2 - 5
    half_spread = spread_angle / 2.0
    draw_line_from_center(img, center, 270 - half_spread, length, thickness)
    draw_line_from_center(img, center, 270 + half_spread, length, thickness)
    return img


def create_T_template(size=51, branch_angle=90, thickness=3, gray_level=128):
    """
    3-arm T-like junction template.
    branch_angle=90 creates a true T (left/right crossbar + stem).
    """
    img = create_base_img(size, gray_level)
    center = (size // 2, size // 2)
    length = size // 2 - 5

    stem_angle = 90
    angles = [stem_angle, stem_angle - branch_angle, stem_angle + branch_angle]
    for angle in angles:
        draw_line_from_center(img, center, angle, length, thickness)
    return img


def create_Y_template(size=51, spread_angle=120, thickness=3, gray_level=128):
    img = create_base_img(size, gray_level)
    center = (size // 2, size // 2)
    length = size // 2 - 5
    for angle in [270, 270 + spread_angle, 270 - spread_angle]:
        draw_line_from_center(img, center, angle, length, thickness)
    return img


def create_X_template(size=51, spread_angle=90, thickness=3, gray_level=128):
    img = create_base_img(size, gray_level)
    center = (size // 2, size // 2)
    length = size // 2 - 5
    for i in range(4):
        draw_line_from_center(img, center, i * spread_angle, length, thickness)
    return img


def rotate_template(template, angle_deg, gray_level):
    """Rotate template while preserving black lines on gray background."""
    center = (template.shape[1] // 2, template.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        template,
        rotation_matrix,
        template.shape[::-1],
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=int(gray_level),
    )
    rotated[rotated > 0] = gray_level
    rotated[rotated == 0] = 0
    return rotated


def _default_list(value, fallback):
    return fallback if value is None else value


def generate_all_templates(
    output_dir="templates",
    template_sizes=None,
    rotations=None,
    background_grays=None,
    t_branch_angles=None,
    y_spreads=None,
    x_spreads=None,
    v_spreads=None,
    thickness=2,
    include_v=False,
):
    """
    Generate junction templates for T/Y/X (optionally legacy V).
    Returns total generated file count.
    """
    template_sizes = _default_list(template_sizes, [41, 61])
    rotations = _default_list(rotations, [0, 45, 90])
    background_grays = _default_list(background_grays, [50, 128, 200])
    t_branch_angles = _default_list(t_branch_angles, [90])
    y_spreads = _default_list(y_spreads, [120])
    x_spreads = _default_list(x_spreads, [90])
    v_spreads = _default_list(v_spreads, [30, 60])

    os.makedirs(output_dir, exist_ok=True)
    total = 0

    for gray in background_grays:
        gray_dir = os.path.join(output_dir, f"gray_{gray}")
        os.makedirs(gray_dir, exist_ok=True)
        print(f"Generating templates for gray level: {gray}")

        generators = [
            ("T", t_branch_angles, create_T_template, "branch_angle"),
            ("Y", y_spreads, create_Y_template, "spread_angle"),
            ("X", x_spreads, create_X_template, "spread_angle"),
        ]
        if include_v:
            generators.append(("V", v_spreads, create_V_template, "spread_angle"))

        for size in template_sizes:
            for prefix, spreads, func, angle_name in generators:
                for spread in spreads:
                    kwargs = {
                        "size": size,
                        "thickness": thickness,
                        "gray_level": gray,
                        angle_name: spread,
                    }
                    base = func(**kwargs)
                    for rotation in rotations:
                        final = rotate_template(base, rotation, gray)
                        filename = f"{prefix}_s{size}_a{int(spread):03d}_r{int(rotation):03d}.png"
                        cv2.imwrite(os.path.join(gray_dir, filename), final)
                        total += 1
    return total


def _parse_csv_ints(value: str) -> List[int]:
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def main():
    parser = argparse.ArgumentParser(description="Generate T/Y/X template library.")
    parser.add_argument("--output-dir", default="templates", help="Output directory root.")
    parser.add_argument("--sizes", default="41,61", help="Comma-separated template sizes.")
    parser.add_argument("--rotations", default="0,45,90", help="Comma-separated rotations.")
    parser.add_argument("--grays", default="50,128,200", help="Comma-separated gray background values.")
    parser.add_argument("--t-angles", default="90", help="Comma-separated T branch angles.")
    parser.add_argument("--y-angles", default="120", help="Comma-separated Y spread angles.")
    parser.add_argument("--x-angles", default="90", help="Comma-separated X spread angles.")
    parser.add_argument("--thickness", type=int, default=2, help="Line thickness.")
    parser.add_argument("--include-v", action="store_true", help="Also generate legacy V templates.")
    args = parser.parse_args()

    total = generate_all_templates(
        output_dir=args.output_dir,
        template_sizes=_parse_csv_ints(args.sizes),
        rotations=_parse_csv_ints(args.rotations),
        background_grays=_parse_csv_ints(args.grays),
        t_branch_angles=_parse_csv_ints(args.t_angles),
        y_spreads=_parse_csv_ints(args.y_angles),
        x_spreads=_parse_csv_ints(args.x_angles),
        thickness=max(1, args.thickness),
        include_v=args.include_v,
    )
    print(f"Generated {total} templates in {args.output_dir}")


if __name__ == "__main__":
    main()
