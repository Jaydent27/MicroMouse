"""
MTRN3100 Micromouse - Octagon-normalised image planner with manual
start/goal selection and manually selected robot start orientation.

Generated with assistance from ChatGPT (OpenAI), 12 Aug 2026.

WHAT THIS VERSION DOES
----------------------
1. Lets you select the maze photo.
2. Lets you fit a fixed-proportion octagon over the outside maze boundary.
3. Lets you manually click a START point and GOAL point.
4. Lets you rotate a heading arrow at START to match the robot's real orientation.
5. Detects the BLACK walls and BLACK circular obstacles.
6. Uses a fixed scale of 40 px = 180 mm plus the 110 mm robot width to
   calculate obstacle clearance automatically.
7. Runs A* through the remaining free space.
8. Uses a distance-transform cost so the path prefers the MIDDLE of corridors.
9. Converts the selected start heading into the FIRST route turn automatically.
10. Converts every path segment directly from image pixels to physical millimetres
    using the fixed scale 40 px = 180 mm; 180 mm is not a movement increment.
11. Scales every displayed image to fit on the current screen while preserving
    the original processing coordinates for interactive mouse selections.
12. Saves the corrected maze image, obstacle mask, clearance mask, and final path.

This version generates the route from the lab camera image and uploads the
route directly to the ESP32 over the ESP32's own Wi-Fi access point.

AI-assisted code is labelled in this file header.
"""

import cv2
import numpy as np
import heapq
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import urllib.request
import urllib.error
import webbrowser

# ============================================================
# USER SETTINGS
# ============================================================

# Fixed image-processing values required for this setup.
# No interactive threshold or pixel-width calibration is performed.
# Pixels darker than 105 are treated as obstacles.
OBSTACLE_BLACK_THRESHOLD = 105

# Fixed width of one 180 mm maze square in the processed image.
UNIT_SPACE_PX = 40.0

# Physical maze / robot dimensions.
#
# The robot is physically 110 mm wide (55 mm half-width), but the live
# left-LiDAR controller now handles much of the side-wall positioning while
# driving.  Therefore the path planner deliberately uses a slightly smaller
# obstacle inflation than the full physical half-width so narrow passages are
# not unnecessarily removed from the A* free-space map.
CELL_SIZE_MM = 180.0
ROBOT_WIDTH_MM = 110.0
ROBOT_HALF_WIDTH_MM = ROBOT_WIDTH_MM / 2.0

# Clearance used by the IMAGE PATH PLANNER.
#
# 45 mm is 10 mm less than the physical 55 mm half-width. This gives A* more
# room to find paths through tight corridors while still retaining meaningful
# protection around walls/corners. If testing shows this is still too
# restrictive, try 40 mm. Avoid reducing it aggressively because in-place
# turns are the part of the route where side-wall LiDAR correction helps least.
PLANNER_CLEARANCE_MM = 45.0

# Optional additional planner margin. Keep at 0 initially.
EXTRA_CLEARANCE_MM = 0.0

# Planning grid size in pixels. Larger = faster but coarser.
GRID_STEP_PX = 4

# Small-dogleg smoothing.
# A turn waypoint can be removed when one of its adjacent moves is shorter
# than the measured maze "unit space", provided the direct shortcut is clear.
CORNER_CUT_UNIT_MULTIPLIER = 1.0

# Keep this slightly below 1.0 if you want to require a visibly short move
# before corner cutting. 1.0 means "shorter than one measured unit".
SHORT_MOVE_FRACTION = 1.0

# ------------------------------------------------------------
# Robot-route generation
# ------------------------------------------------------------
# A segment ending with an obstacle directly ahead is tagged WALL.
# Every segment length is converted directly from pixels to millimetres using
# the fixed 40 px = 180 mm scale. The route is NOT quantised to cells.
WALL_LOOKAHEAD_UNITS = 0.75
WALL_LOOKAHEAD_START_UNITS = 0.08
WALL_RAY_HALF_WIDTH_UNITS = 0.10

# A WALL endpoint now requires a real cross-section of dark pixels rather
# than a single dark pixel anywhere in the look-ahead strip.
WALL_MIN_CROSS_SECTION_FRACTION = 0.30
WALL_REQUIRED_CONSECUTIVE_SLICES = 3

# If True, print a serial-ready route string. Serial transmission can be
# added later; for now it is also saved to generated_robot_route.txt.
GENERATE_ROBOT_ROUTE = True

# ------------------------------------------------------------
# ESP32 Wi-Fi route upload
# ------------------------------------------------------------
# The ESP32 creates the Wi-Fi network "MicroMouse".
# Connect the laptop to that network BEFORE pressing ENTER to upload.
ESP32_BASE_URL = "http://192.168.4.1"
ESP32_ROUTE_URL = ESP32_BASE_URL + "/route"

# After a successful upload, automatically open the ESP32 control webpage.
OPEN_START_PAGE_AFTER_UPLOAD = True

# HTTP timeout in seconds.
ESP32_HTTP_TIMEOUT_S = 5.0

# Larger values bias the path more strongly toward the middle of corridors.
CENTERLINE_WEIGHT = 2.5

# Minimum free-space distance (in corrected-image pixels) for a click to be
# accepted directly. If the click lands on an obstacle, the planner will snap
# to the nearest valid free point.
MIN_CLICK_CLEARANCE_PX = 2

# ------------------------------------------------------------
# Screen-fit display settings
# ------------------------------------------------------------
# Every OpenCV image preview is reduced only for DISPLAY when necessary so the
# complete image fits on screen. The underlying processing image stays at its
# original resolution. Interactive mouse coordinates are mapped back through
# the display scale, so selections still refer to the full-resolution image.
DISPLAY_MAX_WIDTH_FRACTION = 0.90
DISPLAY_MAX_HEIGHT_FRACTION = 0.76

_screen_size_cache = None


def get_screen_size_pixels():
    global _screen_size_cache

    if _screen_size_cache is not None:
        return _screen_size_cache

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        _screen_size_cache = (max(640, width), max(480, height))
    except Exception:
        # Safe fallback if the desktop size cannot be queried for any reason.
        _screen_size_cache = (1600, 900)
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass

    return _screen_size_cache


def fit_image_to_screen(image):
    """
    Return (display_image, scale) where display_image is guaranteed to fit
    inside the configured fraction of the current screen.

    scale is display_pixels / source_pixels. It is never greater than 1.0, so
    smaller images are not enlarged unnecessarily.
    """
    if image is None or image.size == 0:
        return image, 1.0

    screen_width, screen_height = get_screen_size_pixels()
    max_width = max(320, int(screen_width * DISPLAY_MAX_WIDTH_FRACTION))
    max_height = max(240, int(screen_height * DISPLAY_MAX_HEIGHT_FRACTION))

    height, width = image.shape[:2]
    scale = min(
        1.0,
        max_width / float(max(1, width)),
        max_height / float(max(1, height)),
    )

    if scale >= 0.9999:
        return image, 1.0

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    return resized, float(scale)


def display_to_source_xy(x, y, display_scale, source_width, source_height):
    """Map mouse coordinates from a fitted preview back to source pixels."""
    safe_scale = max(float(display_scale), 1e-9)

    source_x = int(round(float(x) / safe_scale))
    source_y = int(round(float(y) / safe_scale))

    source_x = max(0, min(source_width - 1, source_x))
    source_y = max(0, min(source_height - 1, source_y))

    return source_x, source_y

# ============================================================
# IMAGE SELECTION
# ============================================================

def choose_image():
    root = tk.Tk()
    root.withdraw()

    filename = filedialog.askopenfilename(
        title="Select top-down micromouse maze image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

    if not filename:
        raise SystemExit("No image selected.")

    return filename


# ============================================================
# FIXED-PROPORTION OCTAGON SELECTION
# ============================================================

# Long side = 2*S
# Diagonal side = S
# Diagonals assumed 45 degrees in top-down view.
OCTAGON_CORNER_FRACTION = (
    1.0 /
    (
        np.sqrt(2.0) *
        (2.0 + np.sqrt(2.0))
    )
)

def octagon_vertices_from_square(left, top, side):
    a = side * OCTAGON_CORNER_FRACTION
    right = left + side
    bottom = top + side

    return np.array(
        [
            [left + a,  top],
            [right - a, top],
            [right,     top + a],
            [right,     bottom - a],
            [right - a, bottom],
            [left + a,  bottom],
            [left,      bottom - a],
            [left,      top + a],
        ],
        dtype=np.float32,
    )


def rotate_points(points, centre, angle_degrees):
    if abs(angle_degrees) < 1e-9:
        return points.copy()

    angle = np.deg2rad(angle_degrees)
    c = np.cos(angle)
    s = np.sin(angle)

    rotation = np.array(
        [
            [c, -s],
            [s,  c],
        ],
        dtype=np.float32,
    )

    centre = np.asarray(centre, dtype=np.float32)

    return ((points - centre) @ rotation.T) + centre


def select_octagon_roi(image):
    window_name = "Fit fixed-proportion OCTAGON to maze"

    height, width = image.shape[:2]
    HANDLE_RADIUS_PX = 18.0

    state = {
        "creating": False,
        "moving": False,
        "resizing": False,
        "drag_start": None,
        "move_start": None,
        "original_left": 0.0,
        "original_top": 0.0,
        "resize_start_distance": 1.0,
        "resize_original_side": 1.0,
        "resize_centre": None,
        "left": None,
        "top": None,
        "side": None,
        "angle": 0.0,
    }

    display = image.copy()
    display_scale = 1.0

    def shape_exists():
        return (
            state["side"] is not None and
            state["left"] is not None and
            state["top"] is not None
        )

    def get_centre():
        return np.array(
            [
                state["left"] + state["side"] / 2.0,
                state["top"] + state["side"] / 2.0,
            ],
            dtype=np.float32,
        )

    def get_vertices():
        if not shape_exists():
            return None

        vertices = octagon_vertices_from_square(
            state["left"],
            state["top"],
            state["side"],
        )

        return rotate_points(
            vertices,
            get_centre(),
            state["angle"],
        )

    def point_inside_octagon(x, y):
        vertices = get_vertices()
        if vertices is None:
            return False

        polygon = np.round(vertices).astype(np.int32)
        return cv2.pointPolygonTest(
            polygon,
            (float(x), float(y)),
            False
        ) >= 0

    def nearest_handle_index(x, y):
        vertices = get_vertices()
        if vertices is None:
            return None

        point = np.array([x, y], dtype=np.float32)
        distances = np.linalg.norm(vertices - point, axis=1)
        index = int(np.argmin(distances))

        if distances[index] <= HANDLE_RADIUS_PX:
            return index

        return None

    def maximum_side_for_centre(centre):
        cx, cy = centre

        half_max = min(
            cx,
            cy,
            width - 1 - cx,
            height - 1 - cy,
        )

        return max(80.0, 2.0 * half_max)

    def set_side_about_centre(new_side, centre=None):
        if not shape_exists():
            return

        if centre is None:
            centre = get_centre()

        centre = np.asarray(centre, dtype=np.float32)
        max_side = maximum_side_for_centre(centre)

        new_side = float(max(80.0, min(max_side, new_side)))

        state["side"] = new_side
        state["left"] = float(centre[0] - new_side / 2.0)
        state["top"] = float(centre[1] - new_side / 2.0)

    def clamp_position():
        if not shape_exists():
            return

        state["side"] = max(80.0, state["side"])

        state["left"] = max(
            0.0,
            min(float(width) - state["side"], state["left"])
        )

        state["top"] = max(
            0.0,
            min(float(height) - state["side"], state["top"])
        )

    def redraw():
        nonlocal display
        display = image.copy()
        vertices = get_vertices()

        if vertices is not None:
            polygon = np.round(vertices).astype(np.int32)

            overlay = display.copy()
            cv2.fillPoly(overlay, [polygon], (0, 255, 0))
            cv2.addWeighted(overlay, 0.10, display, 0.90, 0.0, display)

            cv2.polylines(
                display,
                [polygon],
                True,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

            for point in polygon:
                cv2.circle(display, tuple(point), 8, (0, 255, 255), -1)
                cv2.circle(
                    display,
                    tuple(point),
                    int(HANDLE_RADIUS_PX),
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            centre = tuple(np.round(get_centre()).astype(int))
            cv2.drawMarker(
                display,
                centre,
                (255, 0, 255),
                cv2.MARKER_CROSS,
                20,
                2,
            )

            physical_s = state["side"] / (2.0 + np.sqrt(2.0))
            long_side_px = 2.0 * physical_s
            diagonal_side_px = physical_s

            info = (
                f"2S sides={long_side_px:.0f}px  "
                f"S diagonals={diagonal_side_px:.0f}px  "
                f"rot={state['angle']:.1f}deg"
            )

            cv2.putText(
                display,
                info,
                (15, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            instruction_1 = "Drag INSIDE=move | drag YELLOW HANDLE=uniform resize"
        else:
            instruction_1 = "Drag once to CREATE the octagon"

        cv2.putText(
            display,
            instruction_1,
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            "[ ]=rotate | W/A/S/D=fine move | -=shrink | ==grow",
            (15, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            "ENTER=accept | R=delete/reset | C=cancel",
            (15, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def mouse_callback(event, x, y, flags, param):
        x, y = display_to_source_xy(
            x, y, display_scale, width, height
        )

        if event == cv2.EVENT_LBUTTONDOWN:
            if not shape_exists():
                state["creating"] = True
                state["drag_start"] = (x, y)

                state["left"] = float(x)
                state["top"] = float(y)
                state["side"] = 1.0
                state["angle"] = 0.0
                redraw()
                return

            handle_index = nearest_handle_index(x, y)

            if handle_index is not None:
                centre = get_centre()
                state["resizing"] = True
                state["resize_centre"] = centre.copy()
                state["resize_original_side"] = state["side"]

                distance = np.linalg.norm(
                    np.array([x, y], dtype=np.float32) - centre
                )

                state["resize_start_distance"] = max(1.0, float(distance))
                return

            if point_inside_octagon(x, y):
                state["moving"] = True
                state["move_start"] = (x, y)
                state["original_left"] = state["left"]
                state["original_top"] = state["top"]
                return

            return

        elif event == cv2.EVENT_MOUSEMOVE:
            if state["creating"]:
                x0, y0 = state["drag_start"]
                dx = x - x0
                dy = y - y0
                side = max(abs(dx), abs(dy))
                sx = 1 if dx >= 0 else -1
                sy = 1 if dy >= 0 else -1

                x1 = x0 + sx * side
                y1 = y0 + sy * side

                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))

                actual_side = min(abs(x1 - x0), abs(y1 - y0))

                x1 = x0 + sx * actual_side
                y1 = y0 + sy * actual_side

                state["left"] = float(min(x0, x1))
                state["top"] = float(min(y0, y1))
                state["side"] = float(max(1.0, actual_side))

                clamp_position()
                redraw()

            elif state["moving"]:
                x0, y0 = state["move_start"]

                state["left"] = state["original_left"] + (x - x0)
                state["top"] = state["original_top"] + (y - y0)

                clamp_position()
                redraw()

            elif state["resizing"]:
                centre = state["resize_centre"]

                current_distance = np.linalg.norm(
                    np.array([x, y], dtype=np.float32) - centre
                )

                scale = float(current_distance) / state["resize_start_distance"]
                new_side = state["resize_original_side"] * scale

                set_side_about_centre(new_side, centre)
                redraw()

        elif event == cv2.EVENT_LBUTTONUP:
            state["creating"] = False
            state["moving"] = False
            state["resizing"] = False

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)
    redraw()

    while True:
        fitted_display, display_scale = fit_image_to_screen(display)
        cv2.imshow(window_name, fitted_display)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):
            if shape_exists() and state["side"] >= 80:
                break

        elif key in (ord("r"), ord("R")):
            state["left"] = None
            state["top"] = None
            state["side"] = None
            state["angle"] = 0.0
            state["creating"] = False
            state["moving"] = False
            state["resizing"] = False
            redraw()

        elif key == ord("["):
            if shape_exists():
                state["angle"] -= 0.5
                redraw()

        elif key == ord("]"):
            if shape_exists():
                state["angle"] += 0.5
                redraw()

        elif key in (ord("a"), ord("A")):
            if shape_exists():
                state["left"] -= 1.0
                clamp_position()
                redraw()

        elif key in (ord("d"), ord("D")):
            if shape_exists():
                state["left"] += 1.0
                clamp_position()
                redraw()

        elif key in (ord("w"), ord("W")):
            if shape_exists():
                state["top"] -= 1.0
                clamp_position()
                redraw()

        elif key in (ord("s"), ord("S")):
            if shape_exists():
                state["top"] += 1.0
                clamp_position()
                redraw()

        elif key in (ord("-"), ord("_")):
            if shape_exists():
                centre = get_centre()
                set_side_about_centre(state["side"] - 2.0, centre)
                redraw()

        elif key in (ord("="), ord("+")):
            if shape_exists():
                centre = get_centre()
                set_side_about_centre(state["side"] + 2.0, centre)
                redraw()

        elif key in (ord("c"), ord("C"), 27):
            cv2.destroyWindow(window_name)
            raise SystemExit("Maze octagon selection cancelled.")

    cv2.destroyWindow(window_name)

    centre = tuple(get_centre())
    rotation_matrix = cv2.getRotationMatrix2D(centre, -state["angle"], 1.0)

    straightened = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    left = int(round(state["left"]))
    top = int(round(state["top"]))
    side = int(round(state["side"]))

    crop = straightened[top:top + side, left:left + side].copy()

    if crop.size == 0:
        raise SystemExit("Octagon crop failed.")

    local_vertices = octagon_vertices_from_square(0.0, 0.0, float(side))
    local_polygon = np.round(local_vertices).astype(np.int32)

    mask = np.zeros((side, side), dtype=np.uint8)
    cv2.fillPoly(mask, [local_polygon], 255)

    white_background = np.full_like(crop, 255)
    crop = np.where(mask[:, :, None] == 255, crop, white_background)

    return crop


def select_maze_roi(image):
    print()
    print("A window will open.")
    print("Fit the GREEN OCTAGON to the OUTSIDE maze boundary.")
    print("Its proportions are locked:")
    print("  horizontal/vertical sides = 2 x S")
    print("  diagonal corner sides     = 1 x S")
    print()
    print("Drag ONCE to create it.")
    print("After that, drag inside it to move it.")
    print("Drag any yellow corner handle to resize the SAME octagon.")
    print("Clicking outside will NOT create a new one.")
    print("Use [ and ] for small rotation corrections.")
    print("Press ENTER or SPACE when it matches the maze.")

    return select_octagon_roi(image)


# ============================================================
# OBSTACLE DETECTION
# ============================================================

def detect_obstacles(maze_image, black_threshold):
    gray = cv2.cvtColor(maze_image, cv2.COLOR_BGR2GRAY)

    # True black walls + black cylinders.
    #
    # The threshold is fixed at OBSTACLE_BLACK_THRESHOLD for every image.
    obstacle_mask = gray < black_threshold
    obstacle_mask = (obstacle_mask.astype(np.uint8) * 255)

    # Close only tiny 1-2 pixel breaks in true black wall lines.
    # This is deliberately small so we do not grow grey shadows into walls.
    kernel = np.ones((3, 3), np.uint8)
    obstacle_mask = cv2.morphologyEx(
        obstacle_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return obstacle_mask


def calculate_robot_clearance_px(unit_space_px):
    """
    Convert the robot's physical half-width into image pixels.

    The fixed UNIT_SPACE_PX value represents one 180 mm maze unit.

        pixels_per_mm = unit_space_px / 180

    The route represents the robot centre. For this feedback-assisted version,
    obstacles are expanded by the dedicated PLANNER_CLEARANCE_MM value rather
    than the full physical half-width.

    Physical robot half-width = 55 mm
    Current planner clearance = 45 mm
    """

    pixels_per_mm = unit_space_px / CELL_SIZE_MM

    clearance_mm = (
        PLANNER_CLEARANCE_MM +
        EXTRA_CLEARANCE_MM
    )

    clearance_px = clearance_mm * pixels_per_mm

    return clearance_px


def inflate_obstacles(obstacle_mask, clearance_px):
    """
    Expand detected walls/circular obstacles by the physical radius of
    the robot expressed in image pixels.

    This makes the path planner operate on the ROBOT CENTRELINE rather
    than pretending the robot is a zero-width point.
    """

    # A dilation radius of R pixels needs an approximately (2R+1) kernel.
    radius_px = max(
        1,
        int(round(clearance_px))
    )

    kernel_size = 2 * radius_px + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    inflated = cv2.dilate(
        obstacle_mask,
        kernel,
        iterations=1,
    )

    return inflated


# ============================================================
# START / GOAL SELECTION
# ============================================================

def nearest_free_point(free_mask, x, y):
    if (
        0 <= y < free_mask.shape[0] and
        0 <= x < free_mask.shape[1] and
        free_mask[y, x] > 0
    ):
        return (x, y)

    free_points = np.column_stack(np.where(free_mask > 0))
    if free_points.size == 0:
        return None

    # free_points are [y, x]
    deltas = free_points - np.array([y, x])
    distances_sq = np.sum(deltas * deltas, axis=1)
    index = int(np.argmin(distances_sq))

    best_y, best_x = free_points[index]
    return (int(best_x), int(best_y))


def select_start_goal_points(maze_image, inflated_obstacles, unit_space_px):
    """
    Select START, GOAL, and the robot's actual starting orientation.

    Workflow:
      1. Click START.
      2. Click GOAL.
      3. Drag around START to rotate the cyan heading arrow until it matches
         the direction the physical robot is facing.
      4. Use [ and ] for one-degree fine adjustment if desired.
      5. Press ENTER / SPACE to accept.

    Returns:
        start_point
        goal_point
        start_heading_vector  unit image-coordinate vector [dx, dy]
    """
    free_mask = (inflated_obstacles == 0).astype(np.uint8) * 255

    points = []
    window_name = "Click START, GOAL, then set ROBOT FACING arrow"

    # Reserve a dedicated header area ABOVE the maze image so the instruction
    # text never covers the selectable maze itself.
    header_height = 96

    # Image-coordinate unit vector. +x = right, +y = down.
    # A default arrow is shown after START is selected, but the user must
    # deliberately set/adjust it after GOAL before ENTER is accepted.
    heading_vector = np.array([1.0, 0.0], dtype=float)
    heading_set = False
    dragging_heading = False
    display_scale = 1.0

    arrow_length_px = int(
        round(max(40.0, min(120.0, 0.45 * float(unit_space_px))))
    )

    def window_to_image_coords(x, y):
        img_x = int(x)
        img_y = int(y - header_height)

        if img_x < 0 or img_y < 0:
            return None

        if img_x >= maze_image.shape[1] or img_y >= maze_image.shape[0]:
            return None

        return img_x, img_y

    def to_canvas_point(point):
        return (int(point[0]), int(point[1] + header_height))

    def set_heading_from_mouse(x, y):
        nonlocal heading_vector, heading_set

        if len(points) < 1:
            return

        image_point = window_to_image_coords(x, y)
        if image_point is None:
            return

        dx = float(image_point[0] - points[0][0])
        dy = float(image_point[1] - points[0][1])
        norm = float(np.hypot(dx, dy))

        # Ignore clicks essentially on top of START so the direction remains
        # well-defined.
        if norm < 5.0:
            return

        heading_vector = np.array([dx / norm, dy / norm], dtype=float)
        heading_set = True

    def rotate_heading(delta_degrees):
        nonlocal heading_vector, heading_set

        if len(points) < 2:
            return

        angle = np.deg2rad(float(delta_degrees))
        c = float(np.cos(angle))
        s = float(np.sin(angle))

        x = float(heading_vector[0])
        y = float(heading_vector[1])

        # Rotation is in image coordinates, where +y points downward.
        heading_vector = np.array(
            [c * x - s * y, s * x + c * y],
            dtype=float,
        )
        heading_set = True

    def mouse_callback(event, x, y, flags, param):
        nonlocal dragging_heading, heading_set

        canvas_height = maze_image.shape[0] + header_height
        canvas_width = maze_image.shape[1]
        x, y = display_to_source_xy(
            x, y, display_scale, canvas_width, canvas_height
        )

        image_point = window_to_image_coords(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                if image_point is None:
                    return

                snapped = nearest_free_point(free_mask, image_point[0], image_point[1])
                if snapped is not None:
                    points.append(snapped)

                    # When a new START is chosen, reset the visible default
                    # heading to the right. It becomes confirmed only after
                    # the user rotates/sets it.
                    if len(points) == 1:
                        heading_set = False

                return

            # START and GOAL already exist: mouse drag now rotates the arrow.
            if image_point is None:
                return

            dragging_heading = True
            set_heading_from_mouse(x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE and dragging_heading:
            set_heading_from_mouse(x, y)
            return

        if event == cv2.EVENT_LBUTTONUP and dragging_heading:
            set_heading_from_mouse(x, y)
            dragging_heading = False

    # The displayed canvas is explicitly fitted to the screen below. Mouse
    # coordinates are mapped back to this full-resolution canvas.
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        maze_display = maze_image.copy()

        # Show inflated obstacle boundary in yellow.
        contours, _ = cv2.findContours(
            inflated_obstacles,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(maze_display, contours, -1, (0, 255, 255), 1)

        if len(points) > 0:
            start = points[0]
            cv2.circle(maze_display, start, 8, (255, 0, 0), -1)
            cv2.putText(
                maze_display,
                "START",
                (start[0] + 8, start[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

            # Heading arrow is always visible once START exists.
            tip = (
                int(round(start[0] + heading_vector[0] * arrow_length_px)),
                int(round(start[1] + heading_vector[1] * arrow_length_px)),
            )

            arrow_colour = (255, 255, 0) if heading_set else (180, 180, 0)
            cv2.arrowedLine(
                maze_display,
                start,
                tip,
                arrow_colour,
                4,
                cv2.LINE_AA,
                tipLength=0.25,
            )
            cv2.putText(
                maze_display,
                "ROBOT FACING",
                (tip[0] + 8, tip[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                arrow_colour,
                2,
                cv2.LINE_AA,
            )

        if len(points) > 1:
            cv2.circle(maze_display, points[1], 8, (0, 255, 255), -1)
            cv2.putText(
                maze_display,
                "GOAL",
                (points[1][0] + 8, points[1][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if len(points) == 0:
            message = "1/3  Click START point"
            second_message = ""
        elif len(points) == 1:
            message = "2/3  Click GOAL point"
            second_message = "The heading arrow becomes adjustable after GOAL is selected."
        elif not heading_set:
            message = "3/3  Drag around START to set ROBOT FACING direction"
            second_message = "[ / ] = fine rotate 1 degree | R = reset"
        else:
            # Show a useful signed orientation diagnostic. This is only a
            # display angle; route turning still uses vector-to-vector maths.
            physical_heading_deg = float(
                np.degrees(np.arctan2(-heading_vector[1], heading_vector[0]))
            )
            message = (
                f"Facing set: {physical_heading_deg:+.1f} deg  |  "
                "drag=adjust | [ ]=1 deg | ENTER=accept"
            )
            second_message = "Positive displayed angle is counter-clockwise from image-right."

        # Build a canvas with a white header area above the maze so the maze
        # remains fully visible and clickable all the way to its top edge.
        display = np.full(
            (maze_image.shape[0] + header_height, maze_image.shape[1], 3),
            255,
            dtype=np.uint8,
        )
        display[header_height:, :, :] = maze_display

        cv2.putText(
            display,
            message,
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        if second_message:
            cv2.putText(
                display,
                second_message,
                (14, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            display,
            "START/GOAL snap to nearest valid free point if necessary.",
            (14, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        fitted_display, display_scale = fit_image_to_screen(display)
        cv2.imshow(window_name, fitted_display)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32) and len(points) == 2 and heading_set:
            break
        elif key == ord("[") and len(points) == 2:
            rotate_heading(-1.0)
        elif key == ord("]") and len(points) == 2:
            rotate_heading(+1.0)
        elif key in (ord("r"), ord("R")):
            points.clear()
            heading_vector = np.array([1.0, 0.0], dtype=float)
            heading_set = False
            dragging_heading = False
        elif key in (ord("c"), ord("C"), 27):
            cv2.destroyWindow(window_name)
            raise SystemExit("Start/goal/orientation selection cancelled.")

    cv2.destroyWindow(window_name)
    return points[0], points[1], heading_vector.copy()



# ============================================================
# A* PLANNING
# ============================================================

def build_planning_grid(inflated_obstacles):
    h, w = inflated_obstacles.shape

    grid_h = h // GRID_STEP_PX
    grid_w = w // GRID_STEP_PX

    if grid_h < 10 or grid_w < 10:
        raise SystemExit("Planning grid became too small.")

    occ_small = cv2.resize(
        inflated_obstacles,
        (grid_w, grid_h),
        interpolation=cv2.INTER_AREA,
    )

    occupied = occ_small > 8

    # Distance transform on full image then resize.
    free_full = (inflated_obstacles == 0).astype(np.uint8) * 255
    distance_full = cv2.distanceTransform(
        free_full,
        cv2.DIST_L2,
        5,
    )

    clearance_small = cv2.resize(
        distance_full,
        (grid_w, grid_h),
        interpolation=cv2.INTER_AREA,
    )

    return occupied, clearance_small


def image_point_to_grid(point):
    x, y = point
    return (
        int(round(y / GRID_STEP_PX)),
        int(round(x / GRID_STEP_PX)),
    )


def grid_point_to_image(rc):
    r, c = rc
    x = int(round((c + 0.5) * GRID_STEP_PX))
    y = int(round((r + 0.5) * GRID_STEP_PX))
    return (x, y)


def snap_grid_point_to_free(occupied, rc):
    rows, cols = occupied.shape
    r0, c0 = rc

    if (
        0 <= r0 < rows and
        0 <= c0 < cols and
        not occupied[r0, c0]
    ):
        return rc

    free_points = np.column_stack(np.where(~occupied))
    if free_points.size == 0:
        return None

    deltas = free_points - np.array([r0, c0])
    distances_sq = np.sum(deltas * deltas, axis=1)
    index = int(np.argmin(distances_sq))
    best_r, best_c = free_points[index]
    return (int(best_r), int(best_c))


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_path(occupied, clearance, start, goal):
    rows, cols = occupied.shape

    start = snap_grid_point_to_free(occupied, start)
    goal = snap_grid_point_to_free(occupied, goal)

    if start is None or goal is None:
        return []

    clearance_norm = clearance.copy()
    max_clear = float(np.max(clearance_norm))

    if max_clear > 1e-6:
        clearance_norm /= max_clear
    else:
        clearance_norm.fill(0.0)

    open_heap = []
    heapq.heappush(open_heap, (0.0, start))

    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    neighbours = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        closed.add(current)

        for dr, dc in neighbours:
            nr = current[0] + dr
            nc = current[1] + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            if occupied[nr, nc]:
                continue

            # Prefer cells with large clearance (toward corridor centre).
            clearance_penalty = (
                1.0 - float(clearance_norm[nr, nc])
            )

            step_cost = 1.0 + CENTERLINE_WEIGHT * clearance_penalty
            tentative_g = g_score[current] + step_cost

            neighbour = (nr, nc)

            if tentative_g < g_score.get(neighbour, float("inf")):
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g

                f_score = tentative_g + heuristic(neighbour, goal)
                heapq.heappush(open_heap, (f_score, neighbour))

    return []


def simplify_grid_path(path):
    """
    First-stage simplification:
    keep only points where the grid direction changes.
    """
    if len(path) <= 2:
        return path[:]

    simplified = [path[0]]

    previous_direction = (
        path[1][0] - path[0][0],
        path[1][1] - path[0][1],
    )

    for i in range(1, len(path) - 1):
        new_direction = (
            path[i + 1][0] - path[i][0],
            path[i + 1][1] - path[i][1],
        )

        if new_direction != previous_direction:
            simplified.append(path[i])

        previous_direction = new_direction

    simplified.append(path[-1])
    return simplified


def line_is_clear(point_a, point_b, inflated_obstacles):
    """
    Check whether a straight shortcut between two image points remains entirely
    inside free space.

    inflated_obstacles already includes robot clearance, so a clear line here
    means the robot centreline has the requested safety margin.
    """

    x0, y0 = point_a
    x1, y1 = point_b

    length = max(
        2,
        int(np.ceil(np.hypot(x1 - x0, y1 - y0)))
    )

    xs = np.linspace(x0, x1, length)
    ys = np.linspace(y0, y1, length)

    h, w = inflated_obstacles.shape

    for xf, yf in zip(xs, ys):
        x = int(round(xf))
        y = int(round(yf))

        if not (0 <= x < w and 0 <= y < h):
            return False

        if inflated_obstacles[y, x] != 0:
            return False

    return True


def remove_short_doglegs(
    waypoints,
    inflated_obstacles,
    unit_space_px,
):
    """
    Remove unnecessary small corner manoeuvres.

    For three consecutive turning points A -> B -> C:

      if AB < one measured maze unit OR BC < one measured maze unit,

    then B is removed IF the direct A -> C shortcut stays completely clear of
    the inflated walls/cylinders.

    The process repeats until no more safe short doglegs can be removed.

    This turns patterns such as:

        right
        tiny forward movement
        left

    into one direct diagonal/straight shortcut whenever there is room.
    """

    if len(waypoints) <= 2:
        return waypoints[:]

    short_threshold = (
        unit_space_px *
        CORNER_CUT_UNIT_MULTIPLIER *
        SHORT_MOVE_FRACTION
    )

    result = waypoints[:]

    changed = True

    while changed and len(result) > 2:
        changed = False

        i = 1

        while i < len(result) - 1:
            a = result[i - 1]
            b = result[i]
            c = result[i + 1]

            ab = float(np.hypot(
                b[0] - a[0],
                b[1] - a[1],
            ))

            bc = float(np.hypot(
                c[0] - b[0],
                c[1] - b[1],
            ))

            has_short_move = (
                ab < short_threshold or
                bc < short_threshold
            )

            if (
                has_short_move and
                line_is_clear(
                    a,
                    c,
                    inflated_obstacles,
                )
            ):
                del result[i]
                changed = True

                # Re-check the new corner formed at the previous point.
                if i > 1:
                    i -= 1
            else:
                i += 1

    return result


# ============================================================
# HYBRID ROBOT ROUTE GENERATION
# ============================================================

def segment_length_px(a, b):
    return float(np.hypot(b[0] - a[0], b[1] - a[1]))


def signed_turn_degrees(previous_vector, next_vector):
    """
    Signed turn from previous path direction to next path direction.

    Image y increases downward, so the usual Cartesian cross-product sign is
    inverted. Positive returned angle is LEFT / counter-clockwise for the
    physical robot; negative is RIGHT / clockwise.
    """
    p = np.asarray(previous_vector, dtype=float)
    n = np.asarray(next_vector, dtype=float)

    p_norm = np.linalg.norm(p)
    n_norm = np.linalg.norm(n)

    if p_norm < 1e-9 or n_norm < 1e-9:
        return 0.0

    p /= p_norm
    n /= n_norm

    dot = float(np.clip(np.dot(p, n), -1.0, 1.0))
    # Cartesian cross would be p_x*n_y - p_y*n_x; invert because image y is down.
    cross_image = p[0] * n[1] - p[1] * n[0]
    cross_robot = -cross_image

    angle = np.degrees(np.arctan2(cross_robot, dot))
    return float(angle)


def obstacle_directly_ahead(endpoint, direction, obstacle_mask, unit_space_px):
    """
    Robust WALL/DISTANCE endpoint classification.

    The older version returned WALL as soon as ONE black pixel appeared
    anywhere in the look-ahead strip. That made it vulnerable to shadows,
    wall edges, isolated pixels and side-wall contamination.

    This version samples a perpendicular cross-section at each forward
    distance and requires:
        - at least WALL_MIN_CROSS_SECTION_FRACTION of that slice to be black,
        - for WALL_REQUIRED_CONSECUTIVE_SLICES consecutive forward slices.

    Returns:
        wall_end             bool
        first_wall_units     first confirmed wall distance / unit size, or None
        max_coverage         maximum black fraction observed in any slice
    """
    d = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-9:
        return False, None, 0.0

    d /= norm
    perp = np.array([-d[1], d[0]], dtype=float)

    start_d = WALL_LOOKAHEAD_START_UNITS * unit_space_px
    end_d = WALL_LOOKAHEAD_UNITS * unit_space_px
    half_w = WALL_RAY_HALF_WIDTH_UNITS * unit_space_px

    h, w = obstacle_mask.shape

    distances = np.linspace(
        start_d,
        end_d,
        max(2, int(end_d - start_d) + 1)
    )

    offsets = np.linspace(
        -half_w,
        half_w,
        max(5, int(2 * half_w) + 1)
    )

    ep = np.asarray(endpoint, dtype=float)

    consecutive = 0
    first_candidate_distance = None
    max_coverage = 0.0

    for dist in distances:
        centre = ep + d * dist

        occupied_count = 0
        valid_count = 0

        for offset in offsets:
            q = centre + perp * offset
            x = int(round(q[0]))
            y = int(round(q[1]))

            # Leaving the cropped octagon counts as a solid outer wall.
            if not (0 <= x < w and 0 <= y < h):
                occupied_count += 1
                valid_count += 1
                continue

            valid_count += 1

            if obstacle_mask[y, x] != 0:
                occupied_count += 1

        coverage = (
            float(occupied_count) / float(valid_count)
            if valid_count > 0
            else 0.0
        )

        max_coverage = max(max_coverage, coverage)

        if coverage >= WALL_MIN_CROSS_SECTION_FRACTION:
            if consecutive == 0:
                first_candidate_distance = dist

            consecutive += 1

            if consecutive >= WALL_REQUIRED_CONSECUTIVE_SLICES:
                first_wall_units = (
                    first_candidate_distance / unit_space_px
                    if unit_space_px > 1e-6
                    else None
                )
                return True, first_wall_units, max_coverage
        else:
            consecutive = 0
            first_candidate_distance = None

    return False, None, max_coverage


def segment_distance_mm(length_px, unit_space_px):
    """
    Convert an arbitrary path-segment pixel length directly to millimetres.

    The fixed image scale defines:
        unit_space_px pixels = CELL_SIZE_MM (180 mm)

    No rounding to whole maze cells is performed.
    """
    if unit_space_px <= 1e-6:
        raise ValueError("unit_space_px must be positive")

    units_float = float(length_px) / float(unit_space_px)
    distance_mm = units_float * CELL_SIZE_MM

    return max(1.0, float(distance_mm)), units_float


def build_hybrid_robot_route(
    smoothed_waypoints,
    obstacle_mask,
    unit_space_px,
    initial_heading_vector=None,
):
    """
    Convert final path waypoints into robot segments.

    Each segment contains:
      turn_deg       : relative turn BEFORE driving this segment
      mode           : WALL or DISTANCE
      distance_mm    : exact physical distance derived from image scale
      length_px      : original image length (diagnostic only)
      units_float    : unrounded pixel / measured-180-mm-unit ratio

    If initial_heading_vector is supplied, the FIRST segment turn is calculated
    from the robot's selected real start orientation to the first path segment.
    Later segment turns remain relative to the preceding path segment.
    """
    if len(smoothed_waypoints) < 2:
        return []

    route = []

    if initial_heading_vector is None:
        previous_vector = None
    else:
        previous_vector = np.asarray(initial_heading_vector, dtype=float).copy()
        heading_norm = float(np.linalg.norm(previous_vector))
        if heading_norm < 1e-9:
            raise ValueError("initial_heading_vector must be non-zero")
        previous_vector /= heading_norm

    for i in range(len(smoothed_waypoints) - 1):
        a = smoothed_waypoints[i]
        b = smoothed_waypoints[i + 1]

        vector = np.array(
            [b[0] - a[0], b[1] - a[1]],
            dtype=float,
        )

        length_px = segment_length_px(a, b)
        distance_mm, units_float = segment_distance_mm(
            length_px, unit_space_px
        )

        if previous_vector is None:
            turn_deg = 0.0
        else:
            turn_deg = signed_turn_degrees(previous_vector, vector)

        (
            wall_end,
            wall_distance_units,
            wall_max_coverage,
        ) = obstacle_directly_ahead(
            b,
            vector,
            obstacle_mask,
            unit_space_px,
        )

        mode = "WALL" if wall_end else "DISTANCE"

        route.append(
            {
                "turn_deg": turn_deg,
                "mode": mode,
                "distance_mm": distance_mm,
                "length_px": length_px,
                "units_float": units_float,
                "wall_distance_units": wall_distance_units,
                "wall_max_coverage": wall_max_coverage,
                "start": a,
                "end": b,
            }
        )

        previous_vector = vector

    return route


def route_to_serial_string(route):
    """
    Compact ESP32 serial format:

      ROUTE:turn,mode,distance_mm;turn,mode,distance_mm;...

    mode is W for front-LiDAR wall termination or D for distance termination.
    The third field is the direct physical segment distance in millimetres.
    Example:
      ROUTE:0.0,D,243.0;-90.0,W,176.4;90.0,D,392.4
    """
    chunks = []

    for item in route:
        mode_letter = "W" if item["mode"] == "WALL" else "D"
        chunks.append(
            f'{item["turn_deg"]:.1f},{mode_letter},{item["distance_mm"]:.1f}'
        )

    return "ROUTE:" + ";".join(chunks)


def draw_route_labels(image, route):
    result = image.copy()

    for index, item in enumerate(route):
        a = item["start"]
        b = item["end"]
        midpoint = (
            int(round((a[0] + b[0]) / 2)),
            int(round((a[1] + b[1]) / 2)),
        )

        mode_letter = "W" if item["mode"] == "WALL" else "D"
        label = f'{index+1}:{mode_letter}{item["distance_mm"]:.0f}mm'

        cv2.putText(
            result,
            label,
            (midpoint[0] + 5, midpoint[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return result


# ============================================================
# ESP32 WI-FI ROUTE TRANSFER
# ============================================================

def upload_route_over_wifi(serial_route):
    """
    Upload the generated route directly to the ESP32 HTTP server.

    The laptop must already be connected to the ESP32 access point:
        Wi-Fi name: MicroMouse

    The Arduino /route endpoint accepts the same ROUTE:... string that the
    earlier USB serial implementation used.
    """

    payload = serial_route.encode("utf-8")

    request = urllib.request.Request(
        ESP32_ROUTE_URL,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "text/plain; charset=utf-8",
            "Content-Length": str(len(payload)),
        },
    )

    print()
    print("Uploading route to MicroMouse over Wi-Fi...")
    print("Target:", ESP32_ROUTE_URL)

    try:
        with urllib.request.urlopen(
            request,
            timeout=ESP32_HTTP_TIMEOUT_S,
        ) as response:
            body = response.read().decode(
                "utf-8",
                errors="replace",
            ).strip()

            status = response.getcode()

    except urllib.error.HTTPError as exc:
        body = exc.read().decode(
            "utf-8",
            errors="replace",
        ).strip()

        print()
        print(f"ESP32 rejected route upload (HTTP {exc.code}).")
        if body:
            print("ESP32 response:", body)
        return False

    except urllib.error.URLError as exc:
        print()
        print("Could not reach the ESP32 Wi-Fi server.")
        print("Make sure this laptop is connected to the Wi-Fi network:")
        print("    MicroMouse")
        print("and that the ESP32 is powered on.")
        print("Error:", exc.reason)
        return False

    except TimeoutError:
        print()
        print("Timed out waiting for the ESP32.")
        print("Check that the laptop is connected to MicroMouse Wi-Fi.")
        return False

    if status != 200:
        print(f"Unexpected ESP32 HTTP status: {status}")
        if body:
            print(body)
        return False

    print()
    print("========================================")
    print("ROUTE UPLOADED TO ESP32 SUCCESSFULLY")
    print("========================================")
    if body:
        print("ESP32:", body)

    print()
    print("The robot has NOT started.")
    print("Place it at the selected START position in the orientation shown by")
    print("the ROBOT FACING arrow. The first route command will rotate it onto")
    print("the first red path segment automatically.")
    print("Then open the MicroMouse webpage and press START.")
    print("Control page:", ESP32_BASE_URL)

    if OPEN_START_PAGE_AFTER_UPLOAD:
        try:
            webbrowser.open(ESP32_BASE_URL)
        except Exception:
            # Browser opening is only a convenience; upload already succeeded.
            pass

    return True


def wait_for_upload_confirmation(result_image, serial_route):
    """
    Show the final planned route and require an explicit ENTER before the
    route is transmitted to the robot.

    The instruction text lives in a separate header ABOVE the path image.
    The complete canvas is fitted to the current screen before display.

    ENTER / SPACE : upload route over Wi-Fi
    Q / ESC       : close without uploading
    """

    window_name = "Generated maze path - ENTER to upload to MicroMouse"
    header_height = 80

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        # Create extra canvas space above the result rather than painting a
        # white rectangle over the top of the route image.
        display = np.full(
            (result_image.shape[0] + header_height, result_image.shape[1], 3),
            255,
            dtype=np.uint8,
        )
        display[header_height:, :, :] = result_image

        cv2.putText(
            display,
            "ENTER = upload route to MicroMouse Wi-Fi",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            "Q / ESC = close without uploading",
            (14, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        fitted_display, _ = fit_image_to_screen(display)
        cv2.imshow(window_name, fitted_display)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):  # ENTER / SPACE
            success = upload_route_over_wifi(serial_route)

            if success:
                cv2.destroyWindow(window_name)
                return True

            print()
            print("Upload failed. The path window is still open.")
            print("Connect to MicroMouse Wi-Fi and press ENTER to retry.")

        elif key in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(window_name)
            print("Route was NOT uploaded to the ESP32.")
            return False


# ============================================================
# VISUALISATION + SAVING
# ============================================================

def draw_path_overlay(
    maze_image,
    full_path_img,
    smoothed_path_img,
    start_point,
    goal_point,
    start_heading_vector=None,
    unit_space_px=None,
):
    result = maze_image.copy()

    # Draw ONLY the final smoothed route.
    #
    # The earlier version also drew the raw A* path underneath this route.
    # When corner cutting changed the path slightly, both red paths remained
    # visible and looked like multiple routes between the same waypoints.
    for i in range(len(smoothed_path_img) - 1):
        cv2.line(
            result,
            smoothed_path_img[i],
            smoothed_path_img[i + 1],
            (0, 0, 255),
            4,
            cv2.LINE_AA,
        )

    # Final turning points in green.
    for point in smoothed_path_img:
        cv2.circle(result, point, 6, (0, 255, 0), -1)

    cv2.circle(result, start_point, 10, (255, 0, 0), 2)
    cv2.circle(result, goal_point, 10, (0, 255, 255), 2)

    cv2.putText(
        result,
        "START",
        (start_point[0] + 10, start_point[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        result,
        "GOAL",
        (goal_point[0] + 10, goal_point[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if start_heading_vector is not None:
        heading = np.asarray(start_heading_vector, dtype=float)
        heading_norm = float(np.linalg.norm(heading))

        if heading_norm > 1e-9:
            heading = heading / heading_norm
            base_length = 70.0 if unit_space_px is None else 0.45 * float(unit_space_px)
            arrow_length_px = int(round(max(40.0, min(120.0, base_length))))

            tip = (
                int(round(start_point[0] + heading[0] * arrow_length_px)),
                int(round(start_point[1] + heading[1] * arrow_length_px)),
            )

            cv2.arrowedLine(
                result,
                start_point,
                tip,
                (255, 255, 0),
                4,
                cv2.LINE_AA,
                tipLength=0.25,
            )

            cv2.putText(
                result,
                "START FACING",
                (tip[0] + 8, tip[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    image_path = choose_image()

    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"Could not open image: {image_path}")

    maze_image = select_maze_roi(image)

    output_dir = Path(image_path).resolve().parent

    corrected_preview = output_dir / "generated_corrected_maze.png"
    cv2.imwrite(str(corrected_preview), maze_image)

    # Fixed image-processing parameters.
    # No manual black-threshold or maze-unit-width calibration is shown.
    black_threshold = OBSTACLE_BLACK_THRESHOLD
    unit_space_px = UNIT_SPACE_PX

    obstacle_mask = detect_obstacles(
        maze_image,
        black_threshold,
    )

    # The fixed 40 px reference represents one real maze square = 180 mm.
    # It establishes pixels/mm only; route distances are not rounded to 180 mm.
    # This is applied before obstacle inflation because robot clearance is
    # calculated from the physical robot dimensions.

    robot_clearance_px = calculate_robot_clearance_px(
        unit_space_px
    )

    inflated_obstacles = inflate_obstacles(
        obstacle_mask,
        robot_clearance_px,
    )

    obstacle_file = output_dir / "generated_obstacle_mask.png"
    inflated_file = output_dir / "generated_inflated_obstacles.png"

    cv2.imwrite(str(obstacle_file), obstacle_mask)
    cv2.imwrite(str(inflated_file), inflated_obstacles)

    print()
    print("========================================")
    print("FIXED IMAGE PARAMETERS")
    print("========================================")
    print(f"Fixed one maze unit:    {unit_space_px:.1f} px = {CELL_SIZE_MM:.0f} mm")
    print(f"Robot width:             {ROBOT_WIDTH_MM:.1f} mm")
    print(f"Robot physical half-width: {ROBOT_HALF_WIDTH_MM:.1f} mm")
    print(f"Planner clearance:         {PLANNER_CLEARANCE_MM:.1f} mm")
    print(f"Extra planner margin:      {EXTRA_CLEARANCE_MM:.1f} mm")
    print(f"Obstacle inflation:        {robot_clearance_px:.1f} px")
    print(
        "Short-turn threshold:   "
        f"{unit_space_px * CORNER_CUT_UNIT_MULTIPLIER * SHORT_MOVE_FRACTION:.1f} px"
    )
    print("========================================")

    start_point, goal_point, start_heading_vector = select_start_goal_points(
        maze_image,
        inflated_obstacles,
        unit_space_px,
    )

    occupied, clearance = build_planning_grid(inflated_obstacles)

    start_grid = image_point_to_grid(start_point)
    goal_grid = image_point_to_grid(goal_point)

    path_grid = astar_path(
        occupied,
        clearance,
        start_grid,
        goal_grid,
    )

    if not path_grid:
        raise SystemExit(
            "No path found. Check octagon fit, start/goal placement, "
            "or the fixed obstacle/clearance settings."
        )

    simplified_grid = simplify_grid_path(path_grid)

    path_img = [grid_point_to_image(p) for p in path_grid]
    simplified_img = [grid_point_to_image(p) for p in simplified_grid]

    # Second-stage simplification: remove short doglegs/corner wiggles using
    # the user-measured maze unit and a collision-checked direct shortcut.
    smoothed_img = remove_short_doglegs(
        simplified_img,
        inflated_obstacles,
        unit_space_px,
    )

    robot_route = build_hybrid_robot_route(
        smoothed_img,
        obstacle_mask,
        unit_space_px,
        initial_heading_vector=start_heading_vector,
    )

    serial_route = route_to_serial_string(robot_route)

    result = draw_path_overlay(
        maze_image,
        path_img,
        smoothed_img,
        start_point,
        goal_point,
        start_heading_vector=start_heading_vector,
        unit_space_px=unit_space_px,
    )

    result = draw_route_labels(result, robot_route)

    result_file = output_dir / "generated_maze_path.png"
    cv2.imwrite(str(result_file), result)

    waypoints_file = output_dir / "generated_waypoints.txt"
    with open(waypoints_file, "w", encoding="utf-8") as f:
        f.write("Start point (pixels):\n")
        f.write(f"{start_point}\n\n")
        f.write("Goal point (pixels):\n")
        f.write(f"{goal_point}\n\n")
        f.write("Selected robot start heading vector [dx, dy] (image coordinates):\n")
        f.write(f"{start_heading_vector[0]:.6f},{start_heading_vector[1]:.6f}\n\n")
        f.write("Fixed one maze unit (pixels):\n")
        f.write(f"{unit_space_px:.3f}\n\n")

        f.write("Final smoothed path waypoints (pixels):\n")
        for point in smoothed_img:
            f.write(f"{point[0]},{point[1]}\n")

    route_file = output_dir / "generated_robot_route.txt"
    with open(route_file, "w", encoding="utf-8") as f:
        f.write("MTRN3100 hybrid robot route\n")
        f.write(f"Scale reference: {unit_space_px:.3f} px = 180 mm\n")
        start_heading_deg = float(
            np.degrees(np.arctan2(-start_heading_vector[1], start_heading_vector[0]))
        )
        f.write(f"Selected start facing angle: {start_heading_deg:+.1f} deg\n")
        f.write("First segment turn is from this selected heading to the first path line.\n\n")
        f.write("Format: index, turn_deg, mode, distance_mm, raw_units\n")

        for index, item in enumerate(robot_route, start=1):
            f.write(
                f'{index}, {item["turn_deg"]:.1f}, {item["mode"]}, '
                f'{item["distance_mm"]:.1f}, {item["units_float"]:.3f}\n'
            )

        f.write("\nSerial command:\n")
        f.write(serial_route + "\n")

    print()
    print("========================================")
    print("Micromouse path planning complete")
    print(f"Black threshold used: {black_threshold}")
    print("========================================")
    print("Saved:", corrected_preview)
    print("Saved:", obstacle_file)
    print("Saved:", inflated_file)
    print("Saved:", result_file)
    print("Saved:", waypoints_file)
    print("Saved:", route_file)
    print("Fixed one maze unit (px):    ", round(unit_space_px, 1))
    print("Number of fine path nodes: ", len(path_img))
    print("Waypoints before corner cutting: ", len(simplified_img))
    print("Waypoints after corner cutting:  ", len(smoothed_img))
    print()
    start_heading_deg = float(
        np.degrees(np.arctan2(-start_heading_vector[1], start_heading_vector[0]))
    )
    print(f"Selected robot start facing: {start_heading_deg:+.1f} deg")
    if robot_route:
        print(
            "Initial alignment turn:       "
            f"{robot_route[0]['turn_deg']:+.1f} deg"
        )
    print()
    print("Robot route segments:")
    for index, item in enumerate(robot_route, start=1):
        wall_info = (
            f'wall@{item["wall_distance_units"]:.2f} units, '
            f'maxCoverage={item["wall_max_coverage"]:.2f}'
            if item["wall_distance_units"] is not None
            else
            f'no confirmed wall, maxCoverage={item["wall_max_coverage"]:.2f}'
        )

        print(
            f'  {index}: turn={item["turn_deg"]:+.1f} deg | '
            f'{item["mode"]} | {item["distance_mm"]:.1f} mm '
            f'(raw {item["units_float"]:.3f} units) | {wall_info}'
        )
    print()
    print("Serial-ready route:")
    print(serial_route)
    print("========================================")

    print()
    print("Connect this laptop to the ESP32 Wi-Fi network 'MicroMouse'.")
    print("Then press ENTER in the final path window to upload this route.")

    wait_for_upload_confirmation(
        result,
        serial_route,
    )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
