import argparse
import glob
import json
import os
import re
from collections import defaultdict

import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize as sk_skeletonize
except ImportError:
    sk_skeletonize = None


DEFAULT_MATCH_THRESHOLD = 0.25
DEFAULT_NMS_DISTANCE = 5
DEFAULT_LABELS = ("T", "Y", "X")
TYPE_ALIAS = {"V": "T"}
TEMPLATE_NAME_RE = re.compile(r"^([A-Za-z]+)_s(\d+)_a(\d+)_r(\d+)$")
DETECTION_MODES = ("template", "graph", "hybrid")
DEFAULT_DETECTION_MODE = "hybrid"
DEFAULT_GRAPH_ARM_INNER_RADIUS = 6
DEFAULT_GRAPH_ARM_OUTER_RADIUS = 13
DEFAULT_GRAPH_MIN_ARM_PIXELS = 2
DEFAULT_GRAPH_MIN_ARM_SPAN = 3.0
DEFAULT_GRAPH_MIN_BRANCH_COMPONENT_AREA = 1
DEFAULT_GRAPH_MAX_CENTERS_PER_COMPONENT = 1
DEFAULT_GRAPH_RING_GAP_BRIDGE_RADIUS = 0
DEFAULT_GRAPH_SPARSE_RECOVERY = False
DEFAULT_GRAPH_SPARSE_RECOVERY_SUPPORT_RADIUS = 2
DEFAULT_GRAPH_SPARSE_RECOVERY_ENDPOINT_RADIUS = 10
DEFAULT_GRAPH_SPARSE_RECOVERY_MAX_COMPONENT_AREA = 180
DEFAULT_GRAPH_SPARSE_RECOVERY_MIN_SCORE = 0.12
DEFAULT_GRAPH_JUNCTION_SNAP_REPAIR = False
DEFAULT_GRAPH_JUNCTION_SNAP_ITERS = 1
DEFAULT_GRAPH_JUNCTION_SNAP_DIST = 3
DEFAULT_GRAPH_JUNCTION_SNAP_MIN_DOT = 0.1
DEFAULT_GRAPH_JUNCTION_SNAP_MAX_EXISTING_FRACTION = 0.85
DEFAULT_GRAPH_T_MIN_LARGEST_ANGLE = 150.0
DEFAULT_GRAPH_T_MAX_SIDE_ANGLE = 124.0
DEFAULT_LOCAL_RECLASSIFY_T = False
DEFAULT_LOCAL_RECLASSIFY_Y_TO_T = True
DEFAULT_TY_MULTIRADIUS_VOTE = True
DEFAULT_TY_VOTE_OUTER_RADII = (29, 45, 61)
DEFAULT_TY_VOTE_MIN_VOTES = 2
DEFAULT_TY_VOTE_MARGIN = 1
DEFAULT_TY_VOTE_REQUIRE_NON_AMBIG = True
DEFAULT_TY_FEATURE_RULES = False
DEFAULT_TY_FEATURE_RULE_MODE = "full"
TY_FEATURE_RULE_MODES = ("lite", "full")
DEFAULT_TY_FEATURE_RULE_OUTER_RADII = (29, 45, 61)
DEFAULT_TY_FEATURE_RULE_INNER_RADIUS = 10
DEFAULT_TY_FEATURE_RULE_MIN_ARM_PIXELS = 2
DEFAULT_TY_FEATURE_RULE_MIN_ARM_SPAN = 1.0
DEFAULT_TY_FEATURE_RULE_T_MIN_LARGEST_ANGLE = 140.0
DEFAULT_TY_FEATURE_RULE_T_MAX_SIDE_ANGLE = 148.0
DEFAULT_TY_FEATURE_RULE_AMBIGUITY_MARGIN = 20.0
DEFAULT_TY_FEATURE_RULE_CLASSIFIER = "symmetric"
TY_FEATURE_RULE_VERSION = "depth7_v1"
DEFAULT_TYX_STRUCTURAL_RULES = False
DEFAULT_TYX_STRUCTURAL_RULE_MODE = "aggressive"
TYX_STRUCTURAL_RULE_MODES = ("lite", "aggressive")
TYX_STRUCTURAL_RULE_VERSION = "det_tree_v1"
DEFAULT_TYX_STRUCTURAL_OOD_GATE = True
DEFAULT_TYX_STRUCTURAL_MIN_MARGIN = 0.35
TYX_STRUCTURAL_MIN_MARGIN_BY_MODE = {
    "lite": 0.20,
    "aggressive": 0.35,
}
DEFAULT_GENERALIZATION_AUTO_SCALE = False
DEFAULT_GENERALIZATION_BRANCH_RATIO_THRESHOLD = 0.03
DEFAULT_GENERALIZATION_LOW_OUTER_RADIUS = 21
DEFAULT_GENERALIZATION_LOW_NMS_DISTANCE = 12
DEFAULT_GENERALIZATION_LOW_MAX_CENTERS = 2
DEFAULT_GENERALIZATION_LOW_RING_GAP_BRIDGE_RADIUS = 3
DEFAULT_GENERALIZATION_LOW_SPARSE_RECOVERY = True
DEFAULT_GENERALIZATION_LOW_SPARSE_ENDPOINT_RADIUS = 1
DEFAULT_GENERALIZATION_LOW_SPARSE_MIN_SCORE = 0.12
DEFAULT_GENERALIZATION_LOW_FINAL_MERGE_DISTANCE = 14
DEFAULT_GENERALIZATION_LOW_CLASS_DEBIAS = True
DEFAULT_GENERALIZATION_ULTRA_LOW_BRANCH_RATIO = 0.02
DEFAULT_GENERALIZATION_MODERATE_BRANCH_RATIO = 0.008
DEFAULT_GENERALIZATION_MODERATE_FINAL_MERGE_DISTANCE = 12
DEFAULT_GENERALIZATION_ULTRA_LOW_MAX_CENTERS = 3
DEFAULT_GENERALIZATION_ULTRA_LOW_FINAL_MERGE_DISTANCE = 10
DEFAULT_GENERALIZATION_ULTRA_LOW_CORNER_RECOVERY = True
DEFAULT_GENERALIZATION_ULTRA_LOW_CORNER_MIN_SCORE = 0.05
DEFAULT_GENERALIZATION_ULTRA_LOW_CORNER_ENDPOINT_RADIUS = 3
DEFAULT_GENERALIZATION_MODERATE_LOW_SCORE_T_TO_Y = 0.02
DEFAULT_GENERALIZATION_MODERATE_WEAK_X_CORNER_SUPPRESS_AREA = 3
DEFAULT_GENERALIZATION_ULTRA_Y_TO_T_MAX_SCORE = 0.08
DEFAULT_GENERALIZATION_ULTRA_Y_TO_T_MAX_ERR_T = 100.0
DEFAULT_GENERALIZATION_ULTRA_Y_TO_T_MAX_ERR_Y = 40.0
DEFAULT_GENERALIZATION_ULTRA_T_VOTE_SNAP = True
DEFAULT_GENERALIZATION_ULTRA_T_VOTE_RADIUS = 8
DEFAULT_GENERALIZATION_ULTRA_T_VOTE_MIN_VOTES = 3
DEFAULT_GENERALIZATION_ULTRA_T_VOTE_MIN_SUPPORT_THREE = 3
DEFAULT_GENERALIZATION_ULTRA_T_VOTE_MAX_SUPPORT_FOUR = 1
DEFAULT_X_CONSISTENCY_RECLASSIFY = False
DEFAULT_GENERALIZATION_LOW_X_CONSISTENCY_RECLASSIFY = True
DEFAULT_X_CONSISTENCY_OUTER_RADII = (15, 19, 23)
DEFAULT_X_CONSISTENCY_MAX_FOURARM = 1
DEFAULT_X_CONSISTENCY_MIN_THREEARM_VOTES = 2
DEFAULT_SPARSE_CORNER_MAX_ANGLE = 150.0
LOW_BRANCH_CLASS_DEBIAS_VERSION = "v4"
_TYX_TREE_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tyx_structural_trees.json")
if os.path.exists(_TYX_TREE_JSON):
    with open(_TYX_TREE_JSON, "r", encoding="utf-8") as _f:
        TYX_STRUCTURAL_TREE_BUNDLES = json.load(_f)
else:
    TYX_STRUCTURAL_TREE_BUNDLES = json.loads(r'''{
  "lite": {
    "feature_names": [
      "pred_code",
      "arm_count",
      "branch_component_area",
      "local_err_t",
      "local_err_y",
      "local_ambiguous",
      "score",
      "ty_rule_err_t_29",
      "ty_rule_err_y_29",
      "ty_rule_t_guardrail_29",
      "ty_rule_a0_29",
      "ty_rule_a1_29",
      "ty_rule_a2_29",
      "ty_rule_opp_angle_29",
      "ty_rule_opp_gap_29",
      "ty_rule_area_cv_29",
      "ty_rule_span_cv_29",
      "ty_rule_area_ratio_29",
      "ty_rule_span_ratio_29",
      "ty_rule_third_area_frac_29",
      "ty_rule_third_span_frac_29",
      "ty_rule_opp_balance_29",
      "ty_rule_opp_area_ratio_29",
      "ty_rule_opp_span_ratio_29",
      "ty_rule_arm_count_29",
      "ty_rule_err_t_45",
      "ty_rule_err_y_45",
      "ty_rule_t_guardrail_45",
      "ty_rule_a0_45",
      "ty_rule_a1_45",
      "ty_rule_a2_45",
      "ty_rule_opp_angle_45",
      "ty_rule_opp_gap_45",
      "ty_rule_area_cv_45",
      "ty_rule_span_cv_45",
      "ty_rule_area_ratio_45",
      "ty_rule_span_ratio_45",
      "ty_rule_third_area_frac_45",
      "ty_rule_third_span_frac_45",
      "ty_rule_opp_balance_45",
      "ty_rule_opp_area_ratio_45",
      "ty_rule_opp_span_ratio_45",
      "ty_rule_arm_count_45",
      "ty_rule_err_t_61",
      "ty_rule_err_y_61",
      "ty_rule_t_guardrail_61",
      "ty_rule_a0_61",
      "ty_rule_a1_61",
      "ty_rule_a2_61",
      "ty_rule_opp_angle_61",
      "ty_rule_opp_gap_61",
      "ty_rule_area_cv_61",
      "ty_rule_span_cv_61",
      "ty_rule_area_ratio_61",
      "ty_rule_span_ratio_61",
      "ty_rule_third_area_frac_61",
      "ty_rule_third_span_frac_61",
      "ty_rule_opp_balance_61",
      "ty_rule_opp_area_ratio_61",
      "ty_rule_opp_span_ratio_61",
      "ty_rule_arm_count_61"
    ],
    "children_left": [
      1,
      2,
      3,
      4,
      -1,
      6,
      -1,
      -1,
      9,
      10,
      -1,
      -1,
      -1,
      14,
      -1,
      16,
      -1,
      18,
      19,
      -1,
      21,
      22,
      -1,
      24,
      25,
      26,
      -1,
      -1,
      -1,
      -1,
      -1,
      32,
      33,
      -1,
      35,
      -1,
      -1,
      -1,
      39,
      40,
      41,
      42,
      -1,
      -1,
      45,
      -1,
      47,
      -1,
      -1,
      50,
      -1,
      -1,
      53,
      54,
      -1,
      -1,
      57,
      58,
      -1,
      -1,
      -1
    ],
    "children_right": [
      38,
      13,
      8,
      5,
      -1,
      7,
      -1,
      -1,
      12,
      11,
      -1,
      -1,
      -1,
      15,
      -1,
      17,
      -1,
      31,
      20,
      -1,
      30,
      23,
      -1,
      29,
      28,
      27,
      -1,
      -1,
      -1,
      -1,
      -1,
      37,
      34,
      -1,
      36,
      -1,
      -1,
      -1,
      52,
      49,
      44,
      43,
      -1,
      -1,
      46,
      -1,
      48,
      -1,
      -1,
      51,
      -1,
      -1,
      56,
      55,
      -1,
      -1,
      60,
      59,
      -1,
      -1,
      -1
    ],
    "feature": [
      0,
      25,
      29,
      10,
      -2,
      2,
      -2,
      -2,
      26,
      33,
      -2,
      -2,
      -2,
      15,
      -2,
      11,
      -2,
      15,
      39,
      -2,
      33,
      2,
      -2,
      52,
      27,
      40,
      -2,
      -2,
      -2,
      -2,
      -2,
      53,
      4,
      -2,
      15,
      -2,
      -2,
      -2,
      0,
      53,
      18,
      2,
      -2,
      -2,
      41,
      -2,
      3,
      -2,
      -2,
      4,
      -2,
      -2,
      2,
      2,
      -2,
      -2,
      2,
      2,
      -2,
      -2,
      -2
    ],
    "threshold": [
      0.5,
      62.0200472559285,
      118.75453023136524,
      104.17192936661648,
      -2.0,
      3.5,
      -2.0,
      -2.0,
      110.75918444291673,
      0.42158554885100125,
      -2.0,
      -2.0,
      -2.0,
      0.10815999399586049,
      -2.0,
      109.35601262950517,
      -2.0,
      0.3693966223993812,
      0.13373071528751754,
      -2.0,
      0.38118731327579064,
      2.0,
      -2.0,
      0.011153880239413312,
      0.5,
      3.212121212121212,
      -2.0,
      -2.0,
      -2.0,
      -2.0,
      -2.0,
      2.9047619047619047,
      26.870229531632766,
      -2.0,
      0.37925453762737515,
      -2.0,
      -2.0,
      -2.0,
      1.5,
      3.3846153846153846,
      1.0130758633329118,
      3.5,
      -2.0,
      -2.0,
      2.0685608741069044,
      -2.0,
      79.35511855908882,
      -2.0,
      -2.0,
      35.62762525592921,
      -2.0,
      -2.0,
      3.5,
      2.0,
      -2.0,
      -2.0,
      5.5,
      4.5,
      -2.0,
      -2.0,
      -2.0
    ],
    "value": [
      [
        234.0,
        144.0,
        35.0
      ],
      [
        229.0,
        20.0,
        6.0
      ],
      [
        169.0,
        5.0,
        0.0
      ],
      [
        150.0,
        1.0,
        0.0
      ],
      [
        143.0,
        0.0,
        0.0
      ],
      [
        7.0,
        1.0,
        0.0
      ],
      [
        6.0,
        0.0,
        0.0
      ],
      [
        1.0,
        1.0,
        0.0
      ],
      [
        19.0,
        4.0,
        0.0
      ],
      [
        19.0,
        2.0,
        0.0
      ],
      [
        18.0,
        0.0,
        0.0
      ],
      [
        1.0,
        2.0,
        0.0
      ],
      [
        0.0,
        2.0,
        0.0
      ],
      [
        60.0,
        15.0,
        6.0
      ],
      [
        24.0,
        0.0,
        0.0
      ],
      [
        36.0,
        15.0,
        6.0
      ],
      [
        0.0,
        3.0,
        0.0
      ],
      [
        36.0,
        12.0,
        6.0
      ],
      [
        28.0,
        11.0,
        1.0
      ],
      [
        10.0,
        0.0,
        0.0
      ],
      [
        18.0,
        11.0,
        1.0
      ],
      [
        11.0,
        11.0,
        1.0
      ],
      [
        4.0,
        0.0,
        0.0
      ],
      [
        7.0,
        11.0,
        1.0
      ],
      [
        4.0,
        11.0,
        1.0
      ],
      [
        2.0,
        11.0,
        0.0
      ],
      [
        0.0,
        10.0,
        0.0
      ],
      [
        2.0,
        1.0,
        0.0
      ],
      [
        2.0,
        0.0,
        1.0
      ],
      [
        3.0,
        0.0,
        0.0
      ],
      [
        7.0,
        0.0,
        0.0
      ],
      [
        8.0,
        1.0,
        5.0
      ],
      [
        8.0,
        1.0,
        2.0
      ],
      [
        0.0,
        0.0,
        2.0
      ],
      [
        8.0,
        1.0,
        0.0
      ],
      [
        1.0,
        1.0,
        0.0
      ],
      [
        7.0,
        0.0,
        0.0
      ],
      [
        0.0,
        0.0,
        3.0
      ],
      [
        5.0,
        124.0,
        29.0
      ],
      [
        3.0,
        122.0,
        2.0
      ],
      [
        3.0,
        120.0,
        0.0
      ],
      [
        2.0,
        6.0,
        0.0
      ],
      [
        2.0,
        2.0,
        0.0
      ],
      [
        0.0,
        4.0,
        0.0
      ],
      [
        1.0,
        114.0,
        0.0
      ],
      [
        0.0,
        109.0,
        0.0
      ],
      [
        1.0,
        5.0,
        0.0
      ],
      [
        1.0,
        1.0,
        0.0
      ],
      [
        0.0,
        4.0,
        0.0
      ],
      [
        0.0,
        2.0,
        2.0
      ],
      [
        0.0,
        2.0,
        0.0
      ],
      [
        0.0,
        0.0,
        2.0
      ],
      [
        2.0,
        2.0,
        27.0
      ],
      [
        1.0,
        0.0,
        16.0
      ],
      [
        1.0,
        0.0,
        5.0
      ],
      [
        0.0,
        0.0,
        11.0
      ],
      [
        1.0,
        2.0,
        11.0
      ],
      [
        1.0,
        2.0,
        8.0
      ],
      [
        1.0,
        1.0,
        6.0
      ],
      [
        0.0,
        1.0,
        2.0
      ],
      [
        0.0,
        0.0,
        3.0
      ]
    ],
    "classes": [
      "T",
      "Y",
      "X"
    ]
  },
  "aggressive": {
    "feature_names": [
      "pred_code",
      "arm_count",
      "branch_component_area",
      "local_err_t",
      "local_err_y",
      "local_ambiguous",
      "score",
      "ty_rule_err_t_29",
      "ty_rule_err_y_29",
      "ty_rule_t_guardrail_29",
      "ty_rule_a0_29",
      "ty_rule_a1_29",
      "ty_rule_a2_29",
      "ty_rule_opp_angle_29",
      "ty_rule_opp_gap_29",
      "ty_rule_area_cv_29",
      "ty_rule_span_cv_29",
      "ty_rule_area_ratio_29",
      "ty_rule_span_ratio_29",
      "ty_rule_third_area_frac_29",
      "ty_rule_third_span_frac_29",
      "ty_rule_opp_balance_29",
      "ty_rule_opp_area_ratio_29",
      "ty_rule_opp_span_ratio_29",
      "ty_rule_arm_count_29",
      "ty_rule_err_t_45",
      "ty_rule_err_y_45",
      "ty_rule_t_guardrail_45",
      "ty_rule_a0_45",
      "ty_rule_a1_45",
      "ty_rule_a2_45",
      "ty_rule_opp_angle_45",
      "ty_rule_opp_gap_45",
      "ty_rule_area_cv_45",
      "ty_rule_span_cv_45",
      "ty_rule_area_ratio_45",
      "ty_rule_span_ratio_45",
      "ty_rule_third_area_frac_45",
      "ty_rule_third_span_frac_45",
      "ty_rule_opp_balance_45",
      "ty_rule_opp_area_ratio_45",
      "ty_rule_opp_span_ratio_45",
      "ty_rule_arm_count_45",
      "ty_rule_err_t_61",
      "ty_rule_err_y_61",
      "ty_rule_t_guardrail_61",
      "ty_rule_a0_61",
      "ty_rule_a1_61",
      "ty_rule_a2_61",
      "ty_rule_opp_angle_61",
      "ty_rule_opp_gap_61",
      "ty_rule_area_cv_61",
      "ty_rule_span_cv_61",
      "ty_rule_area_ratio_61",
      "ty_rule_span_ratio_61",
      "ty_rule_third_area_frac_61",
      "ty_rule_third_span_frac_61",
      "ty_rule_opp_balance_61",
      "ty_rule_opp_area_ratio_61",
      "ty_rule_opp_span_ratio_61",
      "ty_rule_arm_count_61"
    ],
    "children_left": [
      1,
      2,
      3,
      4,
      -1,
      6,
      -1,
      -1,
      9,
      10,
      -1,
      12,
      -1,
      -1,
      -1,
      16,
      -1,
      18,
      -1,
      20,
      21,
      -1,
      23,
      24,
      -1,
      26,
      27,
      28,
      -1,
      30,
      -1,
      -1,
      33,
      -1,
      -1,
      -1,
      -1,
      38,
      39,
      -1,
      41,
      -1,
      -1,
      -1,
      45,
      46,
      47,
      48,
      49,
      50,
      -1,
      52,
      -1,
      -1,
      -1,
      -1,
      57,
      -1,
      59,
      -1,
      -1,
      62,
      -1,
      -1,
      65,
      66,
      -1,
      -1,
      69,
      70,
      -1,
      -1,
      -1
    ],
    "children_right": [
      44,
      15,
      8,
      5,
      -1,
      7,
      -1,
      -1,
      14,
      11,
      -1,
      13,
      -1,
      -1,
      -1,
      17,
      -1,
      19,
      -1,
      37,
      22,
      -1,
      36,
      25,
      -1,
      35,
      32,
      29,
      -1,
      31,
      -1,
      -1,
      34,
      -1,
      -1,
      -1,
      -1,
      43,
      40,
      -1,
      42,
      -1,
      -1,
      -1,
      64,
      61,
      56,
      55,
      54,
      51,
      -1,
      53,
      -1,
      -1,
      -1,
      -1,
      58,
      -1,
      60,
      -1,
      -1,
      63,
      -1,
      -1,
      68,
      67,
      -1,
      -1,
      72,
      71,
      -1,
      -1,
      -1
    ],
    "feature": [
      0,
      25,
      29,
      10,
      -2,
      20,
      -2,
      -2,
      26,
      33,
      -2,
      2,
      -2,
      -2,
      -2,
      15,
      -2,
      11,
      -2,
      15,
      39,
      -2,
      33,
      2,
      -2,
      52,
      27,
      40,
      -2,
      11,
      -2,
      -2,
      3,
      -2,
      -2,
      -2,
      -2,
      53,
      4,
      -2,
      2,
      -2,
      -2,
      -2,
      0,
      53,
      18,
      10,
      2,
      2,
      -2,
      6,
      -2,
      -2,
      -2,
      -2,
      41,
      -2,
      3,
      -2,
      -2,
      4,
      -2,
      -2,
      2,
      2,
      -2,
      -2,
      2,
      2,
      -2,
      -2,
      -2
    ],
    "threshold": [
      0.5,
      62.0200472559285,
      118.75453023136524,
      104.17192936661648,
      -2.0,
      0.34295407130342437,
      -2.0,
      -2.0,
      110.75918444291673,
      0.42158554885100125,
      -2.0,
      3.5,
      -2.0,
      -2.0,
      -2.0,
      0.10815999399586049,
      -2.0,
      109.35601262950517,
      -2.0,
      0.3693966223993812,
      0.13373071528751754,
      -2.0,
      0.38118731327579064,
      2.0,
      -2.0,
      0.011153880239413312,
      0.5,
      3.212121212121212,
      -2.0,
      117.35009436871997,
      -2.0,
      -2.0,
      74.78173882601155,
      -2.0,
      -2.0,
      -2.0,
      -2.0,
      2.9047619047619047,
      26.870229531632766,
      -2.0,
      2.0,
      -2.0,
      -2.0,
      -2.0,
      1.5,
      3.3846153846153846,
      1.0130758633329118,
      115.13331530008227,
      3.5,
      2.0,
      -2.0,
      0.031947789637240494,
      -2.0,
      -2.0,
      -2.0,
      -2.0,
      2.0685608741069044,
      -2.0,
      69.8333604984628,
      -2.0,
      -2.0,
      35.62762525592921,
      -2.0,
      -2.0,
      3.5,
      2.0,
      -2.0,
      -2.0,
      5.5,
      4.5,
      -2.0,
      -2.0,
      -2.0
    ],
    "value": [
      [
        234.0,
        144.0,
        35.0
      ],
      [
        229.0,
        20.0,
        6.0
      ],
      [
        169.0,
        5.0,
        0.0
      ],
      [
        150.0,
        1.0,
        0.0
      ],
      [
        143.0,
        0.0,
        0.0
      ],
      [
        7.0,
        1.0,
        0.0
      ],
      [
        7.0,
        0.0,
        0.0
      ],
      [
        0.0,
        1.0,
        0.0
      ],
      [
        19.0,
        4.0,
        0.0
      ],
      [
        19.0,
        2.0,
        0.0
      ],
      [
        18.0,
        0.0,
        0.0
      ],
      [
        1.0,
        2.0,
        0.0
      ],
      [
        0.0,
        2.0,
        0.0
      ],
      [
        1.0,
        0.0,
        0.0
      ],
      [
        0.0,
        2.0,
        0.0
      ],
      [
        60.0,
        15.0,
        6.0
      ],
      [
        24.0,
        0.0,
        0.0
      ],
      [
        36.0,
        15.0,
        6.0
      ],
      [
        0.0,
        3.0,
        0.0
      ],
      [
        36.0,
        12.0,
        6.0
      ],
      [
        28.0,
        11.0,
        1.0
      ],
      [
        10.0,
        0.0,
        0.0
      ],
      [
        18.0,
        11.0,
        1.0
      ],
      [
        11.0,
        11.0,
        1.0
      ],
      [
        4.0,
        0.0,
        0.0
      ],
      [
        7.0,
        11.0,
        1.0
      ],
      [
        4.0,
        11.0,
        1.0
      ],
      [
        2.0,
        11.0,
        0.0
      ],
      [
        0.0,
        10.0,
        0.0
      ],
      [
        2.0,
        1.0,
        0.0
      ],
      [
        2.0,
        0.0,
        0.0
      ],
      [
        0.0,
        1.0,
        0.0
      ],
      [
        2.0,
        0.0,
        1.0
      ],
      [
        0.0,
        0.0,
        1.0
      ],
      [
        2.0,
        0.0,
        0.0
      ],
      [
        3.0,
        0.0,
        0.0
      ],
      [
        7.0,
        0.0,
        0.0
      ],
      [
        8.0,
        1.0,
        5.0
      ],
      [
        8.0,
        1.0,
        2.0
      ],
      [
        0.0,
        0.0,
        2.0
      ],
      [
        8.0,
        1.0,
        0.0
      ],
      [
        0.0,
        1.0,
        0.0
      ],
      [
        8.0,
        0.0,
        0.0
      ],
      [
        0.0,
        0.0,
        3.0
      ],
      [
        5.0,
        124.0,
        29.0
      ],
      [
        3.0,
        122.0,
        2.0
      ],
      [
        3.0,
        120.0,
        0.0
      ],
      [
        2.0,
        6.0,
        0.0
      ],
      [
        1.0,
        6.0,
        0.0
      ],
      [
        1.0,
        2.0,
        0.0
      ],
      [
        0.0,
        1.0,
        0.0
      ],
      [
        1.0,
        1.0,
        0.0
      ],
      [
        0.0,
        1.0,
        0.0
      ],
      [
        1.0,
        0.0,
        0.0
      ],
      [
        0.0,
        4.0,
        0.0
      ],
      [
        1.0,
        0.0,
        0.0
      ],
      [
        1.0,
        114.0,
        0.0
      ],
      [
        0.0,
        109.0,
        0.0
      ],
      [
        1.0,
        5.0,
        0.0
      ],
      [
        1.0,
        0.0,
        0.0
      ],
      [
        0.0,
        5.0,
        0.0
      ],
      [
        0.0,
        2.0,
        2.0
      ],
      [
        0.0,
        2.0,
        0.0
      ],
      [
        0.0,
        0.0,
        2.0
      ],
      [
        2.0,
        2.0,
        27.0
      ],
      [
        1.0,
        0.0,
        16.0
      ],
      [
        1.0,
        0.0,
        5.0
      ],
      [
        0.0,
        0.0,
        11.0
      ],
      [
        1.0,
        2.0,
        11.0
      ],
      [
        1.0,
        2.0,
        8.0
      ],
      [
        1.0,
        1.0,
        6.0
      ],
      [
        0.0,
        1.0,
        2.0
      ],
      [
        0.0,
        0.0,
        3.0
      ]
    ],
    "classes": [
      "T",
      "Y",
      "X"
    ]
  }
}''')

DEFAULT_GRAPH_CENTER_MODE = "best"
GRAPH_CENTER_MODES = ("centroid", "snap", "best")
DEFAULT_THREE_ARM_CLASSIFIER = "symmetric"
THREE_ARM_CLASSIFIER_MODES = ("legacy", "symmetric", "robust", "robust_vote")
DEFAULT_ROBUST_JITTER_RADIUS = 2
DEFAULT_ROBUST_JITTER_DIST_PENALTY = 1.25
DEFAULT_ROBUST_VOTE_TEMPERATURE = 18.0
DEFAULT_ROBUST_VOTE_GUARDRAIL_SCALE = 0.72
DEFAULT_ROBUST_VOTE_DISTANCE_WEIGHT = 0.35
DEFAULT_ROBUST_VOTE_OUTER_OFFSETS = (0, 4)
DEFAULT_HYBRID_FUSION_MODE = "conflict_aware"
HYBRID_FUSION_MODES = ("legacy", "conflict_aware")


def parse_labels(raw: str):
    labels = [token.strip().upper() for token in raw.split(",") if token.strip()]
    if not labels:
        raise ValueError("No labels provided. Example: --labels T,Y,X")
    return labels


def parse_int_tuple(raw: str):
    vals = []
    for token in str(raw).split(","):
        v = token.strip()
        if not v:
            continue
        vals.append(int(v))
    if not vals:
        raise ValueError("Expected a comma-separated integer list.")
    return tuple(vals)


def load_templates(template_dir, allowed_labels=None, allow_legacy_v=True):
    """
    Load all templates recursively from subdirectories.
    """
    if allowed_labels is None:
        allowed_labels = list(DEFAULT_LABELS)
    allowed_set = set(allowed_labels)
    templates = defaultdict(list)

    pattern = os.path.join(template_dir, "**", "*.png")
    files = glob.glob(pattern, recursive=True)
    print(f"Scanning templates in: {template_dir}")

    for filepath in files:
        basename = os.path.basename(filepath)
        stem = os.path.splitext(basename)[0]
        match = TEMPLATE_NAME_RE.match(stem)
        if match is None:
            continue

        raw_type = match.group(1).upper()
        mapped_type = TYPE_ALIAS.get(raw_type, raw_type) if allow_legacy_v else raw_type
        if mapped_type not in allowed_set:
            continue

        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        size = int(match.group(2))
        spread = int(match.group(3))
        rotation = int(match.group(4))
        parent = os.path.basename(os.path.dirname(filepath))
        gray_level = 0
        if parent.startswith("gray_"):
            try:
                gray_level = int(parent.split("_", 1)[1])
            except ValueError:
                gray_level = 0

        templates[mapped_type].append(
            {
                "image": img,
                "size": size,
                "spread": spread,
                "rotation": rotation,
                "gray": gray_level,
                "path": filepath,
                "type": mapped_type,
                "source_type": raw_type,
            }
        )

    total = sum(len(v) for v in templates.values())
    print(f"Loaded {total} templates across labels: {', '.join(sorted(templates.keys()))}")
    return templates


def template_match_multiscale(image, templates, threshold=DEFAULT_MATCH_THRESHOLD):
    detections = []
    total_templates = sum(len(v) for v in templates.values())
    if total_templates == 0:
        return detections

    processed = 0
    for jtype, template_list in templates.items():
        for template_info in template_list:
            processed += 1
            if processed % 100 == 0:
                print(f"Matching templates: {processed}/{total_templates}", end="\r")

            template = template_info["image"]
            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(result >= threshold)

            for y, x in zip(ys, xs):
                cx = int(x + template.shape[1] // 2)
                cy = int(y + template.shape[0] // 2)
                detections.append(
                    {
                        "type": jtype,
                        "x": cx,
                        "y": cy,
                        "score": float(result[y, x]),
                        "spread": int(template_info["spread"]),
                        "rotation": int(template_info["rotation"]),
                        "size": int(template_info["size"]),
                        "bg_gray": int(template_info["gray"]),
                        "source_type": template_info.get("source_type", jtype),
                    }
                )

    print(f"\nFound {len(detections)} raw matches")
    return detections


def auto_line_mask(image, line_threshold=127):
    """
    Auto-polarity sparse-line extraction:
    choose dark or bright foreground polarity based on which one is sparser.
    """
    threshold = int(np.clip(line_threshold, 1, 254))
    dark = (image < threshold).astype(np.uint8)
    bright = (image > (255 - threshold)).astype(np.uint8)

    dark_frac = float(dark.mean())
    bright_frac = float(bright.mean())
    return dark if dark_frac <= bright_frac else bright


def morphological_skeleton(binary_mask):
    work = (binary_mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(work)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(work, kernel)
        opened = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(work, opened)
        skel = cv2.bitwise_or(skel, temp)
        work = eroded
        if cv2.countNonZero(work) == 0:
            break
    return (skel > 0).astype(np.uint8)


def thin_binary(binary_mask):
    binary = (binary_mask > 0).astype(np.uint8)
    if binary.max() == 0:
        return binary
    if sk_skeletonize is not None:
        return sk_skeletonize(binary.astype(bool)).astype(np.uint8)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return (cv2.ximgproc.thinning(binary * 255) > 0).astype(np.uint8)
    return morphological_skeleton(binary)


def neighbor_count(binary_mask):
    bw = (binary_mask > 0).astype(np.uint8)
    h, w = bw.shape
    nbr = np.zeros_like(bw, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            y0 = max(0, -dy)
            y1 = min(h, h - dy)
            x0 = max(0, -dx)
            x1 = min(w, w - dx)
            nbr[y0:y1, x0:x1] += bw[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
    return nbr


def _endpoint_coords_and_dirs(skel_binary):
    sk = (skel_binary > 0).astype(np.uint8)
    nbr = neighbor_count(sk)
    endpoints = ((sk == 1) & (nbr == 1))
    ys, xs = np.where(endpoints)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 2), dtype=np.float32)

    coords = np.column_stack([ys, xs]).astype(np.int32)
    dirs = np.zeros((len(coords), 2), dtype=np.float32)
    h, w = sk.shape
    for i, (y, x) in enumerate(coords):
        found = False
        for dy in (-1, 0, 1):
            if found:
                break
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                yy = int(y + dy)
                xx = int(x + dx)
                if yy < 0 or yy >= h or xx < 0 or xx >= w:
                    continue
                if sk[yy, xx] > 0:
                    vec = np.array([float(x - xx), float(y - yy)], dtype=np.float32)
                    norm = float(np.linalg.norm(vec))
                    if norm > 1e-6:
                        dirs[i] = vec / norm
                    found = True
                    break
    return coords, dirs


def _line_existing_fraction(binary_mask, p0, p1):
    y0, x0 = p0
    y1, x1 = p1
    steps = int(max(abs(int(y1) - int(y0)), abs(int(x1) - int(x0))))
    if steps <= 0:
        return 1.0
    ys = np.linspace(float(y0), float(y1), steps + 1).round().astype(np.int32)
    xs = np.linspace(float(x0), float(x1), steps + 1).round().astype(np.int32)
    h, w = binary_mask.shape
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)
    vals = binary_mask[ys, xs]
    return float(vals.mean()) if len(vals) > 0 else 0.0


def _build_pixel_buckets(coords, cell_size):
    buckets = {}
    cell_size = int(max(1, cell_size))
    for idx, (y, x) in enumerate(coords):
        key = (int(y // cell_size), int(x // cell_size))
        buckets.setdefault(key, []).append(idx)
    return buckets


def _link_endpoints_to_near_lines_once(
    skel_binary,
    max_dist=4,
    min_dot=0.0,
    max_existing_fraction=0.85,
):
    sk = (skel_binary > 0).astype(np.uint8)
    coords, dirs = _endpoint_coords_and_dirs(sk)
    if len(coords) == 0:
        return sk, 0

    comp_count, comp_labels, _, _ = cv2.connectedComponentsWithStats(sk, connectivity=8)
    if comp_count <= 1:
        return sk, 0

    line_ys, line_xs = np.where(sk > 0)
    line_coords = np.column_stack([line_ys, line_xs]).astype(np.int32)
    if len(line_coords) == 0:
        return sk, 0

    max_dist = int(max(1, max_dist))
    max_d2 = float(max_dist * max_dist)
    min_dot = float(min(max(min_dot, -1.0), 1.0))
    max_existing_fraction = float(min(max(max_existing_fraction, 0.0), 1.0))

    buckets = _build_pixel_buckets(line_coords, max_dist)
    drawn = 0

    for i, (y, x) in enumerate(coords):
        comp_i = int(comp_labels[y, x])
        if comp_i <= 0:
            continue

        best_idx = -1
        best_d2 = 1e30
        key = (int(y // max_dist), int(x // max_dist))

        for gy in range(key[0] - 1, key[0] + 2):
            for gx in range(key[1] - 1, key[1] + 2):
                for cand_idx in buckets.get((gy, gx), []):
                    yy, xx = line_coords[cand_idx]
                    if yy == y and xx == x:
                        continue
                    comp_j = int(comp_labels[yy, xx])
                    if comp_j <= 0 or comp_j == comp_i:
                        continue

                    dy = float(yy - y)
                    dx = float(xx - x)
                    d2 = dx * dx + dy * dy
                    if d2 <= 1.0 or d2 > max_d2:
                        continue

                    dist = float(np.sqrt(d2))
                    vx = dx / dist
                    vy = dy / dist
                    dot = float(dirs[i][0] * vx + dirs[i][1] * vy)
                    if dot < min_dot:
                        continue

                    frac = _line_existing_fraction(sk, (y, x), (yy, xx))
                    if frac > max_existing_fraction:
                        continue

                    if d2 < best_d2:
                        best_d2 = d2
                        best_idx = cand_idx

        if best_idx >= 0:
            yy, xx = line_coords[best_idx]
            cv2.line(sk, (int(x), int(y)), (int(xx), int(yy)), 1, 1)
            drawn += 1

    return sk, int(drawn)


def _candidate_junction_snap_repair(
    skeleton_binary,
    snap_iters=DEFAULT_GRAPH_JUNCTION_SNAP_ITERS,
    snap_dist=DEFAULT_GRAPH_JUNCTION_SNAP_DIST,
    snap_min_dot=DEFAULT_GRAPH_JUNCTION_SNAP_MIN_DOT,
    snap_max_existing_fraction=DEFAULT_GRAPH_JUNCTION_SNAP_MAX_EXISTING_FRACTION,
):
    sk = (skeleton_binary > 0).astype(np.uint8)
    total_links = 0
    applied_iters = 0
    for _ in range(int(max(0, snap_iters))):
        sk, links = _link_endpoints_to_near_lines_once(
            skel_binary=sk,
            max_dist=int(max(1, snap_dist)),
            min_dot=float(snap_min_dot),
            max_existing_fraction=float(snap_max_existing_fraction),
        )
        if links <= 0:
            break
        total_links += int(links)
        applied_iters += 1
        sk = thin_binary(sk)
    stats = {"links_added": int(total_links), "iters_applied": int(applied_iters)}
    return sk, stats


def build_topology_gate(image, line_threshold=127, branch_degree=3, gate_radius=6):
    """
    Build a gate map that is true near likely junction pixels only.
    """
    line = auto_line_mask(image, line_threshold=line_threshold)
    thin = thin_binary(line)
    nbr = neighbor_count(thin)
    junction = ((thin == 1) & (nbr >= int(max(3, branch_degree)))).astype(np.uint8)

    gate = junction
    if int(gate_radius) > 0:
        k = int(gate_radius) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        gate = cv2.dilate(gate, kernel, iterations=1)

    stats = {
        "line_pixels": int(line.sum()),
        "thin_pixels": int(thin.sum()),
        "junction_pixels": int(junction.sum()),
        "gate_pixels": int(gate.sum()),
    }
    return gate, stats


def filter_detections_by_gate(detections, gate_mask):
    if gate_mask is None:
        return detections
    h, w = gate_mask.shape
    filtered = []
    for det in detections:
        x = int(np.clip(det["x"], 0, w - 1))
        y = int(np.clip(det["y"], 0, h - 1))
        if gate_mask[y, x] > 0:
            filtered.append(det)
    return filtered


def _disk_kernel(radius):
    radius = int(max(0, radius))
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def corner_like_two_neighbor_mask(thin_binary, nbr_map, max_angle_deg=DEFAULT_SPARSE_CORNER_MAX_ANGLE):
    thin = (thin_binary > 0).astype(np.uint8)
    nbr = np.asarray(nbr_map)
    h, w = thin.shape
    out = np.zeros_like(thin, dtype=np.uint8)
    ys, xs = np.where((thin == 1) & (nbr == 2))
    if len(xs) == 0:
        return out

    max_angle = float(max(0.0, min(180.0, max_angle_deg)))
    for y, x in zip(ys, xs):
        y0 = max(0, int(y) - 1)
        y1 = min(h, int(y) + 2)
        x0 = max(0, int(x) - 1)
        x1 = min(w, int(x) + 2)
        patch = thin[y0:y1, x0:x1]
        ny, nx = np.where(patch > 0)
        if len(nx) < 3:
            continue
        neighbors = []
        for py, px in zip(ny, nx):
            gy = int(py + y0)
            gx = int(px + x0)
            if gy == int(y) and gx == int(x):
                continue
            neighbors.append((gy, gx))
        if len(neighbors) != 2:
            continue
        (y1n, x1n), (y2n, x2n) = neighbors
        v1 = np.array([float(x1n - x), float(y1n - y)], dtype=float)
        v2 = np.array([float(x2n - x), float(y2n - y)], dtype=float)
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 <= 1e-6 or n2 <= 1e-6:
            continue
        dot = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        ang = float(np.degrees(np.arccos(dot)))
        if ang <= max_angle:
            out[int(y), int(x)] = 1
    return out


def _angles_for_vectors(unit_vectors):
    angles = []
    for i in range(len(unit_vectors)):
        for j in range(i + 1, len(unit_vectors)):
            dot = float(np.clip(np.dot(unit_vectors[i], unit_vectors[j]), -1.0, 1.0))
            angles.append(float(np.degrees(np.arccos(dot))))
    return sorted(angles)


def _classify_three_arm(
    angles,
    t_min_largest_angle=DEFAULT_GRAPH_T_MIN_LARGEST_ANGLE,
    t_max_side_angle=DEFAULT_GRAPH_T_MAX_SIDE_ANGLE,
    ambiguity_margin=6.0,
    classifier_mode=DEFAULT_THREE_ARM_CLASSIFIER,
    return_debug=False,
):
    a0, a1, a2 = sorted(angles)
    mode = str(classifier_mode).strip().lower()
    if mode not in THREE_ARM_CLASSIFIER_MODES:
        mode = DEFAULT_THREE_ARM_CLASSIFIER
    if mode == "robust_vote":
        # Vote-based robust mode requires local center jitter search and cannot be evaluated from a single angle triplet.
        mode = "robust"

    t_guardrail = (a2 >= float(t_min_largest_angle)) and (a1 <= float(t_max_side_angle)) and (a0 <= float(t_max_side_angle))
    if mode == "robust":
        t_target = abs(a2 - 180.0) + 0.75 * (abs(a0 - 90.0) + abs(a1 - 90.0))
        t_largest_under = max(0.0, float(t_min_largest_angle) - float(a2))
        t_side_over = max(0.0, float(a0) - float(t_max_side_angle)) + max(0.0, float(a1) - float(t_max_side_angle))
        t_side_balance = abs(a0 - a1)
        err_t = t_target + 1.30 * t_largest_under + 1.10 * t_side_over + 0.20 * t_side_balance

        y_target = abs(a0 - 120.0) + abs(a1 - 120.0) + abs(a2 - 120.0)
        y_spread = max(0.0, a2 - a0)
        y_mid = 0.5 * (a0 + a1)
        err_y = y_target + 0.25 * y_spread + 0.20 * abs(y_mid - 120.0)
    else:
        err_t = abs(a2 - 180.0) + 0.7 * (abs(a0 - 90.0) + abs(a1 - 90.0))
        err_y = abs(a0 - 120.0) + abs(a1 - 120.0) + abs(a2 - 120.0)

    if mode == "legacy":
        label = "T" if t_guardrail else "Y"
        class_error = float(err_t if label == "T" else err_y)
        debug = {
            "angles": [float(a0), float(a1), float(a2)],
            "err_t": float(err_t),
            "err_y": float(err_y),
            "t_guardrail": bool(t_guardrail),
            "is_ambiguous": False,
            "classifier_mode": mode,
        }
        if return_debug:
            return label, class_error, debug
        return label, class_error

    choose_t = bool(t_guardrail and (err_t <= (err_y + float(ambiguity_margin))))
    label = "T" if choose_t else "Y"
    class_error = float(err_t if choose_t else err_y)
    debug = {
        "angles": [float(a0), float(a1), float(a2)],
        "err_t": float(err_t),
        "err_y": float(err_y),
        "t_guardrail": bool(t_guardrail),
        "is_ambiguous": bool(abs(err_t - err_y) <= float(ambiguity_margin)),
        "classifier_mode": mode,
    }
    if return_debug:
        return label, class_error, debug
    return label, class_error


def _local_skeleton_candidates(thin_binary, center_y, center_x, radius=DEFAULT_ROBUST_JITTER_RADIUS):
    h, w = thin_binary.shape
    y0, y1, x0, x1 = _crop_bounds(h, w, center_y, center_x, int(max(0, radius)))
    ys, xs = np.where(thin_binary[y0:y1, x0:x1] > 0)
    pts = set()
    for yy, xx in zip(ys.tolist(), xs.tolist()):
        pts.add((int(yy + y0), int(xx + x0)))
    pts.add((int(center_y), int(center_x)))
    out = sorted(
        pts,
        key=lambda p: (
            float((p[0] - int(center_y)) ** 2 + (p[1] - int(center_x)) ** 2),
            int(p[0]),
            int(p[1]),
        ),
    )
    return out


def _robust_jitter_hypotheses(
    thin_binary,
    base_y,
    base_x,
    inner_radius,
    outer_radius,
    min_arm_pixels,
    min_arm_span,
    t_min_largest_angle,
    t_max_side_angle,
    ambiguity_margin,
    jitter_radius=DEFAULT_ROBUST_JITTER_RADIUS,
    distance_penalty=DEFAULT_ROBUST_JITTER_DIST_PENALTY,
    outer_offsets=(0,),
):
    candidates = _local_skeleton_candidates(thin_binary, int(base_y), int(base_x), radius=jitter_radius)
    hypotheses = []
    inner_radius = int(max(1, inner_radius))
    base_outer = int(max(inner_radius + 1, outer_radius))
    offsets = sorted({int(v) for v in (outer_offsets or (0,))})
    effective_outers = []
    for off in offsets:
        eff = int(max(inner_radius + 1, base_outer + int(off)))
        if eff not in effective_outers:
            effective_outers.append(eff)

    for cy, cx in candidates:
        d = float(np.hypot(float(cy - int(base_y)), float(cx - int(base_x))))
        for eff_outer in effective_outers:
            ring = extract_arm_components_in_ring(
                thin_binary,
                cy,
                cx,
                inner_radius=inner_radius,
                outer_radius=eff_outer,
                min_arm_pixels=min_arm_pixels,
                min_arm_span=min_arm_span,
            )
            if int(ring["arm_count"]) != 3:
                continue
            angles = _angles_for_vectors(ring["vectors"])
            _, _, dbg = _classify_three_arm(
                angles,
                t_min_largest_angle=t_min_largest_angle,
                t_max_side_angle=t_max_side_angle,
                ambiguity_margin=ambiguity_margin,
                classifier_mode="robust",
                return_debug=True,
            )
            total_t = float(dbg.get("err_t", float("inf"))) + float(distance_penalty) * d
            total_y = float(dbg.get("err_y", float("inf"))) + float(distance_penalty) * d
            hypotheses.append(
                {
                    "angles": list(dbg.get("angles", [])),
                    "ring": ring,
                    "center_y": int(cy),
                    "center_x": int(cx),
                    "distance": d,
                    "t_guardrail": bool(dbg.get("t_guardrail", False)),
                    "err_t": float(dbg.get("err_t", float("inf"))),
                    "err_y": float(dbg.get("err_y", float("inf"))),
                    "total_t": float(total_t),
                    "total_y": float(total_y),
                    "effective_outer_radius": int(eff_outer),
                }
            )
    return hypotheses, candidates


def _robust_refine_three_arm_with_jitter(
    thin_binary,
    base_y,
    base_x,
    inner_radius,
    outer_radius,
    min_arm_pixels,
    min_arm_span,
    t_min_largest_angle,
    t_max_side_angle,
    ambiguity_margin,
    jitter_radius=DEFAULT_ROBUST_JITTER_RADIUS,
    distance_penalty=DEFAULT_ROBUST_JITTER_DIST_PENALTY,
):
    hypotheses, candidates = _robust_jitter_hypotheses(
        thin_binary=thin_binary,
        base_y=base_y,
        base_x=base_x,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        min_arm_pixels=min_arm_pixels,
        min_arm_span=min_arm_span,
        t_min_largest_angle=t_min_largest_angle,
        t_max_side_angle=t_max_side_angle,
        ambiguity_margin=ambiguity_margin,
        jitter_radius=jitter_radius,
        distance_penalty=distance_penalty,
        outer_offsets=(0,),
    )
    if not hypotheses:
        return None

    best_t = min(hypotheses, key=lambda h: (float(h["total_t"]), float(h["distance"]), int(h["center_y"]), int(h["center_x"])))
    best_y = min(hypotheses, key=lambda h: (float(h["total_y"]), float(h["distance"]), int(h["center_y"]), int(h["center_x"])))
    winner_label = "T" if float(best_t["total_t"]) <= float(best_y["total_y"]) + float(ambiguity_margin) else "Y"
    winner = best_t if winner_label == "T" else best_y
    debug = {
        "angles": list(winner["angles"]),
        "err_t": float(best_t["err_t"]),
        "err_y": float(best_y["err_y"]),
        "t_guardrail": bool(best_t["t_guardrail"]),
        "is_ambiguous": bool(
            abs(float(best_t["total_t"]) - float(best_y["total_y"])) <= float(ambiguity_margin)
        ),
        "classifier_mode": "robust",
        "jitter_valid_hypotheses": int(len(hypotheses)),
        "jitter_candidates": int(len(candidates)),
        "jitter_distance": float(winner["distance"]),
        "effective_outer_radius": int(winner.get("effective_outer_radius", int(outer_radius))),
    }
    return {
        "label": str(winner_label),
        "class_error": float(winner["total_t"] if winner_label == "T" else winner["total_y"]),
        "center_y": int(winner["center_y"]),
        "center_x": int(winner["center_x"]),
        "ring": winner["ring"],
        "debug": debug,
    }


def _robust_vote_three_arm_with_jitter(
    thin_binary,
    base_y,
    base_x,
    inner_radius,
    outer_radius,
    min_arm_pixels,
    min_arm_span,
    t_min_largest_angle,
    t_max_side_angle,
    ambiguity_margin,
    jitter_radius=DEFAULT_ROBUST_JITTER_RADIUS,
    distance_penalty=DEFAULT_ROBUST_JITTER_DIST_PENALTY,
    vote_temperature=DEFAULT_ROBUST_VOTE_TEMPERATURE,
    t_guardrail_scale=DEFAULT_ROBUST_VOTE_GUARDRAIL_SCALE,
    vote_distance_weight=DEFAULT_ROBUST_VOTE_DISTANCE_WEIGHT,
    outer_offsets=DEFAULT_ROBUST_VOTE_OUTER_OFFSETS,
):
    hypotheses, candidates = _robust_jitter_hypotheses(
        thin_binary=thin_binary,
        base_y=base_y,
        base_x=base_x,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        min_arm_pixels=min_arm_pixels,
        min_arm_span=min_arm_span,
        t_min_largest_angle=t_min_largest_angle,
        t_max_side_angle=t_max_side_angle,
        ambiguity_margin=ambiguity_margin,
        jitter_radius=jitter_radius,
        distance_penalty=distance_penalty,
        outer_offsets=outer_offsets,
    )
    if not hypotheses:
        return None

    best_t = min(hypotheses, key=lambda h: (float(h["total_t"]), float(h["distance"]), int(h["center_y"]), int(h["center_x"])))
    best_y = min(hypotheses, key=lambda h: (float(h["total_y"]), float(h["distance"]), int(h["center_y"]), int(h["center_x"])))

    temp = max(1e-3, float(vote_temperature))
    guard_scale = float(np.clip(float(t_guardrail_scale), 0.0, 1.0))
    dist_w = max(0.0, float(vote_distance_weight))
    eps = 1e-12
    vote_t = 0.0
    vote_y = 0.0
    for h in hypotheses:
        dist_factor = 1.0 / (1.0 + dist_w * float(h["distance"]))
        t_like = float(np.exp(-float(h["total_t"]) / temp))
        y_like = float(np.exp(-float(h["total_y"]) / temp))
        if not bool(h["t_guardrail"]):
            t_like *= guard_scale
        vote_t += dist_factor * t_like
        vote_y += dist_factor * y_like

    comb_t = float(best_t["total_t"]) - temp * float(np.log(vote_t + eps))
    comb_y = float(best_y["total_y"]) - temp * float(np.log(vote_y + eps))
    choose_t = bool(best_t["t_guardrail"] and (comb_t <= (comb_y + float(ambiguity_margin))))
    winner = best_t if choose_t else best_y
    winner_label = "T" if choose_t else "Y"
    winner_error = float(max(0.0, comb_t if choose_t else comb_y))
    log_ratio = float(np.log((vote_t + eps) / (vote_y + eps)))
    vote_margin = float(abs(log_ratio) * temp)

    debug = {
        "angles": list(winner["angles"]),
        "err_t": float(best_t["err_t"]),
        "err_y": float(best_y["err_y"]),
        "t_guardrail": bool(best_t["t_guardrail"]),
        "is_ambiguous": bool(abs(comb_t - comb_y) <= float(ambiguity_margin)),
        "classifier_mode": "robust_vote",
        "vote_t": float(vote_t),
        "vote_y": float(vote_y),
        "vote_margin": float(vote_margin),
        "combined_t": float(comb_t),
        "combined_y": float(comb_y),
        "jitter_valid_hypotheses": int(len(hypotheses)),
        "jitter_candidates": int(len(candidates)),
        "jitter_distance": float(winner["distance"]),
        "effective_outer_radius": int(winner.get("effective_outer_radius", int(outer_radius))),
    }
    return {
        "label": str(winner_label),
        "class_error": float(winner_error),
        "center_y": int(winner["center_y"]),
        "center_x": int(winner["center_x"]),
        "ring": winner["ring"],
        "debug": debug,
    }


def _crop_bounds(height, width, y, x, radius):
    r = int(max(1, radius))
    y0 = max(0, int(y) - r)
    y1 = min(int(height), int(y) + r + 1)
    x0 = max(0, int(x) - r)
    x1 = min(int(width), int(x) + r + 1)
    return y0, y1, x0, x1


def extract_arm_components_in_ring(
    thin_binary,
    center_y,
    center_x,
    inner_radius,
    outer_radius,
    min_arm_pixels=DEFAULT_GRAPH_MIN_ARM_PIXELS,
    min_arm_span=DEFAULT_GRAPH_MIN_ARM_SPAN,
    bridge_radius=DEFAULT_GRAPH_RING_GAP_BRIDGE_RADIUS,
):
    h, w = thin_binary.shape
    inner_radius = int(max(1, inner_radius))
    outer_radius = int(max(inner_radius + 1, outer_radius))
    y0, y1, x0, x1 = _crop_bounds(h, w, center_y, center_x, outer_radius + 1)

    if y0 >= y1 or x0 >= x1:
        return {
            "arms": [],
            "valid_arms": [],
            "vectors": [],
            "arm_count": 0,
            "ring_pixels": 0,
            "crop_bounds": [int(y0), int(y1), int(x0), int(x1)],
        }

    crop = (thin_binary[y0:y1, x0:x1] > 0).astype(np.uint8)
    bridge_radius = int(max(0, bridge_radius))
    if bridge_radius > 0:
        # Locally bridge 1-2 px breaks in sparse/noisy skeletons before arm component extraction.
        bridge_kernel = _disk_kernel(max(1, bridge_radius))
        crop_for_arms = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, bridge_kernel, iterations=1)
    else:
        crop_for_arms = crop
    yy, xx = np.indices(crop.shape)
    gy = yy + int(y0)
    gx = xx + int(x0)
    d2 = (gy - int(center_y)) * (gy - int(center_y)) + (gx - int(center_x)) * (gx - int(center_x))

    inner2 = int(inner_radius * inner_radius)
    outer2 = int(outer_radius * outer_radius)
    ring = ((d2 <= outer2) & (d2 > inner2)).astype(np.uint8)
    arm_pixels = ((crop_for_arms > 0) & (ring > 0)).astype(np.uint8)
    ring_pixels = int(ring.sum())

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(arm_pixels, connectivity=8)
    arms = []
    valid_arms = []
    vectors = []
    outer_half = float(inner_radius + 0.5 * (outer_radius - inner_radius))
    min_pixels = int(max(1, min_arm_pixels))
    min_span = float(max(0.0, min_arm_span))

    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        ys, xs = np.where(labels == idx)
        if len(xs) == 0:
            continue

        gyi = ys + int(y0)
        gxi = xs + int(x0)
        dy = gyi.astype(np.float32) - float(center_y)
        dx = gxi.astype(np.float32) - float(center_x)
        radii = np.sqrt(dx * dx + dy * dy)
        radial_span = float(radii.max() - radii.min()) if len(radii) > 0 else 0.0
        has_outer_half = bool(np.any(radii >= outer_half))

        arm_cx = float(gxi.mean())
        arm_cy = float(gyi.mean())
        vec = np.array([arm_cx - float(center_x), arm_cy - float(center_y)], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        unit_vec = (vec / norm) if norm > 1e-6 else None

        arm_info = {
            "index": int(idx),
            "area": int(area),
            "radial_span": float(radial_span),
            "has_outer_half": bool(has_outer_half),
            "centroid_x": float(arm_cx),
            "centroid_y": float(arm_cy),
            "unit_vec": unit_vec,
            "valid": bool((area >= min_pixels) and (radial_span >= min_span) and has_outer_half and (unit_vec is not None)),
        }
        arms.append(arm_info)
        if arm_info["valid"]:
            valid_arms.append(arm_info)
            vectors.append(unit_vec)

    return {
        "arms": arms,
        "valid_arms": valid_arms,
        "vectors": vectors,
        "arm_count": int(len(valid_arms)),
        "ring_pixels": int(ring_pixels),
        "bridge_radius": int(bridge_radius),
        "crop_bounds": [int(y0), int(y1), int(x0), int(x1)],
    }


def local_geometry_analysis(
    thin_binary,
    y,
    x,
    inner_radius=DEFAULT_GRAPH_ARM_INNER_RADIUS,
    outer_radius=DEFAULT_GRAPH_ARM_OUTER_RADIUS,
    min_arm_pixels=DEFAULT_GRAPH_MIN_ARM_PIXELS,
    min_arm_span=DEFAULT_GRAPH_MIN_ARM_SPAN,
    t_min_largest_angle=DEFAULT_GRAPH_T_MIN_LARGEST_ANGLE,
    t_max_side_angle=DEFAULT_GRAPH_T_MAX_SIDE_ANGLE,
    ambiguity_margin=6.0,
    classifier_mode=DEFAULT_THREE_ARM_CLASSIFIER,
    snap_radius=None,
    ring_bridge_radius=DEFAULT_GRAPH_RING_GAP_BRIDGE_RADIUS,
):
    original_y = int(y)
    original_x = int(x)
    if snap_radius is None:
        snap_radius = max(2, int(inner_radius))

    if int(snap_radius) <= 0:
        h, w = thin_binary.shape
        yy = int(np.clip(original_y, 0, h - 1))
        xx = int(np.clip(original_x, 0, w - 1))
        snapped_y, snapped_x = yy, xx
        snapped_ok = bool(thin_binary[yy, xx] > 0)
    else:
        snapped_y, snapped_x, snapped_ok = nearest_line_point(
            thin_binary,
            original_y,
            original_x,
            radius=int(max(1, snap_radius)),
        )
    if not snapped_ok:
        return {
            "label": "",
            "arm_count": 0,
            "score": 0.0,
            "class_error": float("inf"),
            "angles": [],
            "err_t": float("inf"),
            "err_y": float("inf"),
            "is_ambiguous": True,
            "t_guardrail": False,
            "snapped_ok": False,
            "snapped_x": int(original_x),
            "snapped_y": int(original_y),
            "snap_distance": float("inf"),
            "ring": extract_arm_components_in_ring(
                thin_binary,
                original_y,
                original_x,
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                min_arm_pixels=min_arm_pixels,
                min_arm_span=min_arm_span,
                bridge_radius=ring_bridge_radius,
            ),
        }

    ring = extract_arm_components_in_ring(
        thin_binary,
        snapped_y,
        snapped_x,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        min_arm_pixels=min_arm_pixels,
        min_arm_span=min_arm_span,
        bridge_radius=ring_bridge_radius,
    )
    vectors = ring["vectors"]
    arm_num = int(ring["arm_count"])
    label = ""
    class_error = float("inf")
    debug = {
        "angles": [],
        "err_t": float("inf"),
        "err_y": float("inf"),
        "is_ambiguous": True,
        "t_guardrail": False,
    }

    if arm_num >= 4:
        label = "X"
        class_error = 0.0
        debug["is_ambiguous"] = False
    elif arm_num == 3:
        mode = str(classifier_mode).strip().lower()
        if mode in ("robust", "robust_vote"):
            if mode == "robust_vote":
                refined = _robust_vote_three_arm_with_jitter(
                    thin_binary=thin_binary,
                    base_y=snapped_y,
                    base_x=snapped_x,
                    inner_radius=inner_radius,
                    outer_radius=outer_radius,
                    min_arm_pixels=min_arm_pixels,
                    min_arm_span=min_arm_span,
                    t_min_largest_angle=t_min_largest_angle,
                    t_max_side_angle=t_max_side_angle,
                    ambiguity_margin=ambiguity_margin,
                )
            else:
                refined = _robust_refine_three_arm_with_jitter(
                    thin_binary=thin_binary,
                    base_y=snapped_y,
                    base_x=snapped_x,
                    inner_radius=inner_radius,
                    outer_radius=outer_radius,
                    min_arm_pixels=min_arm_pixels,
                    min_arm_span=min_arm_span,
                    t_min_largest_angle=t_min_largest_angle,
                    t_max_side_angle=t_max_side_angle,
                    ambiguity_margin=ambiguity_margin,
                )
            if refined is not None:
                label = str(refined["label"])
                class_error = float(refined["class_error"])
                snapped_y = int(refined["center_y"])
                snapped_x = int(refined["center_x"])
                ring = refined["ring"]
                arm_num = int(ring["arm_count"])
                debug.update(refined["debug"])
            else:
                angles = _angles_for_vectors(vectors)
                label, class_error, dbg = _classify_three_arm(
                    angles,
                    t_min_largest_angle=t_min_largest_angle,
                    t_max_side_angle=t_max_side_angle,
                    ambiguity_margin=ambiguity_margin,
                    classifier_mode="robust",
                    return_debug=True,
                )
                debug.update(dbg)
        else:
            angles = _angles_for_vectors(vectors)
            label, class_error, dbg = _classify_three_arm(
                angles,
                t_min_largest_angle=t_min_largest_angle,
                t_max_side_angle=t_max_side_angle,
                ambiguity_margin=ambiguity_margin,
                classifier_mode=classifier_mode,
                return_debug=True,
            )
            debug.update(dbg)
    score = float(1.0 / (1.0 + max(0.0, class_error))) if np.isfinite(class_error) else 0.0

    return {
        "label": str(label),
        "arm_count": int(arm_num),
        "score": float(score),
        "class_error": float(class_error),
        "angles": list(debug.get("angles", [])),
        "err_t": float(debug.get("err_t", float("inf"))),
        "err_y": float(debug.get("err_y", float("inf"))),
        "is_ambiguous": bool(debug.get("is_ambiguous", True)),
        "t_guardrail": bool(debug.get("t_guardrail", False)),
        "snapped_ok": bool(snapped_ok),
        "snapped_x": int(snapped_x),
        "snapped_y": int(snapped_y),
        "snap_distance": float(np.hypot(float(snapped_y - original_y), float(snapped_x - original_x))),
        "ring": ring,
    }


def _legacy_classify_component_at_centroid(
    thin_binary,
    component_mask,
    centroid_x,
    centroid_y,
    arm_inner_radius,
    arm_outer_radius,
    min_arm_pixels,
    t_min_largest_angle,
    t_max_side_angle,
    classifier_mode=DEFAULT_THREE_ARM_CLASSIFIER,
):
    outer_kernel = _disk_kernel(max(arm_outer_radius, arm_inner_radius + 1))
    inner_kernel = _disk_kernel(max(1, arm_inner_radius))
    outer = cv2.dilate(component_mask, outer_kernel, iterations=1)
    inner = cv2.dilate(component_mask, inner_kernel, iterations=1)
    ring = ((outer > 0) & (inner == 0)).astype(np.uint8)
    arm_pixels = ((thin_binary > 0) & (ring > 0)).astype(np.uint8)
    arm_count, arm_labels, arm_stats, _ = cv2.connectedComponentsWithStats(arm_pixels, connectivity=8)
    vectors = []
    for arm_idx in range(1, arm_count):
        arm_area = int(arm_stats[arm_idx, cv2.CC_STAT_AREA])
        if arm_area < int(max(1, min_arm_pixels)):
            continue
        ys, xs = np.where(arm_labels == arm_idx)
        if len(xs) == 0:
            continue
        arm_cx = float(xs.mean())
        arm_cy = float(ys.mean())
        vec = np.array([arm_cx - float(centroid_x), arm_cy - float(centroid_y)], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            continue
        vectors.append(vec / norm)

    arm_num = len(vectors)
    if arm_num >= 4:
        return "X", min(1.0, 0.6 + 0.1 * min(arm_num, 5)), int(arm_num)
    if arm_num == 3:
        angles = _angles_for_vectors(vectors)
        label, err = _classify_three_arm(
            angles,
            t_min_largest_angle=t_min_largest_angle,
            t_max_side_angle=t_max_side_angle,
            classifier_mode=classifier_mode,
        )
        return label, float(1.0 / (1.0 + max(0.0, err))), int(arm_num)
    return "", 0.0, int(arm_num)


def choose_best_junction_center(
    thin_binary,
    component_mask,
    centroid_x,
    centroid_y,
    inner_radius,
    outer_radius,
    min_arm_pixels=DEFAULT_GRAPH_MIN_ARM_PIXELS,
    min_arm_span=DEFAULT_GRAPH_MIN_ARM_SPAN,
    t_min_largest_angle=DEFAULT_GRAPH_T_MIN_LARGEST_ANGLE,
    t_max_side_angle=DEFAULT_GRAPH_T_MAX_SIDE_ANGLE,
    ambiguity_margin=6.0,
    classifier_mode=DEFAULT_THREE_ARM_CLASSIFIER,
    ring_bridge_radius=DEFAULT_GRAPH_RING_GAP_BRIDGE_RADIUS,
    max_centers=1,
    separation_px=None,
):
    h, w = thin_binary.shape
    cand_set = set()
    ys, xs = np.where(component_mask > 0)
    for yy, xx in zip(ys.tolist(), xs.tolist()):
        cand_set.add((int(yy), int(xx)))

    if len(cand_set) == 0:
        return None

    # Include a tight neighborhood around the branch component to recover from centroid offsets.
    dilated = cv2.dilate(component_mask, _disk_kernel(1), iterations=1)
    dy, dx = np.where((dilated > 0) & (thin_binary > 0))
    for yy, xx in zip(dy.tolist(), dx.tolist()):
        cand_set.add((int(yy), int(xx)))

    ranked = []
    c_y = float(centroid_y)
    c_x = float(centroid_x)
    for yy, xx in sorted(cand_set):
        geom = local_geometry_analysis(
            thin_binary=thin_binary,
            y=yy,
            x=xx,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            min_arm_pixels=min_arm_pixels,
            min_arm_span=min_arm_span,
            t_min_largest_angle=t_min_largest_angle,
            t_max_side_angle=t_max_side_angle,
            ambiguity_margin=ambiguity_margin,
            classifier_mode=classifier_mode,
            snap_radius=max(2, int(inner_radius)),
            ring_bridge_radius=ring_bridge_radius,
        )
        if geom["arm_count"] < 3 or not geom["label"]:
            continue
        center_dist = float(np.hypot(float(geom["snapped_y"]) - c_y, float(geom["snapped_x"]) - c_x))
        rank_key = (
            -int(geom["arm_count"]),
            float(geom["class_error"]),
            float(center_dist),
            int(geom["snapped_y"]),
            int(geom["snapped_x"]),
        )
        ranked.append((rank_key, geom))

    if not ranked:
        return None

    ranked.sort(key=lambda item: item[0])
    max_centers = max(1, int(max_centers))
    min_sep = float(max(1.0, float(separation_px if separation_px is not None else inner_radius)))
    min_sep2 = float(min_sep * min_sep)

    selected = []
    for _, geom in ranked:
        sy = int(geom["snapped_y"])
        sx = int(geom["snapped_x"])
        too_close = False
        for picked in selected:
            dy = float(sy - int(picked["snapped_y"]))
            dx = float(sx - int(picked["snapped_x"]))
            if dx * dx + dy * dy < min_sep2:
                too_close = True
                break
        if too_close:
            continue
        selected.append(geom)
        if len(selected) >= max_centers:
            break

    if not selected:
        return None if max_centers == 1 else []

    def _to_center_record(geom):
        return {
            "x": int(geom["snapped_x"]),
            "y": int(geom["snapped_y"]),
            "label": str(geom["label"]),
            "score": float(geom["score"]),
            "arm_count": int(geom["arm_count"]),
            "geometry": geom,
        }

    if max_centers == 1:
        return _to_center_record(selected[0])
    return [_to_center_record(g) for g in selected]


def graph_junction_detections(
    image,
    line_threshold=127,
    branch_degree=3,
    arm_inner_radius=DEFAULT_GRAPH_ARM_INNER_RADIUS,
    arm_outer_radius=DEFAULT_GRAPH_ARM_OUTER_RADIUS,
    min_arm_pixels=DEFAULT_GRAPH_MIN_ARM_PIXELS,
    min_arm_span=DEFAULT_GRAPH_MIN_ARM_SPAN,
    min_branch_component_area=DEFAULT_GRAPH_MIN_BRANCH_COMPONENT_AREA,
    max_centers_per_component=DEFAULT_GRAPH_MAX_CENTERS_PER_COMPONENT,
    ring_gap_bridge_radius=DEFAULT_GRAPH_RING_GAP_BRIDGE_RADIUS,
    sparse_recovery=DEFAULT_GRAPH_SPARSE_RECOVERY,
    sparse_recovery_support_radius=DEFAULT_GRAPH_SPARSE_RECOVERY_SUPPORT_RADIUS,
    sparse_recovery_endpoint_radius=DEFAULT_GRAPH_SPARSE_RECOVERY_ENDPOINT_RADIUS,
    sparse_recovery_max_component_area=DEFAULT_GRAPH_SPARSE_RECOVERY_MAX_COMPONENT_AREA,
    sparse_recovery_min_score=DEFAULT_GRAPH_SPARSE_RECOVERY_MIN_SCORE,
    junction_snap_repair=DEFAULT_GRAPH_JUNCTION_SNAP_REPAIR,
    junction_snap_iters=DEFAULT_GRAPH_JUNCTION_SNAP_ITERS,
    junction_snap_dist=DEFAULT_GRAPH_JUNCTION_SNAP_DIST,
    junction_snap_min_dot=DEFAULT_GRAPH_JUNCTION_SNAP_MIN_DOT,
    junction_snap_max_existing_fraction=DEFAULT_GRAPH_JUNCTION_SNAP_MAX_EXISTING_FRACTION,
    sparse_corner_recovery=False,
    sparse_corner_max_angle=DEFAULT_SPARSE_CORNER_MAX_ANGLE,
    sparse_corner_endpoint_radius=0,
    sparse_corner_min_score=None,
    centroid_best_fallback=False,
    nms_distance=DEFAULT_NMS_DISTANCE,
    t_min_largest_angle=DEFAULT_GRAPH_T_MIN_LARGEST_ANGLE,
    t_max_side_angle=DEFAULT_GRAPH_T_MAX_SIDE_ANGLE,
    ambiguity_margin=6.0,
    center_mode=DEFAULT_GRAPH_CENTER_MODE,
    classifier_mode=DEFAULT_THREE_ARM_CLASSIFIER,
    skip_thinning=False,
):
    """
    Candidate-first graph detector:
    1) thin skeleton,
    2) find branch-pixel connected components,
    3) classify each component by local arm geometry.
    """
    line = auto_line_mask(image, line_threshold=line_threshold)
    if skip_thinning:
        thin = (line > 0).astype(np.uint8)
    else:
        thin = thin_binary(line)
    nbr = neighbor_count(thin)
    thin_for_components = thin
    snap_stats = {"links_added": 0, "iters_applied": 0}
    if bool(junction_snap_repair):
        thin_for_components, snap_stats = _candidate_junction_snap_repair(
            skeleton_binary=thin,
            snap_iters=junction_snap_iters,
            snap_dist=junction_snap_dist,
            snap_min_dot=junction_snap_min_dot,
            snap_max_existing_fraction=junction_snap_max_existing_fraction,
        )
        if int(snap_stats.get("links_added", 0)) > 0:
            print(
                "Graph candidate snap-repair added "
                f"{int(snap_stats['links_added'])} links over "
                f"{int(snap_stats['iters_applied'])} iterations"
            )
    nbr_for_components = neighbor_count(thin_for_components)
    branch = ((thin_for_components == 1) & (nbr_for_components >= int(max(3, branch_degree)))).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(branch, connectivity=8)
    detections = []

    mode = str(center_mode).strip().lower()
    if mode not in GRAPH_CENTER_MODES:
        mode = DEFAULT_GRAPH_CENTER_MODE

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < int(max(1, min_branch_component_area)):
            continue

        comp = (labels == label_idx).astype(np.uint8)
        centroid_x = float(centroids[label_idx][0])
        centroid_y = float(centroids[label_idx][1])
        center_x = int(round(centroid_x))
        center_y = int(round(centroid_y))
        geom = None
        label = ""
        score = 0.0
        arm_num = 0
        center_source = "centroid"
        extra_chosen = []

        if mode == "best":
            chosen = choose_best_junction_center(
                thin_binary=thin,
                component_mask=comp,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                inner_radius=arm_inner_radius,
                outer_radius=arm_outer_radius,
                min_arm_pixels=min_arm_pixels,
                min_arm_span=min_arm_span,
                t_min_largest_angle=t_min_largest_angle,
                t_max_side_angle=t_max_side_angle,
                ambiguity_margin=ambiguity_margin,
                classifier_mode=classifier_mode,
                ring_bridge_radius=ring_gap_bridge_radius,
                max_centers=max(1, int(max_centers_per_component)),
                separation_px=max(1, int(nms_distance)),
            )
            if isinstance(chosen, list):
                chosen_list = [c for c in chosen if c is not None]
            elif chosen is None:
                chosen_list = []
            else:
                chosen_list = [chosen]
            if chosen_list:
                head = chosen_list[0]
                center_x = int(head["x"])
                center_y = int(head["y"])
                label = str(head["label"])
                score = float(head["score"])
                arm_num = int(head["arm_count"])
                geom = head["geometry"]
                center_source = "best_search"
                extra_chosen = chosen_list[1:]

        if not label and mode in ("centroid", "snap"):
            if mode == "snap":
                sy, sx, ok = nearest_line_point(thin, center_y, center_x, radius=max(2, int(arm_inner_radius)))
                if ok:
                    center_y, center_x = int(sy), int(sx)
                    center_source = "centroid_snap"
            geom = local_geometry_analysis(
                thin_binary=thin,
                y=center_y,
                x=center_x,
                inner_radius=arm_inner_radius,
                outer_radius=arm_outer_radius,
                min_arm_pixels=min_arm_pixels,
                min_arm_span=min_arm_span,
                t_min_largest_angle=t_min_largest_angle,
                t_max_side_angle=t_max_side_angle,
                ambiguity_margin=ambiguity_margin,
                classifier_mode=classifier_mode,
                snap_radius=0,
                ring_bridge_radius=ring_gap_bridge_radius,
            )
            label = str(geom["label"])
            score = float(geom["score"])
            arm_num = int(geom["arm_count"])
            center_y = int(geom["snapped_y"])
            center_x = int(geom["snapped_x"])
            # In sparse/noisy domains, centroid-only geometry can miss true nearby centers.
            # Use a local best-center fallback inside the same branch component before dropping.
            if bool(centroid_best_fallback) and bool(sparse_recovery) and (not label or int(arm_num) < 3):
                fallback = choose_best_junction_center(
                    thin_binary=thin,
                    component_mask=comp,
                    centroid_x=centroid_x,
                    centroid_y=centroid_y,
                    inner_radius=arm_inner_radius,
                    outer_radius=arm_outer_radius,
                    min_arm_pixels=min_arm_pixels,
                    min_arm_span=min_arm_span,
                    t_min_largest_angle=t_min_largest_angle,
                    t_max_side_angle=t_max_side_angle,
                    ambiguity_margin=ambiguity_margin,
                    classifier_mode=classifier_mode,
                    ring_bridge_radius=ring_gap_bridge_radius,
                    max_centers=1,
                    separation_px=max(1, int(nms_distance)),
                )
                if fallback and not isinstance(fallback, list):
                    center_x = int(fallback.get("x", center_x))
                    center_y = int(fallback.get("y", center_y))
                    label = str(fallback.get("label", label))
                    score = float(fallback.get("score", score))
                    arm_num = int(fallback.get("arm_count", arm_num))
                    geom = fallback.get("geometry", geom)
                    center_source = "centroid_best_fallback"

        # Preserve centroid-based behavior as fallback if candidate search/local geometry fails.
        if not label:
            label, score, arm_num = _legacy_classify_component_at_centroid(
                thin_binary=thin,
                component_mask=comp,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                arm_inner_radius=arm_inner_radius,
                arm_outer_radius=arm_outer_radius,
                min_arm_pixels=min_arm_pixels,
                t_min_largest_angle=t_min_largest_angle,
                t_max_side_angle=t_max_side_angle,
                classifier_mode=classifier_mode,
            )
            center_x = int(round(centroid_x))
            center_y = int(round(centroid_y))
            center_source = "legacy_centroid_fallback"

        if not label or int(arm_num) < 3:
            continue

        detections.append(
            {
                "type": label,
                "x": int(center_x),
                "y": int(center_y),
                "score": float(score),
                "spread": -1,
                "rotation": -1,
                "size": -1,
                "bg_gray": -1,
                "source_type": "graph",
                "arm_count": int(arm_num),
                "branch_component_area": int(area),
                "graph_center_mode": str(mode),
                "graph_center_source": str(center_source),
            }
        )
        if geom is not None:
            detections[-1]["local_err_t"] = float(geom.get("err_t", float("inf")))
            detections[-1]["local_err_y"] = float(geom.get("err_y", float("inf")))
            detections[-1]["local_ambiguous"] = bool(geom.get("is_ambiguous", True))
        for extra in extra_chosen:
            extra_geom = extra.get("geometry", {})
            extra_label = str(extra.get("label", ""))
            extra_arm = int(extra.get("arm_count", 0))
            if not extra_label or extra_arm < 3:
                continue
            ed = {
                "type": extra_label,
                "x": int(extra.get("x", 0)),
                "y": int(extra.get("y", 0)),
                "score": float(extra.get("score", 0.0)),
                "spread": -1,
                "rotation": -1,
                "size": -1,
                "bg_gray": -1,
                "source_type": "graph",
                "arm_count": extra_arm,
                "branch_component_area": int(area),
                "graph_center_mode": str(mode),
                "graph_center_source": "best_search_secondary",
            }
            if isinstance(extra_geom, dict):
                ed["local_err_t"] = float(extra_geom.get("err_t", float("inf")))
                ed["local_err_y"] = float(extra_geom.get("err_y", float("inf")))
                ed["local_ambiguous"] = bool(extra_geom.get("is_ambiguous", True))
            detections.append(ed)

    if bool(sparse_recovery):
        support_radius = int(max(0, sparse_recovery_support_radius))
        endpoint_radius = int(max(0, sparse_recovery_endpoint_radius))
        max_component_area = max(1, int(sparse_recovery_max_component_area))
        min_score = float(max(0.0, sparse_recovery_min_score))
        corner_recovery = bool(sparse_corner_recovery)
        corner_max_angle = float(max(0.0, min(180.0, sparse_corner_max_angle)))
        corner_endpoint_radius = int(max(0, sparse_corner_endpoint_radius))
        corner_min_score = None if sparse_corner_min_score is None else float(max(0.0, sparse_corner_min_score))
        if support_radius > 0:
            support_mask = cv2.dilate(branch.astype(np.uint8), _disk_kernel(support_radius), iterations=1)
            weak_from_branch = ((thin == 1) & (nbr >= 2) & (support_mask > 0) & (branch == 0)).astype(np.uint8)
        else:
            weak_from_branch = ((thin == 1) & (nbr >= 2) & (branch == 0)).astype(np.uint8)

        endpoints = ((thin == 1) & (nbr <= 1)).astype(np.uint8)
        weak_from_endpoint = np.zeros_like(weak_from_branch, dtype=np.uint8)
        # Endpoint-guided recovery helps when one arm is broken near a true junction.
        if endpoint_radius > 0:
            endpoint_support = cv2.dilate(endpoints, _disk_kernel(endpoint_radius), iterations=1)
            weak_from_endpoint = ((thin == 1) & (nbr == 2) & (endpoint_support > 0) & (branch == 0)).astype(np.uint8)

        weak_from_corner = np.zeros_like(weak_from_branch, dtype=np.uint8)
        corner_mask = None
        if corner_recovery:
            corner_mask = corner_like_two_neighbor_mask(thin, nbr, max_angle_deg=corner_max_angle)
            if corner_endpoint_radius > 0:
                corner_support = cv2.dilate(endpoints, _disk_kernel(corner_endpoint_radius), iterations=1)
                weak_from_corner = (
                    (thin == 1)
                    & (nbr == 2)
                    & (corner_mask > 0)
                    & (corner_support > 0)
                    & (branch == 0)
                ).astype(np.uint8)
            else:
                weak_from_corner = ((thin == 1) & (nbr == 2) & (corner_mask > 0) & (branch == 0)).astype(np.uint8)

        weak = ((weak_from_branch > 0) | (weak_from_endpoint > 0) | (weak_from_corner > 0)).astype(np.uint8)

        if detections:
            suppress = np.zeros_like(weak, dtype=np.uint8)
            suppress_radius = max(2, int(nms_distance))
            for det in detections:
                cv2.circle(suppress, (int(det["x"]), int(det["y"])), int(suppress_radius), 1, thickness=-1)
            weak = ((weak > 0) & (suppress == 0)).astype(np.uint8)

        weak_labels_n, weak_labels, weak_stats, weak_centroids = cv2.connectedComponentsWithStats(weak, connectivity=8)
        recovery_added = 0
        for widx in range(1, weak_labels_n):
            area = int(weak_stats[widx, cv2.CC_STAT_AREA])
            if area <= 0 or area > max_component_area:
                continue
            comp = (weak_labels == widx).astype(np.uint8)
            comp_has_corner = bool(corner_mask is not None and np.any((comp > 0) & (corner_mask > 0)))
            centroid_x = float(weak_centroids[widx][0])
            centroid_y = float(weak_centroids[widx][1])
            chosen = choose_best_junction_center(
                thin_binary=thin,
                component_mask=comp,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                inner_radius=arm_inner_radius,
                outer_radius=arm_outer_radius,
                min_arm_pixels=min_arm_pixels,
                min_arm_span=min_arm_span,
                t_min_largest_angle=t_min_largest_angle,
                t_max_side_angle=t_max_side_angle,
                ambiguity_margin=ambiguity_margin,
                classifier_mode=classifier_mode,
                ring_bridge_radius=ring_gap_bridge_radius,
                max_centers=1,
                separation_px=max(1, int(nms_distance)),
            )
            if not chosen or isinstance(chosen, list):
                continue
            label = str(chosen.get("label", ""))
            arm_num = int(chosen.get("arm_count", 0))
            score = float(chosen.get("score", 0.0))
            component_min_score = float(min_score)
            if comp_has_corner and corner_min_score is not None:
                component_min_score = float(min(component_min_score, corner_min_score))
            if label not in ("T", "Y", "X") or arm_num < 3 or score < component_min_score:
                continue
            geom = chosen.get("geometry", {}) if isinstance(chosen, dict) else {}
            det = {
                "type": label,
                "x": int(chosen.get("x", 0)),
                "y": int(chosen.get("y", 0)),
                "score": float(score),
                "spread": -1,
                "rotation": -1,
                "size": -1,
                "bg_gray": -1,
                "source_type": "graph",
                "arm_count": int(arm_num),
                "branch_component_area": int(area),
                "graph_center_mode": str(mode),
                "graph_center_source": "sparse_recovery_corner" if comp_has_corner else "sparse_recovery",
            }
            if isinstance(geom, dict):
                det["local_err_t"] = float(geom.get("err_t", float("inf")))
                det["local_err_y"] = float(geom.get("err_y", float("inf")))
                det["local_ambiguous"] = bool(geom.get("is_ambiguous", True))
            detections.append(det)
            recovery_added += 1
        if recovery_added > 0:
            print(f"Graph sparse recovery added {recovery_added} detections")

    detections = non_maximum_suppression(detections, distance_threshold=nms_distance)
    stats_out = {
        "line_pixels": int(line.sum()),
        "thin_pixels": int(thin.sum()),
        "candidate_thin_pixels": int(thin_for_components.sum()),
        "junction_pixels": int(branch.sum()),
        "junction_components": int(max(0, num_labels - 1)),
        "junction_snap_repair": bool(junction_snap_repair),
        "junction_snap_links_added": int(snap_stats.get("links_added", 0)),
        "junction_snap_iters_applied": int(snap_stats.get("iters_applied", 0)),
    }
    return detections, stats_out


def nearest_line_point(thin_binary, y, x, radius=4):
    h, w = thin_binary.shape
    y0 = max(0, int(y) - int(radius))
    y1 = min(h, int(y) + int(radius) + 1)
    x0 = max(0, int(x) - int(radius))
    x1 = min(w, int(x) + int(radius) + 1)
    patch = thin_binary[y0:y1, x0:x1]
    ys, xs = np.where(patch > 0)
    if len(xs) == 0:
        return int(y), int(x), False
    ys = ys + y0
    xs = xs + x0
    d2 = (ys - int(y)) ** 2 + (xs - int(x)) ** 2
    idx = int(np.argmin(d2))
    return int(ys[idx]), int(xs[idx]), True


def local_geometry_classify(
    thin_binary,
    y,
    x,
    inner_radius=DEFAULT_GRAPH_ARM_INNER_RADIUS,
    outer_radius=DEFAULT_GRAPH_ARM_OUTER_RADIUS,
    min_arm_pixels=DEFAULT_GRAPH_MIN_ARM_PIXELS,
    min_arm_span=DEFAULT_GRAPH_MIN_ARM_SPAN,
    t_min_largest_angle=DEFAULT_GRAPH_T_MIN_LARGEST_ANGLE,
    t_max_side_angle=DEFAULT_GRAPH_T_MAX_SIDE_ANGLE,
    ambiguity_margin=6.0,
    classifier_mode=DEFAULT_THREE_ARM_CLASSIFIER,
    return_debug=False,
    snap_radius=None,
    ring_bridge_radius=DEFAULT_GRAPH_RING_GAP_BRIDGE_RADIUS,
):
    geom = local_geometry_analysis(
        thin_binary=thin_binary,
        y=y,
        x=x,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        min_arm_pixels=min_arm_pixels,
        min_arm_span=min_arm_span,
        t_min_largest_angle=t_min_largest_angle,
        t_max_side_angle=t_max_side_angle,
        ambiguity_margin=ambiguity_margin,
        classifier_mode=classifier_mode,
        snap_radius=snap_radius,
        ring_bridge_radius=ring_bridge_radius,
    )
    if return_debug:
        return geom
    return str(geom["label"]), int(geom["arm_count"])


def downgrade_t_to_y_by_local_geometry(
    detections,
    image,
    line_threshold=127,
    inner_radius=DEFAULT_GRAPH_ARM_INNER_RADIUS,
    outer_radius=DEFAULT_GRAPH_ARM_OUTER_RADIUS,
    min_arm_pixels=DEFAULT_GRAPH_MIN_ARM_PIXELS,
    min_arm_span=DEFAULT_GRAPH_MIN_ARM_SPAN,
    t_min_largest_angle=DEFAULT_GRAPH_T_MIN_LARGEST_ANGLE,
    t_max_side_angle=DEFAULT_GRAPH_T_MAX_SIDE_ANGLE,
    ambiguity_margin=6.0,
    classifier_mode=DEFAULT_THREE_ARM_CLASSIFIER,
    reclassify_t_to_y=True,
    reclassify_y_to_t=False,
):
    """
    Correct T-vs-Y confusion by re-checking local arm geometry at each detection.
    """
    if not detections:
        return detections

    line = auto_line_mask(image, line_threshold=line_threshold)
    thin = thin_binary(line)
    out = []
    downgraded = 0
    upgraded = 0
    reclassify_t_to_y = bool(reclassify_t_to_y)
    reclassify_y_to_t = bool(reclassify_y_to_t)
    for det in detections:
        d = dict(det)
        dtype = d.get("type")
        should_check = (reclassify_t_to_y and dtype == "T") or (reclassify_y_to_t and dtype == "Y")
        if should_check:
            geom = local_geometry_classify(
                thin_binary=thin,
                y=int(d["y"]),
                x=int(d["x"]),
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                min_arm_pixels=min_arm_pixels,
                min_arm_span=min_arm_span,
                t_min_largest_angle=t_min_largest_angle,
                t_max_side_angle=t_max_side_angle,
                ambiguity_margin=ambiguity_margin,
                classifier_mode=classifier_mode,
                return_debug=True,
            )
            local_label = str(geom["label"])
            arm_num = int(geom["arm_count"])
            d["local_label"] = local_label
            d["local_arm_count"] = int(arm_num)
            d["local_err_t"] = float(geom.get("err_t", float("inf")))
            d["local_err_y"] = float(geom.get("err_y", float("inf")))
            d["local_ambiguous"] = bool(geom.get("is_ambiguous", True))
            if geom.get("snapped_ok", False):
                d["x"] = int(geom["snapped_x"])
                d["y"] = int(geom["snapped_y"])
            if local_label == "Y":
                if reclassify_t_to_y and dtype == "T":
                    d["type"] = "Y"
                    d["post_adjustment"] = "T_to_Y_local_geometry"
                    downgraded += 1
            elif local_label == "T":
                if reclassify_y_to_t and dtype == "Y":
                    d["type"] = "T"
                    d["post_adjustment"] = "Y_to_T_local_geometry"
                    upgraded += 1
        out.append(d)

    if downgraded > 0:
        print(f"Local geometry downgraded {downgraded} detections: T -> Y")
    if upgraded > 0:
        print(f"Local geometry upgraded {upgraded} detections: Y -> T")
    return out


def reclassify_t_y_by_multiradius_vote(
    detections,
    image,
    enabled=DEFAULT_TY_MULTIRADIUS_VOTE,
    line_threshold=127,
    inner_radius=DEFAULT_GRAPH_ARM_INNER_RADIUS,
    outer_radii=DEFAULT_TY_VOTE_OUTER_RADII,
    min_votes=DEFAULT_TY_VOTE_MIN_VOTES,
    vote_margin=DEFAULT_TY_VOTE_MARGIN,
    require_non_ambiguous=DEFAULT_TY_VOTE_REQUIRE_NON_AMBIG,
    min_arm_pixels=DEFAULT_GRAPH_MIN_ARM_PIXELS,
    min_arm_span=DEFAULT_GRAPH_MIN_ARM_SPAN,
    t_min_largest_angle=DEFAULT_GRAPH_T_MIN_LARGEST_ANGLE,
    t_max_side_angle=DEFAULT_GRAPH_T_MAX_SIDE_ANGLE,
    ambiguity_margin=6.0,
    classifier_mode=DEFAULT_THREE_ARM_CLASSIFIER,
):
    if not detections or not bool(enabled):
        return detections

    radii = []
    for r in tuple(outer_radii or ()):
        rv = int(max(int(inner_radius) + 1, int(r)))
        if rv not in radii:
            radii.append(rv)
    if not radii:
        return detections

    min_votes = int(max(1, min_votes))
    vote_margin = int(max(0, vote_margin))
    require_non_ambiguous = bool(require_non_ambiguous)

    line = auto_line_mask(image, line_threshold=line_threshold)
    thin = thin_binary(line)
    out = []
    relabeled = 0

    for det in detections:
        d = dict(det)
        dtype = str(d.get("type", ""))
        if dtype not in ("T", "Y"):
            out.append(d)
            continue

        vote_t = 0
        vote_y = 0
        for radius in radii:
            geom = local_geometry_classify(
                thin_binary=thin,
                y=int(d["y"]),
                x=int(d["x"]),
                inner_radius=inner_radius,
                outer_radius=int(radius),
                min_arm_pixels=min_arm_pixels,
                min_arm_span=min_arm_span,
                t_min_largest_angle=t_min_largest_angle,
                t_max_side_angle=t_max_side_angle,
                ambiguity_margin=ambiguity_margin,
                classifier_mode=classifier_mode,
                return_debug=True,
            )
            label = str(geom.get("label", ""))
            if label not in ("T", "Y"):
                continue
            if require_non_ambiguous and bool(geom.get("is_ambiguous", True)):
                continue
            if label == "T":
                vote_t += 1
            else:
                vote_y += 1

        if max(vote_t, vote_y) >= min_votes and abs(vote_t - vote_y) >= vote_margin:
            voted_label = "T" if vote_t > vote_y else "Y"
            if voted_label != dtype:
                d["type"] = voted_label
                d["post_adjustment"] = "TY_multiradius_vote"
                d["ty_vote_t"] = int(vote_t)
                d["ty_vote_y"] = int(vote_y)
                relabeled += 1
        out.append(d)

    if relabeled > 0:
        print(f"Multi-radius T/Y vote relabeled {relabeled} detections")
    return out


def _safe_cv(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    mean = float(np.mean(arr))
    if not np.isfinite(mean) or mean <= 0.0:
        return float("nan")
    return float(np.std(arr) / mean)


def _safe_ratio(num, den):
    denf = float(den)
    if not np.isfinite(denf) or abs(denf) <= 1e-9:
        return float("nan")
    return float(float(num) / denf)


def compute_three_arm_feature_vector(
    thin_binary,
    y,
    x,
    inner_radius=DEFAULT_TY_FEATURE_RULE_INNER_RADIUS,
    outer_radius=DEFAULT_GRAPH_ARM_OUTER_RADIUS,
    min_arm_pixels=DEFAULT_TY_FEATURE_RULE_MIN_ARM_PIXELS,
    min_arm_span=DEFAULT_TY_FEATURE_RULE_MIN_ARM_SPAN,
    t_min_largest_angle=DEFAULT_TY_FEATURE_RULE_T_MIN_LARGEST_ANGLE,
    t_max_side_angle=DEFAULT_TY_FEATURE_RULE_T_MAX_SIDE_ANGLE,
    ambiguity_margin=DEFAULT_TY_FEATURE_RULE_AMBIGUITY_MARGIN,
    classifier_mode=DEFAULT_TY_FEATURE_RULE_CLASSIFIER,
):
    geom = local_geometry_classify(
        thin_binary=thin_binary,
        y=int(y),
        x=int(x),
        inner_radius=int(inner_radius),
        outer_radius=int(outer_radius),
        min_arm_pixels=int(min_arm_pixels),
        min_arm_span=float(min_arm_span),
        t_min_largest_angle=float(t_min_largest_angle),
        t_max_side_angle=float(t_max_side_angle),
        ambiguity_margin=float(ambiguity_margin),
        classifier_mode=str(classifier_mode),
        return_debug=True,
    )

    features = {
        "arm_count": int(geom.get("arm_count", 0)),
        "label": str(geom.get("label", "")),
        "err_t": float(geom.get("err_t", float("nan"))),
        "err_y": float(geom.get("err_y", float("nan"))),
        "is_ambiguous": bool(geom.get("is_ambiguous", True)),
        "t_guardrail": float(1.0 if bool(geom.get("t_guardrail", False)) else 0.0),
        "a0": float("nan"),
        "a1": float("nan"),
        "a2": float("nan"),
        "opp_angle": float("nan"),
        "opp_gap": float("nan"),
        "areas": [float("nan"), float("nan"), float("nan")],
        "spans": [float("nan"), float("nan"), float("nan")],
        "area_cv": float("nan"),
        "span_cv": float("nan"),
        "area_ratio": float("nan"),
        "span_ratio": float("nan"),
        "third_area_frac": float("nan"),
        "third_span_frac": float("nan"),
        "opp_balance": float("nan"),
        "opp_area_ratio": float("nan"),
        "opp_span_ratio": float("nan"),
    }

    if int(features["arm_count"]) != 3:
        return features

    ring = geom.get("ring", {}) or {}
    valid_arms = list(ring.get("valid_arms", []))
    vectors = list(ring.get("vectors", []))
    if len(valid_arms) != 3 or len(vectors) != 3:
        return features

    angles = sorted(float(v) for v in _angles_for_vectors(vectors))
    if len(angles) != 3:
        return features

    features["a0"] = float(angles[0])
    features["a1"] = float(angles[1])
    features["a2"] = float(angles[2])

    areas = [float(a.get("area", 0.0)) for a in valid_arms]
    spans = [float(a.get("radial_span", 0.0)) for a in valid_arms]
    features["areas"] = [float(v) for v in areas]
    features["spans"] = [float(v) for v in spans]

    area_min = float(np.min(areas)) if areas else float("nan")
    area_max = float(np.max(areas)) if areas else float("nan")
    span_min = float(np.min(spans)) if spans else float("nan")
    span_max = float(np.max(spans)) if spans else float("nan")
    features["area_cv"] = _safe_cv(areas)
    features["span_cv"] = _safe_cv(spans)
    features["area_ratio"] = _safe_ratio(area_max, area_min)
    features["span_ratio"] = _safe_ratio(span_max, span_min)

    pair_angles = []
    for i in range(3):
        for j in range(i + 1, 3):
            dot = float(np.clip(np.dot(vectors[i], vectors[j]), -1.0, 1.0))
            pair_angles.append((float(np.degrees(np.arccos(dot))), i, j))
    pair_angles.sort(key=lambda t: (t[0], -t[1], -t[2]), reverse=True)
    if not pair_angles:
        return features
    opp_angle, i_opp, j_opp = pair_angles[0]
    k_third = [k for k in (0, 1, 2) if k not in (i_opp, j_opp)][0]

    features["opp_angle"] = float(opp_angle)
    features["opp_gap"] = float(180.0 - float(opp_angle))

    area_i = float(areas[i_opp])
    area_j = float(areas[j_opp])
    area_k = float(areas[k_third])
    span_i = float(spans[i_opp])
    span_j = float(spans[j_opp])
    span_k = float(spans[k_third])

    area_sum = float(area_i + area_j + area_k)
    span_sum = float(span_i + span_j + span_k)
    opp_sum_area = float(area_i + area_j)
    opp_sum_span = float(span_i + span_j)
    features["third_area_frac"] = _safe_ratio(area_k, area_sum)
    features["third_span_frac"] = _safe_ratio(span_k, span_sum)
    features["opp_balance"] = _safe_ratio(abs(area_i - area_j), opp_sum_area)
    features["opp_area_ratio"] = _safe_ratio(opp_sum_area, area_k)
    features["opp_span_ratio"] = _safe_ratio(opp_sum_span, span_k)
    return features


def _ty_feature_rule_decision_full(f):
    def le(name, thr):
        val = float(f.get(name, float("nan")))
        if not np.isfinite(val):
            raise ValueError(name)
        return val <= float(thr)

    try:
        if le("opp_angle_29", 141.97):
            if le("third_area_frac_29", 0.32):
                if le("area_cv_29", 0.39):
                    if le("opp_gap_45", 52.64):
                        if le("area_ratio_29", 2.12):
                            if le("span_cv_45", 0.02):
                                if le("opp_balance_45", 0.03):
                                    return "Y"
                                else:
                                    return "T"
                            else:
                                return "Y"
                        else:
                            if le("area_ratio_29", 2.34):
                                return "T"
                            else:
                                return "Y"
                    else:
                        return "Y"
                else:
                    if le("opp_gap_29", 46.70):
                        if le("a0_29", 93.83):
                            return "Y"
                        else:
                            return "T"
                    else:
                        return "Y"
            else:
                if le("opp_gap_61", 44.43):
                    if le("a1_29", 111.49):
                        return "T"
                    else:
                        return "Y"
                else:
                    if le("a1_45", 121.15):
                        if le("opp_angle_61", 122.72):
                            return "T"
                        else:
                            return "Y"
                    else:
                        return "T"
        else:
            if le("a1_45", 108.35):
                if le("a2_29", 143.12):
                    return "Y"
                else:
                    if le("a0_29", 104.75):
                        return "T"
                    else:
                        if le("t_guardrail_61", 0.50):
                            return "Y"
                        else:
                            return "T"
            else:
                if le("a1_45", 113.90):
                    if le("a1_29", 114.75):
                        if le("a0_61", 87.79):
                            return "T"
                        else:
                            if le("opp_balance_45", 0.17):
                                if le("opp_balance_61", 0.06):
                                    return "T"
                                else:
                                    return "Y"
                            else:
                                if le("opp_gap_29", 21.91):
                                    return "Y"
                                else:
                                    return "T"
                    else:
                        if le("span_cv_45", 0.02):
                            return "Y"
                        else:
                            return "T"
                else:
                    if le("third_area_frac_29", 0.31):
                        if le("span_cv_45", 0.01):
                            if le("a0_45", 91.97):
                                if le("area_cv_61", 0.36):
                                    return "Y"
                                else:
                                    return "T"
                            else:
                                if le("a0_61", 106.81):
                                    return "T"
                                else:
                                    return "Y"
                        else:
                            return "T"
                    else:
                        if le("err_y_45", 109.91):
                            return "T"
                        else:
                            return "Y"
    except ValueError:
        return None


def _ty_feature_rule_decision_lite(f):
    def le(name, thr):
        val = float(f.get(name, float("nan")))
        if not np.isfinite(val):
            raise ValueError(name)
        return val <= float(thr)

    try:
        if le("opp_angle_29", 141.97):
            if le("third_area_frac_29", 0.32):
                if le("area_cv_29", 0.39):
                    if le("opp_gap_45", 52.64):
                        return "T"
                    else:
                        return "Y"
                else:
                    return "Y"
            else:
                if le("opp_gap_61", 44.43):
                    return "Y"
                else:
                    if le("opp_span_ratio_45", 1.98):
                        return "Y"
                    else:
                        return "T"
        else:
            if le("a1_45", 108.35):
                return "T"
            else:
                if le("a1_45", 113.90):
                    if le("a1_29", 114.75):
                        return "T"
                    else:
                        return "Y"
                else:
                    if le("third_area_frac_29", 0.31):
                        return "T"
                    else:
                        return "T"
    except ValueError:
        return None


def reclassify_t_y_by_feature_rules(
    detections,
    image,
    enabled=DEFAULT_TY_FEATURE_RULES,
    mode=DEFAULT_TY_FEATURE_RULE_MODE,
    line_threshold=127,
    outer_radii=DEFAULT_TY_FEATURE_RULE_OUTER_RADII,
    inner_radius=DEFAULT_TY_FEATURE_RULE_INNER_RADIUS,
    min_arm_pixels=DEFAULT_TY_FEATURE_RULE_MIN_ARM_PIXELS,
    min_arm_span=DEFAULT_TY_FEATURE_RULE_MIN_ARM_SPAN,
    t_min_largest_angle=DEFAULT_TY_FEATURE_RULE_T_MIN_LARGEST_ANGLE,
    t_max_side_angle=DEFAULT_TY_FEATURE_RULE_T_MAX_SIDE_ANGLE,
    ambiguity_margin=DEFAULT_TY_FEATURE_RULE_AMBIGUITY_MARGIN,
    classifier_mode=DEFAULT_TY_FEATURE_RULE_CLASSIFIER,
):
    if not detections or not bool(enabled):
        return detections

    selected_mode = str(mode).strip().lower()
    if selected_mode not in TY_FEATURE_RULE_MODES:
        selected_mode = DEFAULT_TY_FEATURE_RULE_MODE

    radii = []
    for r in tuple(outer_radii or ()):
        rv = int(max(int(inner_radius) + 1, int(r)))
        if rv not in radii:
            radii.append(rv)
    if not radii:
        return detections

    line = auto_line_mask(image, line_threshold=line_threshold)
    thin = thin_binary(line)
    out = []
    relabeled = 0

    for det in detections:
        d = dict(det)
        dtype = str(d.get("type", ""))
        if dtype not in ("T", "Y"):
            out.append(d)
            continue

        flat = {}
        for r in radii:
            fv = compute_three_arm_feature_vector(
                thin_binary=thin,
                y=int(d["y"]),
                x=int(d["x"]),
                inner_radius=int(inner_radius),
                outer_radius=int(r),
                min_arm_pixels=int(min_arm_pixels),
                min_arm_span=float(min_arm_span),
                t_min_largest_angle=float(t_min_largest_angle),
                t_max_side_angle=float(t_max_side_angle),
                ambiguity_margin=float(ambiguity_margin),
                classifier_mode=str(classifier_mode),
            )
            d[f"ty_rule_arm_count_{r}"] = int(fv.get("arm_count", 0))
            d[f"ty_rule_label_{r}"] = str(fv.get("label", ""))
            d[f"ty_rule_is_ambiguous_{r}"] = bool(fv.get("is_ambiguous", True))
            for key in (
                "err_t",
                "err_y",
                "t_guardrail",
                "a0",
                "a1",
                "a2",
                "opp_angle",
                "opp_gap",
                "area_cv",
                "span_cv",
                "area_ratio",
                "span_ratio",
                "third_area_frac",
                "third_span_frac",
                "opp_balance",
                "opp_area_ratio",
                "opp_span_ratio",
            ):
                val = float(fv.get(key, float("nan")))
                d[f"ty_rule_{key}_{r}"] = val
                flat[f"{key}_{r}"] = val

        d["ty_feature_rule_mode"] = str(selected_mode)
        decision = _ty_feature_rule_decision_full(flat) if selected_mode == "full" else _ty_feature_rule_decision_lite(flat)
        d["ty_rule_decision"] = str(decision) if decision in ("T", "Y") else "fallback_keep"

        if decision in ("T", "Y") and decision != dtype:
            d["type"] = str(decision)
            d["post_adjustment"] = "TY_feature_rules"
            d["ty_feature_rule_version"] = TY_FEATURE_RULE_VERSION
            relabeled += 1
        out.append(d)

    if relabeled > 0:
        print(f"T/Y feature-rule relabeled {relabeled} detections (mode={selected_mode})")
    return out


def _tyx_structural_feature_value(det, feature_name):
    sent = -999.0
    if feature_name == "pred_code":
        dtype = str(det.get("type", ""))
        if dtype == "T":
            return 0.0
        if dtype == "Y":
            return 1.0
        if dtype == "X":
            return 2.0
        return sent
    try:
        val = float(det.get(feature_name, sent))
        if not np.isfinite(val):
            return sent
        return float(val)
    except Exception:
        return sent


def _predict_tyx_structural_rule(det, mode=DEFAULT_TYX_STRUCTURAL_RULE_MODE):
    selected_mode = str(mode).strip().lower()
    if selected_mode not in TYX_STRUCTURAL_RULE_MODES:
        selected_mode = DEFAULT_TYX_STRUCTURAL_RULE_MODE

    bundle = TYX_STRUCTURAL_TREE_BUNDLES.get(selected_mode, {})
    feature_names = list(bundle.get("feature_names", []))
    children_left = list(bundle.get("children_left", []))
    children_right = list(bundle.get("children_right", []))
    feature = list(bundle.get("feature", []))
    threshold = list(bundle.get("threshold", []))
    value = list(bundle.get("value", []))
    classes = list(bundle.get("classes", ["T", "Y", "X"]))

    if not feature_names or not children_left:
        return str(det.get("type", "")), -1, selected_mode, 0.0, 0.0, 0.0, 0.0

    node = 0
    max_steps = max(1, len(children_left) * 2)
    steps = 0

    while 0 <= node < len(children_left) and steps < max_steps:
        steps += 1
        left = int(children_left[node])
        right = int(children_right[node])
        if left < 0 and right < 0:
            break

        fi = int(feature[node])
        thr = float(threshold[node])
        if fi < 0 or fi >= len(feature_names):
            break
        f_name = str(feature_names[fi])
        f_val = _tyx_structural_feature_value(det, f_name)
        node = left if f_val <= thr else right

    leaf = int(node) if 0 <= node < len(value) else -1
    if leaf < 0:
        return str(det.get("type", "")), -1, selected_mode, 0.0, 0.0, 0.0, 0.0

    counts = value[leaf]
    if not isinstance(counts, (list, tuple)) or len(counts) == 0:
        return str(det.get("type", "")), leaf, selected_mode, 0.0, 0.0, 0.0, 0.0
    arr = np.array(counts, dtype=np.float32)
    argmax = int(np.argmax(arr))
    if argmax < 0 or argmax >= len(classes):
        return str(det.get("type", "")), leaf, selected_mode, 0.0, 0.0, 0.0, 0.0
    decision = str(classes[argmax])
    if decision not in ("T", "Y", "X"):
        decision = str(det.get("type", ""))
    sorted_vals = np.sort(arr)[::-1]
    top1 = float(sorted_vals[0]) if sorted_vals.size > 0 and np.isfinite(sorted_vals[0]) else 0.0
    top2 = float(sorted_vals[1]) if sorted_vals.size > 1 and np.isfinite(sorted_vals[1]) else 0.0
    total = float(np.sum(arr)) if np.isfinite(np.sum(arr)) else 0.0
    if total <= 0.0:
        margin = 0.0
    else:
        margin = float((top1 - top2) / max(1e-6, total))
    return decision, leaf, selected_mode, margin, top1, top2, total


def _tyx_structural_multiradius_evidence(det):
    arm_counts = []
    label_votes = defaultdict(int)
    for r in (29, 45, 61):
        key_arm = f"ty_rule_arm_count_{r}"
        arm_val = _tyx_structural_feature_value(det, key_arm)
        if arm_val > -998.0:
            arm_counts.append(int(round(arm_val)))
        key_label = f"ty_rule_label_{r}"
        lbl = str(det.get(key_label, "")).strip().upper()
        if lbl in ("T", "Y", "X"):
            label_votes[lbl] += 1

    valid_arm_counts = [v for v in arm_counts if v >= 0]
    support_eq3 = int(sum(1 for v in valid_arm_counts if v == 3))
    support_ge4 = int(sum(1 for v in valid_arm_counts if v >= 4))
    support_ge3 = int(sum(1 for v in valid_arm_counts if v >= 3))
    arm_hist = defaultdict(int)
    for v in valid_arm_counts:
        arm_hist[int(v)] += 1
    stable_arm_votes = int(max(arm_hist.values())) if arm_hist else 0

    majority_label = ""
    majority_votes = 0
    if label_votes:
        majority_label = max(label_votes.items(), key=lambda kv: (int(kv[1]), str(kv[0])))[0]
        majority_votes = int(label_votes.get(majority_label, 0))

    local_ambiguous = bool(det.get("local_ambiguous", True))
    return {
        "valid_radii": int(len(valid_arm_counts)),
        "support_eq3": support_eq3,
        "support_ge4": support_ge4,
        "support_ge3": support_ge3,
        "stable_arm_votes": stable_arm_votes,
        "majority_label": str(majority_label),
        "majority_votes": majority_votes,
        "local_ambiguous": local_ambiguous,
    }


def _tyx_vote_by_three_arm_error(det):
    t_votes = 0
    y_votes = 0
    for r in (29, 45, 61):
        arm_count = _tyx_structural_feature_value(det, f"ty_rule_arm_count_{r}")
        if arm_count <= -998.0 or int(round(arm_count)) != 3:
            continue
        err_t = _tyx_structural_feature_value(det, f"ty_rule_err_t_{r}")
        err_y = _tyx_structural_feature_value(det, f"ty_rule_err_y_{r}")
        if err_t <= -998.0 or err_y <= -998.0:
            continue
        if float(err_t) + 2.0 < float(err_y):
            t_votes += 1
        elif float(err_y) + 2.0 < float(err_t):
            y_votes += 1
    if t_votes > y_votes:
        return "T", int(t_votes), int(y_votes)
    if y_votes > t_votes:
        return "Y", int(t_votes), int(y_votes)
    return "", int(t_votes), int(y_votes)


def _tyx_weak_x_override(det, branch_ratio=None, branch_ratio_threshold=DEFAULT_GENERALIZATION_BRANCH_RATIO_THRESHOLD):
    old_type = str(det.get("type", ""))
    if old_type != "X":
        return "", "not_x"
    evidence = _tyx_structural_multiradius_evidence(det)
    if evidence["support_ge4"] >= 2:
        return "", "strong_x_support"

    low_branch_domain = False
    if branch_ratio is not None:
        try:
            br = float(branch_ratio)
            if np.isfinite(br) and br < float(branch_ratio_threshold):
                low_branch_domain = True
        except Exception:
            low_branch_domain = False

    if evidence["support_eq3"] < 2 and not low_branch_domain:
        return "", "insufficient_3arm_support"

    maj_label = str(evidence.get("majority_label", ""))
    maj_votes = int(evidence.get("majority_votes", 0))
    if maj_label in ("T", "Y") and maj_votes >= 2:
        return maj_label, "weak_x_majority_vote"

    vote_label, t_votes, y_votes = _tyx_vote_by_three_arm_error(det)
    if vote_label in ("T", "Y"):
        return vote_label, f"weak_x_error_vote_t{t_votes}_y{y_votes}"

    return "", "no_override_signal"


def _tyx_structural_gate_decision(det, old_type, decision, margin, mode, min_margin):
    if old_type not in ("T", "Y", "X"):
        return False, "old_type_not_tyx", {}
    if decision not in ("T", "Y", "X"):
        return False, "decision_not_tyx", {}
    if decision == old_type:
        return True, "no_change", _tyx_structural_multiradius_evidence(det)

    evidence = _tyx_structural_multiradius_evidence(det)
    if evidence["valid_radii"] <= 0:
        return False, "no_multiradius_features", evidence

    base_margin = float(TYX_STRUCTURAL_MIN_MARGIN_BY_MODE.get(str(mode), DEFAULT_TYX_STRUCTURAL_MIN_MARGIN))
    requested_margin = float(max(0.0, min_margin))
    gate_margin = float(max(base_margin, requested_margin))
    if float(margin) < gate_margin:
        return False, f"low_margin<{gate_margin:.2f}", evidence

    if evidence["majority_votes"] < 2 and not (decision == "X" and evidence["support_ge4"] >= 2):
        return False, "weak_multiradius_vote", evidence
    if evidence["majority_votes"] >= 2 and evidence["majority_label"] and evidence["majority_label"] != decision:
        return False, "vote_disagrees_with_decision", evidence

    if decision in ("T", "Y") and evidence["support_eq3"] < 2:
        return False, "needs_three_arm_stability", evidence
    if decision == "X" and evidence["support_ge4"] < 2:
        return False, "needs_four_arm_stability", evidence

    if old_type == "X" and decision in ("T", "Y"):
        if evidence["support_ge4"] > 1:
            return False, "x_to_ty_conflicted_by_4arm", evidence
    if old_type in ("T", "Y") and decision == "X":
        if evidence["support_eq3"] > 1 and evidence["support_ge4"] < 3:
            return False, "ty_to_x_conflicted_by_3arm", evidence

    if evidence["local_ambiguous"] and float(margin) < (gate_margin + 0.15):
        return False, "local_ambiguous_low_margin", evidence

    return True, "pass", evidence


def reclassify_tyx_by_structural_rules(
    detections,
    enabled=False,
    mode=DEFAULT_TYX_STRUCTURAL_RULE_MODE,
    ood_gate=DEFAULT_TYX_STRUCTURAL_OOD_GATE,
    min_margin=DEFAULT_TYX_STRUCTURAL_MIN_MARGIN,
    domain_branch_ratio=None,
    weak_x_override=True,
    weak_x_branch_ratio_threshold=DEFAULT_GENERALIZATION_BRANCH_RATIO_THRESHOLD,
):
    if not detections or not bool(enabled):
        return detections

    selected_mode = str(mode).strip().lower()
    if selected_mode not in TYX_STRUCTURAL_RULE_MODES:
        selected_mode = DEFAULT_TYX_STRUCTURAL_RULE_MODE

    out = []
    relabeled = 0
    gate_blocked = 0
    for det in detections:
        d = dict(det)
        decision, leaf, used_mode, margin, top1, top2, total = _predict_tyx_structural_rule(d, mode=selected_mode)
        d["tyx_structural_rule_mode"] = str(used_mode)
        d["tyx_structural_rule_leaf"] = int(leaf)
        d["tyx_structural_rule_decision"] = str(decision)
        d["tyx_structural_rule_margin"] = float(margin)
        d["tyx_structural_rule_top1"] = float(top1)
        d["tyx_structural_rule_top2"] = float(top2)
        d["tyx_structural_rule_leaf_total"] = float(total)
        if domain_branch_ratio is not None:
            try:
                br = float(domain_branch_ratio)
                if np.isfinite(br):
                    d["global_branch_ratio"] = float(br)
            except Exception:
                pass

        old_type = str(d.get("type", ""))
        if bool(weak_x_override) and old_type == "X" and decision == "X":
            x_override, x_reason = _tyx_weak_x_override(
                d,
                branch_ratio=domain_branch_ratio,
                branch_ratio_threshold=weak_x_branch_ratio_threshold,
            )
            if x_override in ("T", "Y"):
                decision = str(x_override)
                d["tyx_structural_rule_decision"] = str(decision)
                d["tyx_structural_x_override"] = str(x_reason)

        gate_ok = True
        gate_reason = "disabled"
        evidence = {}
        if bool(ood_gate):
            gate_ok, gate_reason, evidence = _tyx_structural_gate_decision(
                d,
                old_type=old_type,
                decision=decision,
                margin=float(margin),
                mode=selected_mode,
                min_margin=float(min_margin),
            )
            d["tyx_structural_gate_pass"] = bool(gate_ok)
            d["tyx_structural_gate_reason"] = str(gate_reason)
            if evidence:
                d["tyx_structural_vote_majority"] = str(evidence.get("majority_label", ""))
                d["tyx_structural_vote_majority_count"] = int(evidence.get("majority_votes", 0))
                d["tyx_structural_support_eq3"] = int(evidence.get("support_eq3", 0))
                d["tyx_structural_support_ge4"] = int(evidence.get("support_ge4", 0))
                d["tyx_structural_valid_radii"] = int(evidence.get("valid_radii", 0))
        else:
            d["tyx_structural_gate_pass"] = True
            d["tyx_structural_gate_reason"] = "disabled"

        if decision in ("T", "Y", "X") and decision != old_type and gate_ok:
            d["type"] = str(decision)
            d["post_adjustment"] = "TYX_structural_rules"
            d["tyx_structural_rule_version"] = TYX_STRUCTURAL_RULE_VERSION
            relabeled += 1
        elif decision in ("T", "Y", "X") and decision != old_type and not gate_ok:
            gate_blocked += 1
        out.append(d)

    if relabeled > 0:
        print(f"TYX structural-rule relabeled {relabeled} detections (mode={selected_mode})")
    if gate_blocked > 0 and bool(ood_gate):
        print(f"TYX structural-rule OOD gate blocked {gate_blocked} relabel attempts")
    return out


def reclassify_x_by_multiradius_consistency(
    detections,
    image,
    enabled=DEFAULT_X_CONSISTENCY_RECLASSIFY,
    line_threshold=127,
    inner_radius=DEFAULT_GRAPH_ARM_INNER_RADIUS,
    outer_radii=DEFAULT_X_CONSISTENCY_OUTER_RADII,
    min_arm_pixels=DEFAULT_GRAPH_MIN_ARM_PIXELS,
    min_arm_span=DEFAULT_GRAPH_MIN_ARM_SPAN,
    t_min_largest_angle=DEFAULT_GRAPH_T_MIN_LARGEST_ANGLE,
    t_max_side_angle=DEFAULT_GRAPH_T_MAX_SIDE_ANGLE,
    ambiguity_margin=6.0,
    classifier_mode=DEFAULT_THREE_ARM_CLASSIFIER,
    ring_bridge_radius=DEFAULT_GRAPH_RING_GAP_BRIDGE_RADIUS,
    max_fourarm_votes=DEFAULT_X_CONSISTENCY_MAX_FOURARM,
    min_threearm_votes=DEFAULT_X_CONSISTENCY_MIN_THREEARM_VOTES,
):
    if not detections or not bool(enabled):
        return detections
    if image is None:
        return detections

    line = auto_line_mask(image, line_threshold=int(np.clip(line_threshold, 1, 254)))
    thin = thin_binary(line)

    radii = [int(max(inner_radius + 1, int(r))) for r in tuple(outer_radii) if int(r) > int(inner_radius)]
    if not radii:
        return detections

    out = []
    relabeled = 0
    for det in detections:
        d = dict(det)
        if str(d.get("type", "")) != "X":
            out.append(d)
            continue

        votes = defaultdict(int)
        support_four = 0
        support_three = 0
        for r in radii:
            geom = local_geometry_classify(
                thin_binary=thin,
                y=int(d.get("y", 0)),
                x=int(d.get("x", 0)),
                inner_radius=int(inner_radius),
                outer_radius=int(r),
                min_arm_pixels=min_arm_pixels,
                min_arm_span=min_arm_span,
                t_min_largest_angle=t_min_largest_angle,
                t_max_side_angle=t_max_side_angle,
                ambiguity_margin=ambiguity_margin,
                classifier_mode=classifier_mode,
                return_debug=True,
                snap_radius=max(2, int(inner_radius)),
                ring_bridge_radius=ring_bridge_radius,
            )
            arm_count = int(geom.get("arm_count", 0))
            lbl = str(geom.get("label", ""))
            if arm_count >= 4:
                support_four += 1
            if arm_count == 3 and lbl in ("T", "Y"):
                support_three += 1
                votes[lbl] += 1

        d["x_consistency_support_four"] = int(support_four)
        d["x_consistency_support_three"] = int(support_three)
        majority_label = max(votes.items(), key=lambda kv: (int(kv[1]), str(kv[0])))[0] if votes else ""
        majority_votes = int(votes.get(majority_label, 0)) if majority_label else 0
        d["x_consistency_majority_label"] = str(majority_label)
        d["x_consistency_majority_votes"] = int(majority_votes)

        if int(support_four) <= int(max_fourarm_votes) and int(majority_votes) >= int(min_threearm_votes):
            d["type"] = str(majority_label)
            d["post_adjustment"] = "X_multiradius_consistency"
            d["x_consistency_rule_version"] = "v1"
            relabeled += 1
        out.append(d)

    if relabeled > 0:
        print(f"X multiradius-consistency relabeled {relabeled} detections")
    return out


def _finite_float_or_none(value):
    try:
        val = float(value)
        if not np.isfinite(val):
            return None
        return float(val)
    except Exception:
        return None


def reclassify_low_branch_sparse_bias(
    detections,
    enabled=False,
    y_advantage_margin=2.0,
    suppress_low_score_legacy_fallback=True,
    legacy_fallback_score_threshold=0.01,
    domain_branch_ratio=None,
    moderate_branch_ratio_threshold=DEFAULT_GENERALIZATION_MODERATE_BRANCH_RATIO,
):
    if not detections or not bool(enabled):
        return detections

    out = []
    relabeled = 0
    suppressed = 0
    branch_ratio = _finite_float_or_none(domain_branch_ratio)
    ultra_low_branch = (
        branch_ratio is not None
        and float(branch_ratio) < float(max(0.0, moderate_branch_ratio_threshold))
    )
    moderate_low_branch = (
        branch_ratio is not None
        and float(branch_ratio) >= float(max(0.0, moderate_branch_ratio_threshold))
    )
    for det in detections:
        d = dict(det)
        old_type = str(d.get("type", ""))
        source_type = str(d.get("source_type", "")).strip().lower()
        is_graphish = source_type in ("graph", "hybrid")
        center_source = str(d.get("graph_center_source", ""))
        score = _finite_float_or_none(d.get("score"))
        decision = old_type
        reason = "keep"

        # Remove very weak fallback centers that are typically tiny stub hallucinations.
        if bool(suppress_low_score_legacy_fallback) and source_type == "graph":
            if center_source == "legacy_centroid_fallback" and score is not None:
                if float(score) < float(max(0.0, legacy_fallback_score_threshold)):
                    suppressed += 1
                    continue

        if is_graphish and old_type == "X":
            support_four = int(round(_finite_float_or_none(d.get("x_consistency_support_four")) or 0.0))
            support_three = int(round(_finite_float_or_none(d.get("x_consistency_support_three")) or 0.0))
            majority_label = str(d.get("x_consistency_majority_label", "")).strip().upper()
            majority_votes = int(round(_finite_float_or_none(d.get("x_consistency_majority_votes")) or 0.0))
            local_ambiguous = bool(d.get("local_ambiguous", True))
            arm_count = int(round(_finite_float_or_none(d.get("arm_count")) or 0.0))
            branch_component_area = int(round(_finite_float_or_none(d.get("branch_component_area")) or 0.0))

            if moderate_low_branch and support_four <= 3 and support_three <= 1:
                # In moderately sparse scenes, weak 4-arm corner recoveries can overproduce
                # phantom Y detections around noisy bends. Suppress only the noisiest subset.
                if (
                    center_source == "sparse_recovery_corner"
                    and arm_count >= 4
                    and branch_component_area >= int(DEFAULT_GENERALIZATION_MODERATE_WEAK_X_CORNER_SUPPRESS_AREA)
                ):
                    suppressed += 1
                    continue
                if majority_label in ("T", "Y") and majority_votes >= 1:
                    decision = str(majority_label)
                    reason = "moderate_low_branch_weak_x_vote"
                else:
                    decision = "Y"
                    reason = "moderate_low_branch_weak_x_to_y"
            elif support_four <= 2 and support_three >= 1 and majority_label in ("T", "Y") and majority_votes >= 1:
                decision = str(majority_label)
                reason = "x_threearm_majority_vote"

            elif support_four <= 1 and support_three <= 1:
                decision = "Y"
                reason = "weak_x_multiradius_support"
            elif center_source == "legacy_centroid_fallback" and support_four <= 1:
                decision = "Y"
                reason = "fallback_center_weak_x"
            elif support_four <= 1 and majority_votes >= 1 and majority_label in ("T", "Y"):
                if majority_label == "T" and local_ambiguous:
                    decision = "Y"
                    reason = "weak_x_t_vote_ambiguous"
                else:
                    decision = str(majority_label)
                    reason = "weak_x_majority_vote"

        elif is_graphish and old_type == "T":
            arm_29 = _finite_float_or_none(d.get("ty_rule_arm_count_29"))
            label_29 = str(d.get("ty_rule_label_29", "")).strip().upper()
            center_source = str(d.get("graph_center_source", ""))
            post_adjustment = str(d.get("post_adjustment", ""))
            local_ambiguous = bool(d.get("local_ambiguous", True))
            local_err_t = _finite_float_or_none(d.get("local_err_t"))
            local_err_y = _finite_float_or_none(d.get("local_err_y"))
            arm_count = int(round(_finite_float_or_none(d.get("arm_count")) or 0.0))

            weak_three_arm_support = (arm_29 is None) or int(round(arm_29)) != 3 or label_29 not in ("T", "Y")
            sparse_flags = []
            if local_ambiguous:
                sparse_flags.append("ambiguous")
            if center_source == "legacy_centroid_fallback":
                sparse_flags.append("fallback_center")
            if local_err_t is None or local_err_y is None:
                sparse_flags.append("invalid_local_err")
            elif float(local_err_y) + float(y_advantage_margin) < float(local_err_t):
                sparse_flags.append("y_error_better")

            # Preserve high-confidence corner-recovery T candidates; otherwise sparse-domain
            # debias can erase true but fragmented Ts on low-connectivity skeletons.
            corner_t_keep = (
                center_source == "sparse_recovery_corner"
                and score is not None
                and float(score) >= 0.9
                and not local_ambiguous
            )

            if (
                moderate_low_branch
                and post_adjustment == "X_multiradius_consistency"
                and center_source in ("sparse_recovery_corner", "sparse_recovery")
                and arm_count >= 4
            ):
                decision = "Y"
                reason = "moderate_low_branch_xmr_t_to_y"
            elif (
                moderate_low_branch
                and center_source == "centroid"
                and arm_count == 3
                and score is not None
                and float(score) < float(DEFAULT_GENERALIZATION_MODERATE_LOW_SCORE_T_TO_Y)
            ):
                decision = "Y"
                reason = "moderate_low_branch_low_score_t_to_y"
            elif weak_three_arm_support and len(sparse_flags) >= 1 and not corner_t_keep:
                decision = "Y"
                reason = "weak_t_sparse_geometry"
                d["low_branch_t_sparse_flags"] = ",".join(sparse_flags)
        elif is_graphish and old_type == "Y":
            arm_count = int(round(_finite_float_or_none(d.get("arm_count")) or 0.0))
            post_adjustment = str(d.get("post_adjustment", ""))
            local_err_t = _finite_float_or_none(d.get("local_err_t"))
            local_err_y = _finite_float_or_none(d.get("local_err_y"))
            local_ambiguous = bool(d.get("local_ambiguous", True))
            # Conservative stub suppressor for moderate low-branch scenes:
            # very low-score centroid 3-arm Y with extremely high Y-error is usually a tiny spur.
            if (
                moderate_low_branch
                and center_source == "centroid"
                and arm_count == 3
                and score is not None
                and float(score) < 0.03
                and local_err_y is not None
                and float(local_err_y) > 120.0
            ):
                suppressed += 1
                continue
            if ultra_low_branch:
                if post_adjustment == "X_multiradius_consistency" and arm_count >= 4:
                    decision = "T"
                    reason = "ultra_low_branch_xmr_y_to_t"
                elif (
                    center_source == "sparse_recovery_corner"
                    and arm_count == 3
                    and score is not None
                    and float(score) <= float(DEFAULT_GENERALIZATION_ULTRA_Y_TO_T_MAX_SCORE)
                    and local_err_t is not None
                    and float(local_err_t) <= float(DEFAULT_GENERALIZATION_ULTRA_Y_TO_T_MAX_ERR_T)
                    and local_err_y is not None
                    and float(local_err_y) <= float(DEFAULT_GENERALIZATION_ULTRA_Y_TO_T_MAX_ERR_Y)
                    and not local_ambiguous
                ):
                    decision = "T"
                    reason = "ultra_low_branch_corner_y_to_t"

        d["low_branch_class_debias_decision"] = str(decision)
        d["low_branch_class_debias_reason"] = str(reason)
        if decision in ("T", "Y", "X") and decision != old_type:
            d["type"] = str(decision)
            d["post_adjustment"] = "low_branch_class_debias"
            d["low_branch_class_debias_version"] = LOW_BRANCH_CLASS_DEBIAS_VERSION
            relabeled += 1
        out.append(d)

    if relabeled > 0:
        print(f"Low-branch class debias relabeled {relabeled} detections")
    if suppressed > 0:
        print(f"Low-branch class debias suppressed {suppressed} weak fallback detections")
    return out


def refine_ultra_sparse_t_vote_centers(
    detections,
    image,
    enabled=DEFAULT_GENERALIZATION_ULTRA_T_VOTE_SNAP,
    domain_branch_ratio=None,
    moderate_branch_ratio_threshold=DEFAULT_GENERALIZATION_MODERATE_BRANCH_RATIO,
    line_threshold=127,
    inner_radius=DEFAULT_GRAPH_ARM_INNER_RADIUS,
    outer_radius=DEFAULT_GRAPH_ARM_OUTER_RADIUS,
    min_arm_pixels=DEFAULT_GRAPH_MIN_ARM_PIXELS,
    min_arm_span=DEFAULT_GRAPH_MIN_ARM_SPAN,
    t_min_largest_angle=DEFAULT_GRAPH_T_MIN_LARGEST_ANGLE,
    t_max_side_angle=DEFAULT_GRAPH_T_MAX_SIDE_ANGLE,
    ambiguity_margin=6.0,
    classifier_mode=DEFAULT_THREE_ARM_CLASSIFIER,
    ring_bridge_radius=DEFAULT_GRAPH_RING_GAP_BRIDGE_RADIUS,
    search_radius=DEFAULT_GENERALIZATION_ULTRA_T_VOTE_RADIUS,
    min_majority_votes=DEFAULT_GENERALIZATION_ULTRA_T_VOTE_MIN_VOTES,
    min_support_three=DEFAULT_GENERALIZATION_ULTRA_T_VOTE_MIN_SUPPORT_THREE,
    max_support_four=DEFAULT_GENERALIZATION_ULTRA_T_VOTE_MAX_SUPPORT_FOUR,
):
    if not detections or image is None or not bool(enabled):
        return detections

    branch_ratio = _finite_float_or_none(domain_branch_ratio)
    if branch_ratio is None or float(branch_ratio) >= float(max(0.0, moderate_branch_ratio_threshold)):
        return detections

    line = auto_line_mask(image, line_threshold=int(np.clip(line_threshold, 1, 254)))
    thin = thin_binary(line)
    h, w = thin.shape
    r = int(max(1, search_radius))
    promoted = 0
    moved = 0
    out = []

    for det in detections:
        d = dict(det)
        if str(d.get("type", "")) != "Y":
            out.append(d)
            continue
        source_type = str(d.get("source_type", "")).strip().lower()
        if source_type not in ("graph", "hybrid"):
            out.append(d)
            continue

        majority_label = str(d.get("x_consistency_majority_label", "")).strip().upper()
        majority_votes = int(round(_finite_float_or_none(d.get("x_consistency_majority_votes")) or 0.0))
        support_three = int(round(_finite_float_or_none(d.get("x_consistency_support_three")) or 0.0))
        support_four = int(round(_finite_float_or_none(d.get("x_consistency_support_four")) or 0.0))
        if not (
            majority_label == "T"
            and majority_votes >= int(max(1, min_majority_votes))
            and support_three >= int(max(1, min_support_three))
            and support_four <= int(max(0, max_support_four))
        ):
            out.append(d)
            continue

        y0 = int(d.get("y", 0))
        x0 = int(d.get("x", 0))
        best = None
        for yy in range(max(0, y0 - r), min(h, y0 + r + 1)):
            for xx in range(max(0, x0 - r), min(w, x0 + r + 1)):
                if thin[yy, xx] == 0:
                    continue
                geom = local_geometry_analysis(
                    thin_binary=thin,
                    y=yy,
                    x=xx,
                    inner_radius=inner_radius,
                    outer_radius=outer_radius,
                    min_arm_pixels=min_arm_pixels,
                    min_arm_span=min_arm_span,
                    t_min_largest_angle=t_min_largest_angle,
                    t_max_side_angle=t_max_side_angle,
                    ambiguity_margin=ambiguity_margin,
                    classifier_mode=classifier_mode,
                    snap_radius=0,
                    ring_bridge_radius=ring_bridge_radius,
                )
                if str(geom.get("label", "")) != "T" or int(geom.get("arm_count", 0)) < 3:
                    continue
                score = float(geom.get("score", 0.0))
                err_t = _finite_float_or_none(geom.get("err_t"))
                err_t_term = -float(err_t) if err_t is not None else -1e9
                dist2 = float((yy - y0) * (yy - y0) + (xx - x0) * (xx - x0))
                # Prefer stronger local T evidence; break ties by closeness.
                key = (score, err_t_term, -dist2)
                if best is None or key > best[0]:
                    best = (key, yy, xx)

        if best is None:
            out.append(d)
            continue

        ny = int(best[1])
        nx = int(best[2])
        d["type"] = "T"
        d["ultra_t_vote_snap_from_x"] = int(x0)
        d["ultra_t_vote_snap_from_y"] = int(y0)
        d["x"] = int(nx)
        d["y"] = int(ny)
        d["post_adjustment"] = "ultra_t_vote_snap"
        d["ultra_t_vote_snap_version"] = "v1"
        d["ultra_t_vote_snap_votes"] = int(majority_votes)
        d["ultra_t_vote_snap_support_three"] = int(support_three)
        d["ultra_t_vote_snap_support_four"] = int(support_four)
        promoted += 1
        if nx != x0 or ny != y0:
            moved += 1
        out.append(d)

    if promoted > 0:
        print(f"Ultra sparse T-vote snap promoted {promoted} detections (moved={moved})")
    return out


def _source_kind(det):
    if str(det.get("source_kind", "")).lower() in ("graph", "template", "hybrid"):
        return str(det.get("source_kind")).lower()
    if str(det.get("source_type", "")).lower() == "graph":
        return "graph"
    return "template"


def _cluster_indices_by_distance(detections, distance_threshold):
    if not detections:
        return []
    dist2 = float(distance_threshold * distance_threshold)
    order = sorted(
        range(len(detections)),
        key=lambda i: (
            -float(detections[i].get("score", 0.0)),
            int(detections[i]["y"]),
            int(detections[i]["x"]),
            str(detections[i].get("type", "")),
        ),
    )
    visited = [False] * len(detections)
    clusters = []
    for idx in order:
        if visited[idx]:
            continue
        queue = [idx]
        visited[idx] = True
        cluster = []
        while queue:
            cur = queue.pop()
            cluster.append(cur)
            cy = float(detections[cur]["y"])
            cx = float(detections[cur]["x"])
            for other in order:
                if visited[other]:
                    continue
                dy = float(detections[other]["y"]) - cy
                dx = float(detections[other]["x"]) - cx
                if dx * dx + dy * dy <= dist2:
                    visited[other] = True
                    queue.append(other)
        clusters.append(sorted(cluster))
    return clusters


def _medoid_xy(detections):
    if len(detections) == 1:
        return int(detections[0]["x"]), int(detections[0]["y"])
    pts = np.array([[float(d["x"]), float(d["y"])] for d in detections], dtype=np.float32)
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
    medoid_idx = int(np.argmin(d2.sum(axis=1)))
    return int(round(float(pts[medoid_idx, 0]))), int(round(float(pts[medoid_idx, 1])))


def _merge_detection_group(detections, thin_binary=None, snap_radius=2, forced_label=None, forced_kind=None):
    base = dict(max(detections, key=lambda d: float(d.get("score", 0.0))))
    cx, cy = _medoid_xy(detections)
    if thin_binary is not None:
        sy, sx, ok = nearest_line_point(thin_binary, cy, cx, radius=max(1, int(snap_radius)))
        if ok:
            cy, cx = int(sy), int(sx)
    base["x"] = int(cx)
    base["y"] = int(cy)
    base["score"] = float(max(float(d.get("score", 0.0)) for d in detections))
    if forced_label:
        base["type"] = str(forced_label)
    kinds = sorted({_source_kind(d) for d in detections})
    if forced_kind:
        base["source_kind"] = str(forced_kind)
    elif len(kinds) == 1:
        base["source_kind"] = kinds[0]
    else:
        base["source_kind"] = "hybrid"
    if base["source_kind"] == "hybrid":
        base["source_type"] = "hybrid"
    base["cluster_size"] = int(len(detections))
    base["cluster_source_kinds"] = kinds
    return base


def validate_template_detection(
    detection,
    thin_binary,
    inner_radius=DEFAULT_GRAPH_ARM_INNER_RADIUS,
    outer_radius=DEFAULT_GRAPH_ARM_OUTER_RADIUS,
    min_arm_pixels=DEFAULT_GRAPH_MIN_ARM_PIXELS,
    min_arm_span=DEFAULT_GRAPH_MIN_ARM_SPAN,
    t_min_largest_angle=DEFAULT_GRAPH_T_MIN_LARGEST_ANGLE,
    t_max_side_angle=DEFAULT_GRAPH_T_MAX_SIDE_ANGLE,
    ambiguity_margin=6.0,
    classifier_mode=DEFAULT_THREE_ARM_CLASSIFIER,
):
    geom = local_geometry_classify(
        thin_binary=thin_binary,
        y=int(detection["y"]),
        x=int(detection["x"]),
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        min_arm_pixels=min_arm_pixels,
        min_arm_span=min_arm_span,
        t_min_largest_angle=t_min_largest_angle,
        t_max_side_angle=t_max_side_angle,
        ambiguity_margin=ambiguity_margin,
        classifier_mode=classifier_mode,
        return_debug=True,
    )
    if int(geom["arm_count"]) < 3:
        return None

    out = dict(detection)
    if geom.get("snapped_ok", False):
        out["x"] = int(geom["snapped_x"])
        out["y"] = int(geom["snapped_y"])
    out["template_local_label"] = str(geom["label"])
    out["template_local_arm_count"] = int(geom["arm_count"])
    out["template_local_err_t"] = float(geom.get("err_t", float("inf")))
    out["template_local_err_y"] = float(geom.get("err_y", float("inf")))
    out["template_local_ambiguous"] = bool(geom.get("is_ambiguous", True))

    local_label = str(geom["label"])
    if local_label and local_label != str(out.get("type", "")):
        if not bool(geom.get("is_ambiguous", True)):
            out["type"] = local_label
            out["template_local_relabel"] = True
    return out


def validate_template_detections(
    detections,
    image,
    line_threshold=127,
    inner_radius=DEFAULT_GRAPH_ARM_INNER_RADIUS,
    outer_radius=DEFAULT_GRAPH_ARM_OUTER_RADIUS,
    min_arm_pixels=DEFAULT_GRAPH_MIN_ARM_PIXELS,
    min_arm_span=DEFAULT_GRAPH_MIN_ARM_SPAN,
    t_min_largest_angle=DEFAULT_GRAPH_T_MIN_LARGEST_ANGLE,
    t_max_side_angle=DEFAULT_GRAPH_T_MAX_SIDE_ANGLE,
    ambiguity_margin=6.0,
    classifier_mode=DEFAULT_THREE_ARM_CLASSIFIER,
):
    if not detections:
        return []
    thin = thin_binary(auto_line_mask(image, line_threshold=line_threshold))
    out = []
    rejected = 0
    relabeled = 0
    for det in detections:
        v = validate_template_detection(
            detection=det,
            thin_binary=thin,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            min_arm_pixels=min_arm_pixels,
            min_arm_span=min_arm_span,
            t_min_largest_angle=t_min_largest_angle,
            t_max_side_angle=t_max_side_angle,
            ambiguity_margin=ambiguity_margin,
            classifier_mode=classifier_mode,
        )
        if v is None:
            rejected += 1
            continue
        if v.get("template_local_relabel"):
            relabeled += 1
        out.append(v)
    if rejected > 0 or relabeled > 0:
        print(f"Template local validation rejected={rejected}, relabeled={relabeled}")
    return out


def _resolve_hybrid_cluster(
    cluster_dets,
    thin_binary,
    distance_threshold,
    inner_radius,
    outer_radius,
    min_arm_pixels,
    min_arm_span,
    t_min_largest_angle,
    t_max_side_angle,
    ambiguity_margin,
    classifier_mode,
):
    if len(cluster_dets) == 1:
        return [_merge_detection_group(cluster_dets, thin_binary=thin_binary, snap_radius=max(2, int(inner_radius)))]

    labels = {str(d.get("type", "")) for d in cluster_dets}
    if len(labels) == 1:
        return [_merge_detection_group(cluster_dets, thin_binary=thin_binary, snap_radius=max(2, int(inner_radius)))]

    graph_dets = [d for d in cluster_dets if _source_kind(d) == "graph"]
    template_dets = [d for d in cluster_dets if _source_kind(d) != "graph"]

    centers = []
    if graph_dets:
        gbest = max(graph_dets, key=lambda d: float(d.get("score", 0.0)))
        centers.append((int(gbest["y"]), int(gbest["x"]), "graph_best"))
    if template_dets:
        tbest = max(template_dets, key=lambda d: float(d.get("score", 0.0)))
        centers.append((int(tbest["y"]), int(tbest["x"]), "template_best"))
    wsum = float(sum(max(1e-6, float(d.get("score", 0.0))) for d in cluster_dets))
    if wsum > 0.0:
        avg_x = int(round(sum(float(d["x"]) * max(1e-6, float(d.get("score", 0.0))) for d in cluster_dets) / wsum))
        avg_y = int(round(sum(float(d["y"]) * max(1e-6, float(d.get("score", 0.0))) for d in cluster_dets) / wsum))
        centers.append((avg_y, avg_x, "score_weighted"))

    best_local = None
    for cy, cx, source in centers:
        geom = local_geometry_classify(
            thin_binary=thin_binary,
            y=cy,
            x=cx,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            min_arm_pixels=min_arm_pixels,
            min_arm_span=min_arm_span,
            t_min_largest_angle=t_min_largest_angle,
            t_max_side_angle=t_max_side_angle,
            ambiguity_margin=ambiguity_margin,
            classifier_mode=classifier_mode,
            return_debug=True,
        )
        if int(geom["arm_count"]) < 3 or not str(geom["label"]):
            continue
        if str(geom["label"]) in ("T", "Y") and bool(geom.get("is_ambiguous", True)):
            continue
        key = (
            -int(geom["arm_count"]),
            float(geom["class_error"]),
            float(geom["snap_distance"]),
            source,
        )
        if best_local is None or key < best_local[0]:
            best_local = (key, source, geom)

    if best_local is not None:
        _, source, geom = best_local
        local_label = str(geom["label"])
        supporting = [d for d in cluster_dets if str(d.get("type", "")) == local_label]
        merged = _merge_detection_group(
            supporting if supporting else cluster_dets,
            thin_binary=thin_binary,
            snap_radius=max(2, int(inner_radius)),
            forced_label=local_label,
            forced_kind="hybrid",
        )
        merged["x"] = int(geom["snapped_x"])
        merged["y"] = int(geom["snapped_y"])
        merged["fusion_reason"] = "local_geometry_arbitration"
        merged["fusion_center_source"] = str(source)
        merged["fusion_local_err_t"] = float(geom.get("err_t", float("inf")))
        merged["fusion_local_err_y"] = float(geom.get("err_y", float("inf")))
        return [merged]

    if graph_dets and template_dets:
        gbest = max(graph_dets, key=lambda d: float(d.get("score", 0.0)))
        tbest = max(template_dets, key=lambda d: float(d.get("score", 0.0)))
        gscore = float(gbest.get("score", 0.0))
        tscore = float(tbest.get("score", 0.0))
        if gscore >= tscore + 0.05:
            chosen = _merge_detection_group(
                [d for d in graph_dets if str(d.get("type", "")) == str(gbest.get("type", ""))],
                thin_binary=thin_binary,
                snap_radius=max(2, int(inner_radius)),
            )
            chosen["fusion_reason"] = "graph_priority_confidence"
            return [chosen]
        if tscore >= gscore + 0.05:
            chosen = _merge_detection_group(
                [d for d in template_dets if str(d.get("type", "")) == str(tbest.get("type", ""))],
                thin_binary=thin_binary,
                snap_radius=max(2, int(inner_radius)),
            )
            chosen["fusion_reason"] = "template_priority_confidence"
            return [chosen]

        ggeom = local_geometry_classify(
            thin_binary=thin_binary,
            y=int(gbest["y"]),
            x=int(gbest["x"]),
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            min_arm_pixels=min_arm_pixels,
            min_arm_span=min_arm_span,
            t_min_largest_angle=t_min_largest_angle,
            t_max_side_angle=t_max_side_angle,
            ambiguity_margin=ambiguity_margin,
            classifier_mode=classifier_mode,
            return_debug=True,
        )
        tgeom = local_geometry_classify(
            thin_binary=thin_binary,
            y=int(tbest["y"]),
            x=int(tbest["x"]),
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            min_arm_pixels=min_arm_pixels,
            min_arm_span=min_arm_span,
            t_min_largest_angle=t_min_largest_angle,
            t_max_side_angle=t_max_side_angle,
            ambiguity_margin=ambiguity_margin,
            classifier_mode=classifier_mode,
            return_debug=True,
        )
        sep = float(np.hypot(float(ggeom["snapped_y"]) - float(tgeom["snapped_y"]), float(ggeom["snapped_x"]) - float(tgeom["snapped_x"])))
        if sep >= float(distance_threshold):
            gkeep = _merge_detection_group(graph_dets, thin_binary=thin_binary, snap_radius=max(2, int(inner_radius)))
            tkeep = _merge_detection_group(template_dets, thin_binary=thin_binary, snap_radius=max(2, int(inner_radius)))
            gkeep["fusion_reason"] = "separate_after_snap"
            tkeep["fusion_reason"] = "separate_after_snap"
            return [gkeep, tkeep]
        keep = _merge_detection_group([gbest if gscore >= tscore else tbest], thin_binary=thin_binary, snap_radius=max(2, int(inner_radius)))
        keep["fusion_reason"] = "ambiguous_fallback_single"
        return [keep]

    return [_merge_detection_group(cluster_dets, thin_binary=thin_binary, snap_radius=max(2, int(inner_radius)))]


def fuse_hybrid_detections(
    template_dets,
    graph_dets,
    distance_threshold=DEFAULT_NMS_DISTANCE,
    image=None,
    line_threshold=127,
    inner_radius=DEFAULT_GRAPH_ARM_INNER_RADIUS,
    outer_radius=DEFAULT_GRAPH_ARM_OUTER_RADIUS,
    min_arm_pixels=DEFAULT_GRAPH_MIN_ARM_PIXELS,
    min_arm_span=DEFAULT_GRAPH_MIN_ARM_SPAN,
    t_min_largest_angle=DEFAULT_GRAPH_T_MIN_LARGEST_ANGLE,
    t_max_side_angle=DEFAULT_GRAPH_T_MAX_SIDE_ANGLE,
    ambiguity_margin=6.0,
    classifier_mode=DEFAULT_THREE_ARM_CLASSIFIER,
):
    combined = []
    for d in graph_dets:
        item = dict(d)
        item["source_kind"] = "graph"
        combined.append(item)
    for d in template_dets:
        item = dict(d)
        item["source_kind"] = "template"
        combined.append(item)
    if not combined:
        return []

    if image is None:
        # Fallback mode preserves graph-priority behavior if geometry image is unavailable.
        dist2 = float(distance_threshold * distance_threshold)
        fused = list(graph_dets)
        for det in template_dets:
            keep = True
            for g in graph_dets:
                dy = float(det["y"] - g["y"])
                dx = float(det["x"] - g["x"])
                if dx * dx + dy * dy <= dist2:
                    keep = False
                    break
            if keep:
                fused.append(det)
        return non_maximum_suppression(fused, distance_threshold=distance_threshold)

    thin = thin_binary(auto_line_mask(image, line_threshold=line_threshold))
    clusters = _cluster_indices_by_distance(combined, distance_threshold)
    fused = []
    for cluster in clusters:
        cluster_dets = [combined[i] for i in cluster]
        fused.extend(
            _resolve_hybrid_cluster(
                cluster_dets=cluster_dets,
                thin_binary=thin,
                distance_threshold=distance_threshold,
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                min_arm_pixels=min_arm_pixels,
                min_arm_span=min_arm_span,
                t_min_largest_angle=t_min_largest_angle,
                t_max_side_angle=t_max_side_angle,
                ambiguity_margin=ambiguity_margin,
                classifier_mode=classifier_mode,
            )
        )
    return non_maximum_suppression(fused, distance_threshold=distance_threshold)


def non_maximum_suppression(detections, distance_threshold=DEFAULT_NMS_DISTANCE, respect_label=True):
    if not detections:
        return []

    merged = []
    if respect_label:
        by_label = defaultdict(list)
        for det in detections:
            by_label[str(det.get("type", ""))].append(det)
        for label in sorted(by_label.keys()):
            group = by_label[label]
            clusters = _cluster_indices_by_distance(group, distance_threshold)
            for cluster in clusters:
                members = [group[i] for i in cluster]
                merged.append(
                    _merge_detection_group(
                        members,
                        thin_binary=None,
                        snap_radius=0,
                        forced_label=label,
                    )
                )
    else:
        clusters = _cluster_indices_by_distance(detections, distance_threshold)
        for cluster in clusters:
            members = [detections[i] for i in cluster]
            by_label = defaultdict(list)
            for det in members:
                by_label[str(det.get("type", ""))].append(det)
            label_items = []
            for label, label_members in by_label.items():
                max_score = max(float(d.get("score", 0.0)) for d in label_members)
                label_items.append((label, max_score, len(label_members), label_members))
            label_items.sort(key=lambda item: (-item[1], -item[2], item[0]))
            winner_label, _, _, winner_members = label_items[0]
            merged.append(
                _merge_detection_group(
                    winner_members,
                    thin_binary=None,
                    snap_radius=0,
                    forced_label=winner_label,
                )
            )

    merged.sort(
        key=lambda d: (
            -float(d.get("score", 0.0)),
            int(d["y"]),
            int(d["x"]),
            str(d.get("type", "")),
        )
    )
    return merged


def filter_by_type_agreement(detections, distance_threshold=DEFAULT_NMS_DISTANCE):
    # Second NMS pass keeps only strongest local hypothesis across competing labels.
    return non_maximum_suppression(detections, distance_threshold, respect_label=False)


def count_by_type(detections, labels):
    counts = {label: 0 for label in labels}
    for det in detections:
        if det["type"] in counts:
            counts[det["type"]] += 1
    return counts


def visualize_detections(image, detections, output_path):
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = vis.copy()
    colors = {
        "T": (0, 140, 255),   # Orange
        "Y": (0, 255, 0),     # Green
        "X": (255, 0, 255),   # Magenta
        "V": (0, 165, 255),   # Legacy fallback
    }

    for det in detections:
        color = colors.get(det["type"], (255, 255, 255))
        x = int(det["x"])
        y = int(det["y"])
        cv2.circle(overlay, (x, y), 6, color, -1)
        cv2.putText(
            vis,
            det["type"],
            (x + 8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.5, vis, 0.5, 0.0, vis)
    cv2.imwrite(output_path, vis)
    print(f"Saved visualization: {output_path}")


def estimate_skeleton_branch_ratio(image, line_threshold=127, branch_degree=3):
    line = auto_line_mask(image, line_threshold=int(np.clip(line_threshold, 1, 254)))
    thin = thin_binary(line)
    nbr = neighbor_count(thin)
    degree = int(max(3, branch_degree))
    branch = ((thin == 1) & (nbr >= degree)).astype(np.uint8)
    thin_pixels = int(thin.sum())
    branch_pixels = int(branch.sum())
    branch_ratio = float(branch_pixels / max(1, thin_pixels))
    return {
        "thin_pixels": thin_pixels,
        "branch_pixels": branch_pixels,
        "branch_ratio": branch_ratio,
        "branch_degree": degree,
    }


def apply_generalization_geometry_profile(
    image,
    line_threshold,
    branch_degree,
    nms_distance,
    graph_arm_outer_radius,
    graph_max_centers_per_component,
    enabled=DEFAULT_GENERALIZATION_AUTO_SCALE,
    branch_ratio_threshold=DEFAULT_GENERALIZATION_BRANCH_RATIO_THRESHOLD,
    low_outer_radius=DEFAULT_GENERALIZATION_LOW_OUTER_RADIUS,
    low_nms_distance=DEFAULT_GENERALIZATION_LOW_NMS_DISTANCE,
    low_max_centers=DEFAULT_GENERALIZATION_LOW_MAX_CENTERS,
):
    eff = {
        "nms_distance": int(max(1, nms_distance)),
        "graph_arm_outer_radius": int(max(2, graph_arm_outer_radius)),
        "graph_max_centers_per_component": int(max(1, graph_max_centers_per_component)),
    }
    profile = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled",
        "branch_ratio_threshold": float(branch_ratio_threshold),
    }
    if not bool(enabled):
        return eff, profile

    stats = estimate_skeleton_branch_ratio(
        image=image,
        line_threshold=line_threshold,
        branch_degree=branch_degree,
    )
    profile.update(stats)
    threshold = float(max(0.0, branch_ratio_threshold))
    if float(stats["branch_ratio"]) < threshold:
        eff["graph_arm_outer_radius"] = int(min(eff["graph_arm_outer_radius"], max(2, int(low_outer_radius))))
        eff["nms_distance"] = int(max(eff["nms_distance"], max(1, int(low_nms_distance))))
        eff["graph_max_centers_per_component"] = int(max(eff["graph_max_centers_per_component"], max(1, int(low_max_centers))))
        profile["applied"] = True
        profile["reason"] = "low_branch_ratio_profile"
    else:
        profile["reason"] = "high_branch_ratio_keep_default"
    return eff, profile


def detect_junctions(
    image,
    template_dir,
    match_threshold=DEFAULT_MATCH_THRESHOLD,
    nms_distance=DEFAULT_NMS_DISTANCE,
    labels=None,
    allow_legacy_v=True,
    use_topology_gate=True,
    line_threshold=127,
    branch_degree=3,
    gate_radius=6,
    detection_mode=DEFAULT_DETECTION_MODE,
    graph_arm_inner_radius=DEFAULT_GRAPH_ARM_INNER_RADIUS,
    graph_arm_outer_radius=DEFAULT_GRAPH_ARM_OUTER_RADIUS,
    graph_min_arm_pixels=DEFAULT_GRAPH_MIN_ARM_PIXELS,
    graph_min_arm_span=DEFAULT_GRAPH_MIN_ARM_SPAN,
    graph_min_branch_component_area=DEFAULT_GRAPH_MIN_BRANCH_COMPONENT_AREA,
    graph_max_centers_per_component=DEFAULT_GRAPH_MAX_CENTERS_PER_COMPONENT,
    graph_ring_gap_bridge_radius=DEFAULT_GRAPH_RING_GAP_BRIDGE_RADIUS,
    graph_sparse_recovery=DEFAULT_GRAPH_SPARSE_RECOVERY,
    graph_sparse_recovery_support_radius=DEFAULT_GRAPH_SPARSE_RECOVERY_SUPPORT_RADIUS,
    graph_sparse_recovery_endpoint_radius=DEFAULT_GRAPH_SPARSE_RECOVERY_ENDPOINT_RADIUS,
    graph_sparse_recovery_max_component_area=DEFAULT_GRAPH_SPARSE_RECOVERY_MAX_COMPONENT_AREA,
    graph_sparse_recovery_min_score=DEFAULT_GRAPH_SPARSE_RECOVERY_MIN_SCORE,
    graph_junction_snap_repair=DEFAULT_GRAPH_JUNCTION_SNAP_REPAIR,
    graph_junction_snap_iters=DEFAULT_GRAPH_JUNCTION_SNAP_ITERS,
    graph_junction_snap_dist=DEFAULT_GRAPH_JUNCTION_SNAP_DIST,
    graph_junction_snap_min_dot=DEFAULT_GRAPH_JUNCTION_SNAP_MIN_DOT,
    graph_junction_snap_max_existing_fraction=DEFAULT_GRAPH_JUNCTION_SNAP_MAX_EXISTING_FRACTION,
    graph_t_min_largest_angle=DEFAULT_GRAPH_T_MIN_LARGEST_ANGLE,
    graph_t_max_side_angle=DEFAULT_GRAPH_T_MAX_SIDE_ANGLE,
    graph_center_mode=DEFAULT_GRAPH_CENTER_MODE,
    three_arm_classifier=DEFAULT_THREE_ARM_CLASSIFIER,
    hybrid_fusion_mode=DEFAULT_HYBRID_FUSION_MODE,
    geometry_ambiguity_margin=6.0,
    local_reclassify_t=DEFAULT_LOCAL_RECLASSIFY_T,
    local_reclassify_y_to_t=DEFAULT_LOCAL_RECLASSIFY_Y_TO_T,
    ty_multiradius_vote=DEFAULT_TY_MULTIRADIUS_VOTE,
    ty_vote_outer_radii=DEFAULT_TY_VOTE_OUTER_RADII,
    ty_vote_min_votes=DEFAULT_TY_VOTE_MIN_VOTES,
    ty_vote_margin=DEFAULT_TY_VOTE_MARGIN,
    ty_vote_require_non_ambiguous=DEFAULT_TY_VOTE_REQUIRE_NON_AMBIG,
    ty_feature_rules=DEFAULT_TY_FEATURE_RULES,
    ty_feature_rule_mode=DEFAULT_TY_FEATURE_RULE_MODE,
    tyx_structural_rules=DEFAULT_TYX_STRUCTURAL_RULES,
    tyx_structural_rule_mode=DEFAULT_TYX_STRUCTURAL_RULE_MODE,
    tyx_structural_ood_gate=DEFAULT_TYX_STRUCTURAL_OOD_GATE,
    tyx_structural_min_margin=DEFAULT_TYX_STRUCTURAL_MIN_MARGIN,
    x_consistency_reclassify=DEFAULT_X_CONSISTENCY_RECLASSIFY,
    x_consistency_outer_radii=DEFAULT_X_CONSISTENCY_OUTER_RADII,
    x_consistency_max_fourarm_votes=DEFAULT_X_CONSISTENCY_MAX_FOURARM,
    x_consistency_min_threearm_votes=DEFAULT_X_CONSISTENCY_MIN_THREEARM_VOTES,
    generalization_auto_scale=DEFAULT_GENERALIZATION_AUTO_SCALE,
    generalization_branch_ratio_threshold=DEFAULT_GENERALIZATION_BRANCH_RATIO_THRESHOLD,
    generalization_low_outer_radius=DEFAULT_GENERALIZATION_LOW_OUTER_RADIUS,
    generalization_low_nms_distance=DEFAULT_GENERALIZATION_LOW_NMS_DISTANCE,
    generalization_low_max_centers=DEFAULT_GENERALIZATION_LOW_MAX_CENTERS,
    generalization_low_ring_gap_bridge_radius=DEFAULT_GENERALIZATION_LOW_RING_GAP_BRIDGE_RADIUS,
    generalization_low_sparse_recovery=DEFAULT_GENERALIZATION_LOW_SPARSE_RECOVERY,
    generalization_low_sparse_endpoint_radius=DEFAULT_GENERALIZATION_LOW_SPARSE_ENDPOINT_RADIUS,
    generalization_low_sparse_min_score=DEFAULT_GENERALIZATION_LOW_SPARSE_MIN_SCORE,
    generalization_low_x_consistency_reclassify=DEFAULT_GENERALIZATION_LOW_X_CONSISTENCY_RECLASSIFY,
    generalization_low_final_merge_distance=DEFAULT_GENERALIZATION_LOW_FINAL_MERGE_DISTANCE,
    generalization_low_class_debias=DEFAULT_GENERALIZATION_LOW_CLASS_DEBIAS,
    preloaded_templates=None,
    skip_thinning=False,
):
    if labels is None:
        labels = list(DEFAULT_LABELS)
    mode = str(detection_mode).strip().lower()
    if mode not in DETECTION_MODES:
        raise ValueError(f"Unknown detection_mode '{detection_mode}'. Use one of: {', '.join(DETECTION_MODES)}")

    final = []
    gate_stats = None

    effective, gen_profile = apply_generalization_geometry_profile(
        image=image,
        line_threshold=line_threshold,
        branch_degree=branch_degree,
        nms_distance=nms_distance,
        graph_arm_outer_radius=graph_arm_outer_radius,
        graph_max_centers_per_component=graph_max_centers_per_component,
        enabled=generalization_auto_scale,
        branch_ratio_threshold=generalization_branch_ratio_threshold,
        low_outer_radius=generalization_low_outer_radius,
        low_nms_distance=generalization_low_nms_distance,
        low_max_centers=generalization_low_max_centers,
    )
    effective_nms_distance = int(effective["nms_distance"])
    effective_graph_outer_radius = int(effective["graph_arm_outer_radius"])
    effective_graph_max_centers = int(effective["graph_max_centers_per_component"])
    effective_ring_gap_bridge_radius = int(max(0, graph_ring_gap_bridge_radius))
    effective_sparse_recovery = bool(graph_sparse_recovery)
    effective_sparse_endpoint_radius = int(max(0, graph_sparse_recovery_endpoint_radius))
    effective_sparse_min_score = float(max(0.0, graph_sparse_recovery_min_score))
    effective_x_consistency_reclassify = bool(x_consistency_reclassify)
    effective_low_branch_class_debias = False
    effective_low_branch_final_merge_distance = 0
    effective_ultra_low_corner_recovery = False
    effective_ultra_low_corner_min_score = None
    effective_ultra_low_corner_endpoint_radius = 0
    low_branch_ultra_profile = False
    low_branch_moderate_profile = False
    if bool(gen_profile.get("enabled", False)) and bool(gen_profile.get("applied", False)):
        effective_graph_outer_radius = int(max(effective_graph_outer_radius, int(DEFAULT_GENERALIZATION_LOW_OUTER_RADIUS)))
        effective_nms_distance = int(max(effective_nms_distance, int(DEFAULT_GENERALIZATION_LOW_NMS_DISTANCE)))
        effective_graph_max_centers = int(min(effective_graph_max_centers, int(DEFAULT_GENERALIZATION_LOW_MAX_CENTERS)))
        effective_ring_gap_bridge_radius = int(max(effective_ring_gap_bridge_radius, int(generalization_low_ring_gap_bridge_radius)))
        effective_sparse_recovery = bool(generalization_low_sparse_recovery)
        effective_sparse_endpoint_radius = int(max(effective_sparse_endpoint_radius, int(generalization_low_sparse_endpoint_radius)))
        effective_sparse_min_score = float(max(effective_sparse_min_score, float(generalization_low_sparse_min_score)))
        effective_x_consistency_reclassify = bool(generalization_low_x_consistency_reclassify)
        effective_low_branch_class_debias = bool(generalization_low_class_debias)
        effective_low_branch_final_merge_distance = int(max(0, int(generalization_low_final_merge_distance)))
        branch_ratio = float(gen_profile.get("branch_ratio", 1.0))
        low_branch_ultra_profile = (
            branch_ratio >= 0.0
            and branch_ratio < float(DEFAULT_GENERALIZATION_MODERATE_BRANCH_RATIO)
        )
        low_branch_moderate_profile = (
            branch_ratio >= float(DEFAULT_GENERALIZATION_MODERATE_BRANCH_RATIO)
        )
        if low_branch_moderate_profile:
            # Moderate low-branch tiles are recall-richer; avoid over-bridging and
            # suppress noisy X->T reclassification from sparse corner artifacts.
            effective_x_consistency_reclassify = False
            # Keep moderate profile at least at the low-branch default bridge level;
            # lower bridge values increased isolated stub/phantom Y detections in transfer tiles.
            effective_ring_gap_bridge_radius = int(
                max(
                    int(DEFAULT_GENERALIZATION_LOW_RING_GAP_BRIDGE_RADIUS),
                    int(max(0, int(generalization_low_ring_gap_bridge_radius))),
                )
            )
            effective_sparse_endpoint_radius = int(max(0, int(generalization_low_sparse_endpoint_radius)))
            effective_sparse_min_score = float(max(0.0, float(generalization_low_sparse_min_score)))
            if effective_low_branch_final_merge_distance > 0:
                effective_low_branch_final_merge_distance = int(
                    min(
                        int(effective_low_branch_final_merge_distance),
                        int(DEFAULT_GENERALIZATION_MODERATE_FINAL_MERGE_DISTANCE),
                    )
                )
        if low_branch_ultra_profile:
            # Ultra-low connectivity tiles need more candidate recovery and less
            # aggressive dedup to avoid dropping nearby true nodes.
            effective_graph_max_centers = int(max(effective_graph_max_centers, int(DEFAULT_GENERALIZATION_ULTRA_LOW_MAX_CENTERS)))
            effective_ring_gap_bridge_radius = int(max(effective_ring_gap_bridge_radius, int(DEFAULT_GENERALIZATION_LOW_RING_GAP_BRIDGE_RADIUS)))
            # Keep ultra profile endpoint support tightly bounded to low-branch controls.
            # Inheriting a large global endpoint radius (e.g., 10) can over-seed sparse
            # recovery and create phantom stub detections on fragmented skeletons.
            effective_sparse_endpoint_radius = int(
                max(
                    int(max(0, int(generalization_low_sparse_endpoint_radius))),
                    int(DEFAULT_GENERALIZATION_ULTRA_LOW_CORNER_ENDPOINT_RADIUS),
                )
            )
            effective_sparse_min_score = float(min(effective_sparse_min_score, float(DEFAULT_GENERALIZATION_LOW_SPARSE_MIN_SCORE)))
            if effective_low_branch_final_merge_distance > 0:
                effective_low_branch_final_merge_distance = int(
                    min(
                        int(effective_low_branch_final_merge_distance),
                        int(DEFAULT_GENERALIZATION_ULTRA_LOW_FINAL_MERGE_DISTANCE),
                    )
                )
        if (
            bool(DEFAULT_GENERALIZATION_ULTRA_LOW_CORNER_RECOVERY)
            and branch_ratio >= 0.0
            and branch_ratio < float(DEFAULT_GENERALIZATION_ULTRA_LOW_BRANCH_RATIO)
        ):
            effective_ultra_low_corner_recovery = True
            effective_ultra_low_corner_min_score = float(DEFAULT_GENERALIZATION_ULTRA_LOW_CORNER_MIN_SCORE)
            effective_ultra_low_corner_endpoint_radius = int(max(effective_sparse_endpoint_radius, DEFAULT_GENERALIZATION_ULTRA_LOW_CORNER_ENDPOINT_RADIUS))
    if bool(gen_profile.get("enabled", False)):
        print(
            "Generalization auto-scale "
            f"(ratio={float(gen_profile.get('branch_ratio', -1.0)):.4f}, "
            f"threshold={float(gen_profile.get('branch_ratio_threshold', 0.0)):.4f}, "
            f"applied={bool(gen_profile.get('applied', False))}, "
            f"outer={effective_graph_outer_radius}, nms={effective_nms_distance}, "
            f"max_centers={effective_graph_max_centers}, bridge={effective_ring_gap_bridge_radius}, "
            f"sparse_recovery={effective_sparse_recovery}, endpoint_r={effective_sparse_endpoint_radius}, "
            f"ultra_corner={effective_ultra_low_corner_recovery}, "
            f"profile_ultra={low_branch_ultra_profile}, profile_moderate={low_branch_moderate_profile}, "
            f"x_consistency={effective_x_consistency_reclassify}, "
            f"class_debias={effective_low_branch_class_debias}, "
            f"sparse_min_score={effective_sparse_min_score:.3f}, "
            f"final_merge={effective_low_branch_final_merge_distance})"
        )

    if mode in ("template", "hybrid"):
        if not template_dir:
            raise ValueError("template_dir is required for template or hybrid detection modes.")
        templates = preloaded_templates
        if templates is None:
            templates = load_templates(template_dir, allowed_labels=labels, allow_legacy_v=allow_legacy_v)
        if not templates:
            raise FileNotFoundError(f"No templates loaded from {template_dir}")

        # Auto-detect polarity mismatch and invert templates if needed.
        # Templates are dark lines on light background (~gray 128+), but skeleton
        # images are bright pixels on black. NCC requires matching polarity.
        if image.mean() < 128:
            templates = {
                jtype: [dict(t, image=255 - t["image"]) for t in tlist]
                for jtype, tlist in templates.items()
            }
            print("Auto-inverted template polarity for bright-on-dark input")

        raw = template_match_multiscale(image, templates, threshold=match_threshold)
        nms_once = non_maximum_suppression(raw, distance_threshold=effective_nms_distance)
        templ_final = filter_by_type_agreement(nms_once, distance_threshold=effective_nms_distance)
        if use_topology_gate:
            gate_mask, gate_stats = build_topology_gate(
                image=image,
                line_threshold=line_threshold,
                branch_degree=branch_degree,
                gate_radius=gate_radius,
            )
            before = len(templ_final)
            templ_final = filter_detections_by_gate(templ_final, gate_mask)
            print(
                f"Topology gate kept {len(templ_final)}/{before} detections "
                f"(junction_pixels={gate_stats['junction_pixels']})"
            )
        templ_final = validate_template_detections(
            detections=templ_final,
            image=image,
            line_threshold=line_threshold,
            inner_radius=graph_arm_inner_radius,
            outer_radius=effective_graph_outer_radius,
            min_arm_pixels=graph_min_arm_pixels,
            min_arm_span=graph_min_arm_span,
            t_min_largest_angle=graph_t_min_largest_angle,
            t_max_side_angle=graph_t_max_side_angle,
            ambiguity_margin=geometry_ambiguity_margin,
            classifier_mode=three_arm_classifier,
        )
        final.extend(templ_final)

    if mode in ("graph", "hybrid"):
        graph_final, graph_stats = graph_junction_detections(
            image=image,
            line_threshold=line_threshold,
            branch_degree=branch_degree,
            arm_inner_radius=graph_arm_inner_radius,
            arm_outer_radius=effective_graph_outer_radius,
            min_arm_pixels=graph_min_arm_pixels,
            min_arm_span=graph_min_arm_span,
            min_branch_component_area=graph_min_branch_component_area,
            max_centers_per_component=effective_graph_max_centers,
            ring_gap_bridge_radius=effective_ring_gap_bridge_radius,
            sparse_recovery=effective_sparse_recovery,
            sparse_recovery_support_radius=graph_sparse_recovery_support_radius,
            sparse_recovery_endpoint_radius=effective_sparse_endpoint_radius,
            sparse_recovery_max_component_area=graph_sparse_recovery_max_component_area,
            sparse_recovery_min_score=effective_sparse_min_score,
            junction_snap_repair=graph_junction_snap_repair,
            junction_snap_iters=graph_junction_snap_iters,
            junction_snap_dist=graph_junction_snap_dist,
            junction_snap_min_dot=graph_junction_snap_min_dot,
            junction_snap_max_existing_fraction=graph_junction_snap_max_existing_fraction,
            sparse_corner_recovery=effective_ultra_low_corner_recovery,
            sparse_corner_max_angle=DEFAULT_SPARSE_CORNER_MAX_ANGLE,
            sparse_corner_endpoint_radius=effective_ultra_low_corner_endpoint_radius,
            sparse_corner_min_score=effective_ultra_low_corner_min_score,
            centroid_best_fallback=bool(low_branch_moderate_profile),
            nms_distance=effective_nms_distance,
            t_min_largest_angle=graph_t_min_largest_angle,
            t_max_side_angle=graph_t_max_side_angle,
            ambiguity_margin=geometry_ambiguity_margin,
            center_mode=graph_center_mode,
            classifier_mode=three_arm_classifier,
            skip_thinning=skip_thinning,
        )
        if gate_stats is None:
            gate_stats = graph_stats
        else:
            gate_stats.update(
                {
                    "graph_line_pixels": graph_stats["line_pixels"],
                    "graph_thin_pixels": graph_stats["thin_pixels"],
                    "graph_candidate_thin_pixels": graph_stats.get("candidate_thin_pixels", graph_stats["thin_pixels"]),
                    "graph_junction_pixels": graph_stats["junction_pixels"],
                    "graph_junction_components": graph_stats["junction_components"],
                    "graph_junction_snap_repair": bool(graph_stats.get("junction_snap_repair", False)),
                    "graph_junction_snap_links_added": int(graph_stats.get("junction_snap_links_added", 0)),
                    "graph_junction_snap_iters_applied": int(graph_stats.get("junction_snap_iters_applied", 0)),
                }
            )
        print(
            f"Graph mode produced {len(graph_final)} detections "
            f"(junction_components={graph_stats['junction_components']})"
        )
        final.extend(graph_final)

    if mode == "hybrid":
        fusion_mode = str(hybrid_fusion_mode).strip().lower()
        if fusion_mode not in HYBRID_FUSION_MODES:
            fusion_mode = DEFAULT_HYBRID_FUSION_MODE
        final = fuse_hybrid_detections(
            template_dets=[d for d in final if d.get("source_type") != "graph"],
            graph_dets=[d for d in final if d.get("source_type") == "graph"],
            distance_threshold=effective_nms_distance,
            image=image if fusion_mode == "conflict_aware" else None,
            line_threshold=line_threshold,
            inner_radius=graph_arm_inner_radius,
            outer_radius=effective_graph_outer_radius,
            min_arm_pixels=graph_min_arm_pixels,
            min_arm_span=graph_min_arm_span,
            t_min_largest_angle=graph_t_min_largest_angle,
            t_max_side_angle=graph_t_max_side_angle,
            ambiguity_margin=geometry_ambiguity_margin,
            classifier_mode=three_arm_classifier,
        )
    if local_reclassify_t or local_reclassify_y_to_t:
        final = downgrade_t_to_y_by_local_geometry(
            detections=final,
            image=image,
            line_threshold=line_threshold,
            inner_radius=graph_arm_inner_radius,
            outer_radius=effective_graph_outer_radius,
            min_arm_pixels=graph_min_arm_pixels,
            min_arm_span=graph_min_arm_span,
            t_min_largest_angle=graph_t_min_largest_angle,
            t_max_side_angle=graph_t_max_side_angle,
            ambiguity_margin=geometry_ambiguity_margin,
            classifier_mode=three_arm_classifier,
            reclassify_t_to_y=local_reclassify_t,
            reclassify_y_to_t=local_reclassify_y_to_t,
        )
    if ty_multiradius_vote:
        final = reclassify_t_y_by_multiradius_vote(
            detections=final,
            image=image,
            enabled=ty_multiradius_vote,
            line_threshold=line_threshold,
            inner_radius=graph_arm_inner_radius,
            outer_radii=ty_vote_outer_radii,
            min_votes=ty_vote_min_votes,
            vote_margin=ty_vote_margin,
            require_non_ambiguous=ty_vote_require_non_ambiguous,
            min_arm_pixels=graph_min_arm_pixels,
            min_arm_span=graph_min_arm_span,
            t_min_largest_angle=graph_t_min_largest_angle,
            t_max_side_angle=graph_t_max_side_angle,
            ambiguity_margin=geometry_ambiguity_margin,
            classifier_mode=three_arm_classifier,
        )
    if ty_feature_rules:
        final = reclassify_t_y_by_feature_rules(
            detections=final,
            image=image,
            enabled=ty_feature_rules,
            mode=ty_feature_rule_mode,
            line_threshold=line_threshold,
            outer_radii=DEFAULT_TY_FEATURE_RULE_OUTER_RADII,
            inner_radius=DEFAULT_TY_FEATURE_RULE_INNER_RADIUS,
            min_arm_pixels=DEFAULT_TY_FEATURE_RULE_MIN_ARM_PIXELS,
            min_arm_span=DEFAULT_TY_FEATURE_RULE_MIN_ARM_SPAN,
            t_min_largest_angle=DEFAULT_TY_FEATURE_RULE_T_MIN_LARGEST_ANGLE,
            t_max_side_angle=DEFAULT_TY_FEATURE_RULE_T_MAX_SIDE_ANGLE,
            ambiguity_margin=DEFAULT_TY_FEATURE_RULE_AMBIGUITY_MARGIN,
            classifier_mode=DEFAULT_TY_FEATURE_RULE_CLASSIFIER,
        )
    if tyx_structural_rules:
        final = reclassify_tyx_by_structural_rules(
            detections=final,
            enabled=tyx_structural_rules,
            mode=tyx_structural_rule_mode,
            ood_gate=tyx_structural_ood_gate,
            min_margin=tyx_structural_min_margin,
            domain_branch_ratio=gen_profile.get("branch_ratio", None),
            weak_x_override=bool(generalization_auto_scale),
            weak_x_branch_ratio_threshold=generalization_branch_ratio_threshold,
        )
    if effective_x_consistency_reclassify:
        final = reclassify_x_by_multiradius_consistency(
            detections=final,
            image=image,
            enabled=effective_x_consistency_reclassify,
            line_threshold=line_threshold,
            inner_radius=graph_arm_inner_radius,
            outer_radii=x_consistency_outer_radii,
            min_arm_pixels=graph_min_arm_pixels,
            min_arm_span=graph_min_arm_span,
            t_min_largest_angle=graph_t_min_largest_angle,
            t_max_side_angle=graph_t_max_side_angle,
            ambiguity_margin=geometry_ambiguity_margin,
            classifier_mode=three_arm_classifier,
            ring_bridge_radius=effective_ring_gap_bridge_radius,
            max_fourarm_votes=max(0, int(x_consistency_max_fourarm_votes)),
            min_threearm_votes=max(1, int(x_consistency_min_threearm_votes)),
        )
    if effective_low_branch_class_debias:
        final = reclassify_low_branch_sparse_bias(
            detections=final,
            enabled=effective_low_branch_class_debias,
            y_advantage_margin=2.0,
            domain_branch_ratio=gen_profile.get("branch_ratio", None) if isinstance(gen_profile, dict) else None,
            moderate_branch_ratio_threshold=DEFAULT_GENERALIZATION_MODERATE_BRANCH_RATIO,
        )
    final = refine_ultra_sparse_t_vote_centers(
        detections=final,
        image=image,
        enabled=DEFAULT_GENERALIZATION_ULTRA_T_VOTE_SNAP,
        domain_branch_ratio=gen_profile.get("branch_ratio", None) if isinstance(gen_profile, dict) else None,
        moderate_branch_ratio_threshold=DEFAULT_GENERALIZATION_MODERATE_BRANCH_RATIO,
        line_threshold=line_threshold,
        inner_radius=graph_arm_inner_radius,
        outer_radius=effective_graph_outer_radius,
        min_arm_pixels=graph_min_arm_pixels,
        min_arm_span=graph_min_arm_span,
        t_min_largest_angle=graph_t_min_largest_angle,
        t_max_side_angle=graph_t_max_side_angle,
        ambiguity_margin=geometry_ambiguity_margin,
        classifier_mode=three_arm_classifier,
        ring_bridge_radius=effective_ring_gap_bridge_radius,
    )
    if effective_low_branch_final_merge_distance > 0:
        final = non_maximum_suppression(
            final,
            distance_threshold=max(1, int(effective_low_branch_final_merge_distance)),
            respect_label=False,
        )

    if gen_profile:
        if gate_stats is None:
            gate_stats = {}
        gate_stats["generalization_profile"] = gen_profile

    counts = count_by_type(final, labels)
    return final, counts, gate_stats


def run_detection_on_path(
    image_path,
    template_dir,
    output_image,
    output_json="",
    match_threshold=DEFAULT_MATCH_THRESHOLD,
    nms_distance=DEFAULT_NMS_DISTANCE,
    labels=None,
    allow_legacy_v=True,
    use_topology_gate=True,
    line_threshold=127,
    branch_degree=3,
    gate_radius=6,
    detection_mode=DEFAULT_DETECTION_MODE,
    graph_arm_inner_radius=DEFAULT_GRAPH_ARM_INNER_RADIUS,
    graph_arm_outer_radius=DEFAULT_GRAPH_ARM_OUTER_RADIUS,
    graph_min_arm_pixels=DEFAULT_GRAPH_MIN_ARM_PIXELS,
    graph_min_arm_span=DEFAULT_GRAPH_MIN_ARM_SPAN,
    graph_min_branch_component_area=DEFAULT_GRAPH_MIN_BRANCH_COMPONENT_AREA,
    graph_max_centers_per_component=DEFAULT_GRAPH_MAX_CENTERS_PER_COMPONENT,
    graph_ring_gap_bridge_radius=DEFAULT_GRAPH_RING_GAP_BRIDGE_RADIUS,
    graph_sparse_recovery=DEFAULT_GRAPH_SPARSE_RECOVERY,
    graph_sparse_recovery_support_radius=DEFAULT_GRAPH_SPARSE_RECOVERY_SUPPORT_RADIUS,
    graph_sparse_recovery_endpoint_radius=DEFAULT_GRAPH_SPARSE_RECOVERY_ENDPOINT_RADIUS,
    graph_sparse_recovery_max_component_area=DEFAULT_GRAPH_SPARSE_RECOVERY_MAX_COMPONENT_AREA,
    graph_sparse_recovery_min_score=DEFAULT_GRAPH_SPARSE_RECOVERY_MIN_SCORE,
    graph_junction_snap_repair=DEFAULT_GRAPH_JUNCTION_SNAP_REPAIR,
    graph_junction_snap_iters=DEFAULT_GRAPH_JUNCTION_SNAP_ITERS,
    graph_junction_snap_dist=DEFAULT_GRAPH_JUNCTION_SNAP_DIST,
    graph_junction_snap_min_dot=DEFAULT_GRAPH_JUNCTION_SNAP_MIN_DOT,
    graph_junction_snap_max_existing_fraction=DEFAULT_GRAPH_JUNCTION_SNAP_MAX_EXISTING_FRACTION,
    graph_t_min_largest_angle=DEFAULT_GRAPH_T_MIN_LARGEST_ANGLE,
    graph_t_max_side_angle=DEFAULT_GRAPH_T_MAX_SIDE_ANGLE,
    graph_center_mode=DEFAULT_GRAPH_CENTER_MODE,
    three_arm_classifier=DEFAULT_THREE_ARM_CLASSIFIER,
    hybrid_fusion_mode=DEFAULT_HYBRID_FUSION_MODE,
    geometry_ambiguity_margin=6.0,
    local_reclassify_t=DEFAULT_LOCAL_RECLASSIFY_T,
    local_reclassify_y_to_t=DEFAULT_LOCAL_RECLASSIFY_Y_TO_T,
    ty_multiradius_vote=DEFAULT_TY_MULTIRADIUS_VOTE,
    ty_vote_outer_radii=DEFAULT_TY_VOTE_OUTER_RADII,
    ty_vote_min_votes=DEFAULT_TY_VOTE_MIN_VOTES,
    ty_vote_margin=DEFAULT_TY_VOTE_MARGIN,
    ty_vote_require_non_ambiguous=DEFAULT_TY_VOTE_REQUIRE_NON_AMBIG,
    ty_feature_rules=DEFAULT_TY_FEATURE_RULES,
    ty_feature_rule_mode=DEFAULT_TY_FEATURE_RULE_MODE,
    tyx_structural_rules=DEFAULT_TYX_STRUCTURAL_RULES,
    tyx_structural_rule_mode=DEFAULT_TYX_STRUCTURAL_RULE_MODE,
    tyx_structural_ood_gate=DEFAULT_TYX_STRUCTURAL_OOD_GATE,
    tyx_structural_min_margin=DEFAULT_TYX_STRUCTURAL_MIN_MARGIN,
    x_consistency_reclassify=DEFAULT_X_CONSISTENCY_RECLASSIFY,
    x_consistency_outer_radii=DEFAULT_X_CONSISTENCY_OUTER_RADII,
    x_consistency_max_fourarm_votes=DEFAULT_X_CONSISTENCY_MAX_FOURARM,
    x_consistency_min_threearm_votes=DEFAULT_X_CONSISTENCY_MIN_THREEARM_VOTES,
    generalization_auto_scale=DEFAULT_GENERALIZATION_AUTO_SCALE,
    generalization_branch_ratio_threshold=DEFAULT_GENERALIZATION_BRANCH_RATIO_THRESHOLD,
    generalization_low_outer_radius=DEFAULT_GENERALIZATION_LOW_OUTER_RADIUS,
    generalization_low_nms_distance=DEFAULT_GENERALIZATION_LOW_NMS_DISTANCE,
    generalization_low_max_centers=DEFAULT_GENERALIZATION_LOW_MAX_CENTERS,
    generalization_low_ring_gap_bridge_radius=DEFAULT_GENERALIZATION_LOW_RING_GAP_BRIDGE_RADIUS,
    generalization_low_sparse_recovery=DEFAULT_GENERALIZATION_LOW_SPARSE_RECOVERY,
    generalization_low_sparse_endpoint_radius=DEFAULT_GENERALIZATION_LOW_SPARSE_ENDPOINT_RADIUS,
    generalization_low_sparse_min_score=DEFAULT_GENERALIZATION_LOW_SPARSE_MIN_SCORE,
    generalization_low_x_consistency_reclassify=DEFAULT_GENERALIZATION_LOW_X_CONSISTENCY_RECLASSIFY,
    generalization_low_final_merge_distance=DEFAULT_GENERALIZATION_LOW_FINAL_MERGE_DISTANCE,
    generalization_low_class_debias=DEFAULT_GENERALIZATION_LOW_CLASS_DEBIAS,
    preloaded_templates=None,
):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    detections, counts, gate_stats = detect_junctions(
        image=image,
        template_dir=template_dir,
        match_threshold=match_threshold,
        nms_distance=nms_distance,
        labels=labels,
        allow_legacy_v=allow_legacy_v,
        use_topology_gate=use_topology_gate,
        line_threshold=line_threshold,
        branch_degree=branch_degree,
        gate_radius=gate_radius,
        detection_mode=detection_mode,
        graph_arm_inner_radius=graph_arm_inner_radius,
        graph_arm_outer_radius=graph_arm_outer_radius,
        graph_min_arm_pixels=graph_min_arm_pixels,
        graph_min_arm_span=graph_min_arm_span,
        graph_min_branch_component_area=graph_min_branch_component_area,
        graph_max_centers_per_component=graph_max_centers_per_component,
        graph_ring_gap_bridge_radius=graph_ring_gap_bridge_radius,
        graph_sparse_recovery=graph_sparse_recovery,
        graph_sparse_recovery_support_radius=graph_sparse_recovery_support_radius,
        graph_sparse_recovery_endpoint_radius=graph_sparse_recovery_endpoint_radius,
        graph_sparse_recovery_max_component_area=graph_sparse_recovery_max_component_area,
        graph_sparse_recovery_min_score=graph_sparse_recovery_min_score,
        graph_junction_snap_repair=graph_junction_snap_repair,
        graph_junction_snap_iters=graph_junction_snap_iters,
        graph_junction_snap_dist=graph_junction_snap_dist,
        graph_junction_snap_min_dot=graph_junction_snap_min_dot,
        graph_junction_snap_max_existing_fraction=graph_junction_snap_max_existing_fraction,
        graph_t_min_largest_angle=graph_t_min_largest_angle,
        graph_t_max_side_angle=graph_t_max_side_angle,
        graph_center_mode=graph_center_mode,
        three_arm_classifier=three_arm_classifier,
        hybrid_fusion_mode=hybrid_fusion_mode,
        geometry_ambiguity_margin=geometry_ambiguity_margin,
        local_reclassify_t=local_reclassify_t,
        local_reclassify_y_to_t=local_reclassify_y_to_t,
        ty_multiradius_vote=ty_multiradius_vote,
        ty_vote_outer_radii=ty_vote_outer_radii,
        ty_vote_min_votes=ty_vote_min_votes,
        ty_vote_margin=ty_vote_margin,
        ty_vote_require_non_ambiguous=ty_vote_require_non_ambiguous,
        ty_feature_rules=ty_feature_rules,
        ty_feature_rule_mode=ty_feature_rule_mode,
        tyx_structural_rules=tyx_structural_rules,
        tyx_structural_rule_mode=tyx_structural_rule_mode,
        tyx_structural_ood_gate=tyx_structural_ood_gate,
        tyx_structural_min_margin=tyx_structural_min_margin,
        x_consistency_reclassify=x_consistency_reclassify,
        x_consistency_outer_radii=x_consistency_outer_radii,
        x_consistency_max_fourarm_votes=x_consistency_max_fourarm_votes,
        x_consistency_min_threearm_votes=x_consistency_min_threearm_votes,
        generalization_auto_scale=generalization_auto_scale,
        generalization_branch_ratio_threshold=generalization_branch_ratio_threshold,
        generalization_low_outer_radius=generalization_low_outer_radius,
        generalization_low_nms_distance=generalization_low_nms_distance,
        generalization_low_max_centers=generalization_low_max_centers,
        generalization_low_ring_gap_bridge_radius=generalization_low_ring_gap_bridge_radius,
        generalization_low_sparse_recovery=generalization_low_sparse_recovery,
        generalization_low_sparse_endpoint_radius=generalization_low_sparse_endpoint_radius,
        generalization_low_sparse_min_score=generalization_low_sparse_min_score,
        generalization_low_x_consistency_reclassify=generalization_low_x_consistency_reclassify,
        generalization_low_final_merge_distance=generalization_low_final_merge_distance,
        generalization_low_class_debias=generalization_low_class_debias,
        preloaded_templates=preloaded_templates,
    )
    visualize_detections(image, detections, output_image)

    payload = {
        "image_path": image_path,
        "template_dir": template_dir,
        "labels": labels or list(DEFAULT_LABELS),
        "match_threshold": float(match_threshold),
        "nms_distance": int(nms_distance),
        "allow_legacy_v": bool(allow_legacy_v),
        "use_topology_gate": bool(use_topology_gate),
        "line_threshold": int(line_threshold),
        "branch_degree": int(branch_degree),
        "gate_radius": int(gate_radius),
        "detection_mode": detection_mode,
        "graph_arm_inner_radius": int(graph_arm_inner_radius),
        "graph_arm_outer_radius": int(graph_arm_outer_radius),
        "graph_min_arm_pixels": int(graph_min_arm_pixels),
        "graph_min_arm_span": float(graph_min_arm_span),
        "graph_min_branch_component_area": int(graph_min_branch_component_area),
        "graph_max_centers_per_component": int(graph_max_centers_per_component),
        "graph_ring_gap_bridge_radius": int(graph_ring_gap_bridge_radius),
        "graph_sparse_recovery": bool(graph_sparse_recovery),
        "graph_sparse_recovery_support_radius": int(graph_sparse_recovery_support_radius),
        "graph_sparse_recovery_endpoint_radius": int(graph_sparse_recovery_endpoint_radius),
        "graph_sparse_recovery_max_component_area": int(graph_sparse_recovery_max_component_area),
        "graph_sparse_recovery_min_score": float(graph_sparse_recovery_min_score),
        "graph_junction_snap_repair": bool(graph_junction_snap_repair),
        "graph_junction_snap_iters": int(graph_junction_snap_iters),
        "graph_junction_snap_dist": int(graph_junction_snap_dist),
        "graph_junction_snap_min_dot": float(graph_junction_snap_min_dot),
        "graph_junction_snap_max_existing_fraction": float(graph_junction_snap_max_existing_fraction),
        "graph_t_min_largest_angle": float(graph_t_min_largest_angle),
        "graph_t_max_side_angle": float(graph_t_max_side_angle),
        "graph_center_mode": str(graph_center_mode),
        "three_arm_classifier": str(three_arm_classifier),
        "hybrid_fusion_mode": str(hybrid_fusion_mode),
        "geometry_ambiguity_margin": float(geometry_ambiguity_margin),
        "local_reclassify_t": bool(local_reclassify_t),
        "local_reclassify_y_to_t": bool(local_reclassify_y_to_t),
        "ty_multiradius_vote": bool(ty_multiradius_vote),
        "ty_vote_outer_radii": [int(v) for v in ty_vote_outer_radii],
        "ty_vote_min_votes": int(ty_vote_min_votes),
        "ty_vote_margin": int(ty_vote_margin),
        "ty_vote_require_non_ambiguous": bool(ty_vote_require_non_ambiguous),
        "ty_feature_rules": bool(ty_feature_rules),
        "ty_feature_rule_mode": str(ty_feature_rule_mode),
        "tyx_structural_rules": bool(tyx_structural_rules),
        "tyx_structural_rule_mode": str(tyx_structural_rule_mode),
        "tyx_structural_ood_gate": bool(tyx_structural_ood_gate),
        "tyx_structural_min_margin": float(tyx_structural_min_margin),
        "x_consistency_reclassify": bool(x_consistency_reclassify),
        "x_consistency_outer_radii": [int(v) for v in tuple(x_consistency_outer_radii)],
        "x_consistency_max_fourarm_votes": int(x_consistency_max_fourarm_votes),
        "x_consistency_min_threearm_votes": int(x_consistency_min_threearm_votes),
        "generalization_auto_scale": bool(generalization_auto_scale),
        "generalization_branch_ratio_threshold": float(generalization_branch_ratio_threshold),
        "generalization_low_outer_radius": int(generalization_low_outer_radius),
        "generalization_low_nms_distance": int(generalization_low_nms_distance),
        "generalization_low_max_centers": int(generalization_low_max_centers),
        "generalization_low_ring_gap_bridge_radius": int(generalization_low_ring_gap_bridge_radius),
        "generalization_low_sparse_recovery": bool(generalization_low_sparse_recovery),
        "generalization_low_sparse_endpoint_radius": int(generalization_low_sparse_endpoint_radius),
        "generalization_low_sparse_min_score": float(generalization_low_sparse_min_score),
        "generalization_low_x_consistency_reclassify": bool(generalization_low_x_consistency_reclassify),
        "generalization_low_final_merge_distance": int(generalization_low_final_merge_distance),
        "generalization_low_class_debias": bool(generalization_low_class_debias),
        "gate_stats": gate_stats,
        "counts": counts,
        "num_detections": len(detections),
        "detections": detections,
    }
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved detections JSON: {output_json}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Template-based junction detection (T/Y/X).")
    _here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--image-path",
        default=None,
        help="Input skeleton-like grayscale image path.",
    )
    parser.add_argument(
        "--template-dir",
        default=os.path.join(_here, "..", "data", "templates", "templates_tyx_exact"),
        help="Template root directory (default: ../data/templates/templates_tyx_exact relative to this script).",
    )
    parser.add_argument(
        "--output-image",
        default="detection_results.png",
        help="Output visualization path.",
    )
    parser.add_argument("--output-json", default="", help="Optional JSON output path.")
    parser.add_argument("--labels", default="T,Y,X", help="Comma-separated labels to detect.")
    parser.add_argument("--match-threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    parser.add_argument("--nms-distance", type=int, default=DEFAULT_NMS_DISTANCE)
    parser.add_argument("--allow-legacy-v", dest="allow_legacy_v", action="store_true")
    parser.add_argument("--no-legacy-v", dest="allow_legacy_v", action="store_false")
    parser.add_argument("--topology-gate", dest="use_topology_gate", action="store_true")
    parser.add_argument("--no-topology-gate", dest="use_topology_gate", action="store_false")
    parser.add_argument("--line-threshold", type=int, default=127)
    parser.add_argument("--branch-degree", type=int, default=3)
    parser.add_argument("--gate-radius", type=int, default=6)
    parser.add_argument("--detection-mode", default=DEFAULT_DETECTION_MODE, choices=list(DETECTION_MODES))
    parser.add_argument("--graph-arm-inner-radius", type=int, default=DEFAULT_GRAPH_ARM_INNER_RADIUS)
    parser.add_argument("--graph-arm-outer-radius", type=int, default=DEFAULT_GRAPH_ARM_OUTER_RADIUS)
    parser.add_argument("--graph-min-arm-pixels", type=int, default=DEFAULT_GRAPH_MIN_ARM_PIXELS)
    parser.add_argument("--graph-min-arm-span", type=float, default=DEFAULT_GRAPH_MIN_ARM_SPAN)
    parser.add_argument("--graph-min-branch-component-area", type=int, default=DEFAULT_GRAPH_MIN_BRANCH_COMPONENT_AREA)
    parser.add_argument("--graph-max-centers-per-component", type=int, default=DEFAULT_GRAPH_MAX_CENTERS_PER_COMPONENT)
    parser.add_argument("--graph-ring-gap-bridge-radius", type=int, default=DEFAULT_GRAPH_RING_GAP_BRIDGE_RADIUS)
    parser.add_argument("--graph-sparse-recovery", dest="graph_sparse_recovery", action="store_true")
    parser.add_argument("--no-graph-sparse-recovery", dest="graph_sparse_recovery", action="store_false")
    parser.add_argument("--graph-sparse-recovery-support-radius", type=int, default=DEFAULT_GRAPH_SPARSE_RECOVERY_SUPPORT_RADIUS)
    parser.add_argument("--graph-sparse-recovery-endpoint-radius", type=int, default=DEFAULT_GRAPH_SPARSE_RECOVERY_ENDPOINT_RADIUS)
    parser.add_argument("--graph-sparse-recovery-max-component-area", type=int, default=DEFAULT_GRAPH_SPARSE_RECOVERY_MAX_COMPONENT_AREA)
    parser.add_argument("--graph-sparse-recovery-min-score", type=float, default=DEFAULT_GRAPH_SPARSE_RECOVERY_MIN_SCORE)
    parser.add_argument("--graph-junction-snap-repair", dest="graph_junction_snap_repair", action="store_true")
    parser.add_argument("--no-graph-junction-snap-repair", dest="graph_junction_snap_repair", action="store_false")
    parser.add_argument("--graph-junction-snap-iters", type=int, default=DEFAULT_GRAPH_JUNCTION_SNAP_ITERS)
    parser.add_argument("--graph-junction-snap-dist", type=int, default=DEFAULT_GRAPH_JUNCTION_SNAP_DIST)
    parser.add_argument("--graph-junction-snap-min-dot", type=float, default=DEFAULT_GRAPH_JUNCTION_SNAP_MIN_DOT)
    parser.add_argument(
        "--graph-junction-snap-max-existing-fraction",
        type=float,
        default=DEFAULT_GRAPH_JUNCTION_SNAP_MAX_EXISTING_FRACTION,
    )
    parser.add_argument("--graph-t-min-largest-angle", type=float, default=DEFAULT_GRAPH_T_MIN_LARGEST_ANGLE)
    parser.add_argument("--graph-t-max-side-angle", type=float, default=DEFAULT_GRAPH_T_MAX_SIDE_ANGLE)
    parser.add_argument("--graph-center-mode", default=DEFAULT_GRAPH_CENTER_MODE, choices=list(GRAPH_CENTER_MODES))
    parser.add_argument("--three-arm-classifier", default=DEFAULT_THREE_ARM_CLASSIFIER, choices=list(THREE_ARM_CLASSIFIER_MODES))
    parser.add_argument("--hybrid-fusion-mode", default=DEFAULT_HYBRID_FUSION_MODE, choices=list(HYBRID_FUSION_MODES))
    parser.add_argument("--geometry-ambiguity-margin", type=float, default=6.0)
    parser.add_argument("--local-reclassify-t", dest="local_reclassify_t", action="store_true")
    parser.add_argument("--no-local-reclassify-t", dest="local_reclassify_t", action="store_false")
    parser.add_argument("--local-reclassify-y-to-t", dest="local_reclassify_y_to_t", action="store_true")
    parser.add_argument("--no-local-reclassify-y-to-t", dest="local_reclassify_y_to_t", action="store_false")
    parser.add_argument("--ty-multiradius-vote", dest="ty_multiradius_vote", action="store_true")
    parser.add_argument("--no-ty-multiradius-vote", dest="ty_multiradius_vote", action="store_false")
    parser.add_argument(
        "--ty-vote-outers",
        default=",".join(str(v) for v in DEFAULT_TY_VOTE_OUTER_RADII),
        help="Comma-separated outer radii for post-fusion T/Y vote relabeling.",
    )
    parser.add_argument("--ty-vote-min-votes", type=int, default=DEFAULT_TY_VOTE_MIN_VOTES)
    parser.add_argument("--ty-vote-margin", type=int, default=DEFAULT_TY_VOTE_MARGIN)
    parser.add_argument("--ty-vote-require-non-ambiguous", dest="ty_vote_require_non_ambiguous", action="store_true")
    parser.add_argument("--no-ty-vote-require-non-ambiguous", dest="ty_vote_require_non_ambiguous", action="store_false")
    parser.add_argument("--ty-feature-rules", dest="ty_feature_rules", action="store_true")
    parser.add_argument("--no-ty-feature-rules", dest="ty_feature_rules", action="store_false")
    parser.add_argument("--ty-feature-rule-mode", default=DEFAULT_TY_FEATURE_RULE_MODE, choices=list(TY_FEATURE_RULE_MODES))
    parser.add_argument("--tyx-structural-rules", dest="tyx_structural_rules", action="store_true")
    parser.add_argument("--no-tyx-structural-rules", dest="tyx_structural_rules", action="store_false")
    parser.add_argument("--tyx-structural-rule-mode", default=DEFAULT_TYX_STRUCTURAL_RULE_MODE, choices=list(TYX_STRUCTURAL_RULE_MODES))
    parser.add_argument("--tyx-structural-ood-gate", dest="tyx_structural_ood_gate", action="store_true")
    parser.add_argument("--no-tyx-structural-ood-gate", dest="tyx_structural_ood_gate", action="store_false")
    parser.add_argument("--tyx-structural-min-margin", type=float, default=DEFAULT_TYX_STRUCTURAL_MIN_MARGIN)
    parser.add_argument("--generalization-auto-scale", dest="generalization_auto_scale", action="store_true")
    parser.add_argument("--no-generalization-auto-scale", dest="generalization_auto_scale", action="store_false")
    parser.add_argument("--generalization-branch-ratio-threshold", type=float, default=DEFAULT_GENERALIZATION_BRANCH_RATIO_THRESHOLD)
    parser.add_argument("--generalization-low-outer-radius", type=int, default=DEFAULT_GENERALIZATION_LOW_OUTER_RADIUS)
    parser.add_argument("--generalization-low-nms-distance", type=int, default=DEFAULT_GENERALIZATION_LOW_NMS_DISTANCE)
    parser.add_argument("--generalization-low-max-centers", type=int, default=DEFAULT_GENERALIZATION_LOW_MAX_CENTERS)
    parser.add_argument("--generalization-low-ring-gap-bridge-radius", type=int, default=DEFAULT_GENERALIZATION_LOW_RING_GAP_BRIDGE_RADIUS)
    parser.add_argument("--generalization-low-sparse-recovery", dest="generalization_low_sparse_recovery", action="store_true")
    parser.add_argument("--no-generalization-low-sparse-recovery", dest="generalization_low_sparse_recovery", action="store_false")
    parser.add_argument("--generalization-low-sparse-endpoint-radius", type=int, default=DEFAULT_GENERALIZATION_LOW_SPARSE_ENDPOINT_RADIUS)
    parser.add_argument("--generalization-low-sparse-min-score", type=float, default=DEFAULT_GENERALIZATION_LOW_SPARSE_MIN_SCORE)
    parser.add_argument("--x-consistency-reclassify", dest="x_consistency_reclassify", action="store_true")
    parser.add_argument("--no-x-consistency-reclassify", dest="x_consistency_reclassify", action="store_false")
    parser.add_argument("--x-consistency-outers", default=",".join(str(v) for v in DEFAULT_X_CONSISTENCY_OUTER_RADII))
    parser.add_argument("--x-consistency-max-fourarm-votes", type=int, default=DEFAULT_X_CONSISTENCY_MAX_FOURARM)
    parser.add_argument("--x-consistency-min-threearm-votes", type=int, default=DEFAULT_X_CONSISTENCY_MIN_THREEARM_VOTES)
    parser.add_argument("--generalization-low-x-consistency-reclassify", dest="generalization_low_x_consistency_reclassify", action="store_true")
    parser.add_argument("--no-generalization-low-x-consistency-reclassify", dest="generalization_low_x_consistency_reclassify", action="store_false")
    parser.add_argument("--generalization-low-final-merge-distance", type=int, default=DEFAULT_GENERALIZATION_LOW_FINAL_MERGE_DISTANCE)
    parser.add_argument("--generalization-low-class-debias", dest="generalization_low_class_debias", action="store_true")
    parser.add_argument("--no-generalization-low-class-debias", dest="generalization_low_class_debias", action="store_false")
    parser.set_defaults(allow_legacy_v=True, use_topology_gate=True)
    parser.set_defaults(
        local_reclassify_t=DEFAULT_LOCAL_RECLASSIFY_T,
        local_reclassify_y_to_t=DEFAULT_LOCAL_RECLASSIFY_Y_TO_T,
        ty_multiradius_vote=DEFAULT_TY_MULTIRADIUS_VOTE,
        ty_vote_require_non_ambiguous=DEFAULT_TY_VOTE_REQUIRE_NON_AMBIG,
        ty_feature_rules=DEFAULT_TY_FEATURE_RULES,
        tyx_structural_rules=DEFAULT_TYX_STRUCTURAL_RULES,
        tyx_structural_ood_gate=DEFAULT_TYX_STRUCTURAL_OOD_GATE,
        generalization_auto_scale=DEFAULT_GENERALIZATION_AUTO_SCALE,
        graph_sparse_recovery=DEFAULT_GRAPH_SPARSE_RECOVERY,
        graph_junction_snap_repair=DEFAULT_GRAPH_JUNCTION_SNAP_REPAIR,
        generalization_low_sparse_recovery=DEFAULT_GENERALIZATION_LOW_SPARSE_RECOVERY,
        x_consistency_reclassify=DEFAULT_X_CONSISTENCY_RECLASSIFY,
        generalization_low_x_consistency_reclassify=DEFAULT_GENERALIZATION_LOW_X_CONSISTENCY_RECLASSIFY,
        generalization_low_class_debias=DEFAULT_GENERALIZATION_LOW_CLASS_DEBIAS,
    )
    args = parser.parse_args()

    labels = parse_labels(args.labels)
    result = run_detection_on_path(
        image_path=args.image_path,
        template_dir=args.template_dir,
        output_image=args.output_image,
        output_json=args.output_json,
        match_threshold=args.match_threshold,
        nms_distance=max(1, args.nms_distance),
        labels=labels,
        allow_legacy_v=args.allow_legacy_v,
        use_topology_gate=args.use_topology_gate,
        line_threshold=int(np.clip(args.line_threshold, 1, 254)),
        branch_degree=max(3, args.branch_degree),
        gate_radius=max(0, args.gate_radius),
        detection_mode=args.detection_mode,
        graph_arm_inner_radius=max(1, args.graph_arm_inner_radius),
        graph_arm_outer_radius=max(2, args.graph_arm_outer_radius),
        graph_min_arm_pixels=max(1, args.graph_min_arm_pixels),
        graph_min_arm_span=max(0.0, float(args.graph_min_arm_span)),
        graph_min_branch_component_area=max(1, args.graph_min_branch_component_area),
        graph_max_centers_per_component=max(1, int(args.graph_max_centers_per_component)),
        graph_ring_gap_bridge_radius=max(0, int(args.graph_ring_gap_bridge_radius)),
        graph_sparse_recovery=bool(args.graph_sparse_recovery),
        graph_sparse_recovery_support_radius=max(1, int(args.graph_sparse_recovery_support_radius)),
        graph_sparse_recovery_endpoint_radius=max(0, int(args.graph_sparse_recovery_endpoint_radius)),
        graph_sparse_recovery_max_component_area=max(1, int(args.graph_sparse_recovery_max_component_area)),
        graph_sparse_recovery_min_score=max(0.0, float(args.graph_sparse_recovery_min_score)),
        graph_junction_snap_repair=bool(args.graph_junction_snap_repair),
        graph_junction_snap_iters=max(0, int(args.graph_junction_snap_iters)),
        graph_junction_snap_dist=max(1, int(args.graph_junction_snap_dist)),
        graph_junction_snap_min_dot=float(np.clip(args.graph_junction_snap_min_dot, -1.0, 1.0)),
        graph_junction_snap_max_existing_fraction=float(np.clip(args.graph_junction_snap_max_existing_fraction, 0.0, 1.0)),
        graph_t_min_largest_angle=float(args.graph_t_min_largest_angle),
        graph_t_max_side_angle=float(args.graph_t_max_side_angle),
        graph_center_mode=str(args.graph_center_mode),
        three_arm_classifier=str(args.three_arm_classifier),
        hybrid_fusion_mode=str(args.hybrid_fusion_mode),
        geometry_ambiguity_margin=max(0.0, float(args.geometry_ambiguity_margin)),
        local_reclassify_t=bool(args.local_reclassify_t),
        local_reclassify_y_to_t=bool(args.local_reclassify_y_to_t),
        ty_multiradius_vote=bool(args.ty_multiradius_vote),
        ty_vote_outer_radii=parse_int_tuple(args.ty_vote_outers),
        ty_vote_min_votes=max(1, int(args.ty_vote_min_votes)),
        ty_vote_margin=max(0, int(args.ty_vote_margin)),
        ty_vote_require_non_ambiguous=bool(args.ty_vote_require_non_ambiguous),
        ty_feature_rules=bool(args.ty_feature_rules),
        ty_feature_rule_mode=str(args.ty_feature_rule_mode),
        tyx_structural_rules=bool(args.tyx_structural_rules),
        tyx_structural_rule_mode=str(args.tyx_structural_rule_mode),
        tyx_structural_ood_gate=bool(args.tyx_structural_ood_gate),
        tyx_structural_min_margin=max(0.0, float(args.tyx_structural_min_margin)),
        x_consistency_reclassify=bool(args.x_consistency_reclassify),
        x_consistency_outer_radii=parse_int_tuple(args.x_consistency_outers),
        x_consistency_max_fourarm_votes=max(0, int(args.x_consistency_max_fourarm_votes)),
        x_consistency_min_threearm_votes=max(1, int(args.x_consistency_min_threearm_votes)),
        generalization_auto_scale=bool(args.generalization_auto_scale),
        generalization_branch_ratio_threshold=max(0.0, float(args.generalization_branch_ratio_threshold)),
        generalization_low_outer_radius=max(2, int(args.generalization_low_outer_radius)),
        generalization_low_nms_distance=max(1, int(args.generalization_low_nms_distance)),
        generalization_low_max_centers=max(1, int(args.generalization_low_max_centers)),
        generalization_low_ring_gap_bridge_radius=max(0, int(args.generalization_low_ring_gap_bridge_radius)),
        generalization_low_sparse_recovery=bool(args.generalization_low_sparse_recovery),
        generalization_low_sparse_endpoint_radius=max(0, int(args.generalization_low_sparse_endpoint_radius)),
        generalization_low_sparse_min_score=max(0.0, float(args.generalization_low_sparse_min_score)),
        generalization_low_x_consistency_reclassify=bool(args.generalization_low_x_consistency_reclassify),
        generalization_low_final_merge_distance=max(0, int(args.generalization_low_final_merge_distance)),
        generalization_low_class_debias=bool(args.generalization_low_class_debias),
    )

    print("\n--- Final Results ---")
    for label in labels:
        print(f"{label} junctions found: {result['counts'].get(label, 0)}")


if __name__ == "__main__":
    main()
