// Generic numeric floors
static const float SMALL_VALUE = 1e-6;
static const float TINY_VALUE = 1e-8;
static const float RAY_DENOMINATOR_FLOOR = 1e-10;
static const float POSITIVE_CONIC_FLOOR = 1e-10;
static const float CONIC_DETERMINANT_FLOOR = 1e-12;
static const float CONIC_AXIS_FLOOR = 1e-20;
static const float CONIC_INTERSECTION_SLACK = 1e-5;
static const float DEPTH_VISIBILITY_FLOOR = 1e-4;
static const float GRADIENT_LENGTH_FLOOR = 1e-4;
static const float SOUTH_POLE_DOT_FLOOR = -0.9999999;

// Rendering constants
static const float SPLAT_PIXEL_CLAMP_PX = 0.75;
static const float MIN_SCREEN_RADIUS_PX = 1.0;
static const float MIN_BBOX_EXTENT_PX = 1e-4;
static const float BBOX_PAD_PX = 1.0;
static const float ELLIPSE_SAFETY_SCALE = 1.0;
static const float ELLIPSE_RADIUS_PAD_PX = 1.0;
static const float ELLIPSE_EPS = 1e-6;
static const float OUTPUT_GAMMA = 2.2;
static const float SOFTPLUS_INPUT_LIMIT = 30.0;

// Debug visualization tuning
static const float MIN_DEBUG_THICKNESS_PX = 0.25;
static const float DEBUG_GRAD_NORM_FLOOR = 1e-12;
static const float LOG2_TO_LOG10 = 0.30102999566;
static const float DEBUG_GRAD_THRESHOLD_MIN_SCALE = 0.001;
static const float DEBUG_GRAD_THRESHOLD_MAX_SCALE = 10.0;

// Misc math helpers
static const float TAU_RADIANS = 6.28318530718;
static const float LOG_TWO = 0.69314718056;
static const float RNG_OPEN01_SCALE = 1.0 / 4294967297.0;
