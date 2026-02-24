#pragma IMAGINET_INCLUDES_BEGIN
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#pragma IMAGINET_INCLUDES_END

#pragma IMAGINET_FRAGMENT_BEGIN "object_detection_counter_init_f32"

// Initialize counter state - called once during Init phase
static int object_detection_counter_init_f32(void* restrict counter_state) {
    ObjectDetectionCounterState* state = (ObjectDetectionCounterState*)counter_state;
    
    memset(state->tracked_objects, 0, sizeof(state->tracked_objects));
    state->global_frame_counter = 0;
    state->global_in_count = 0;
    state->global_out_count = 0;
    state->tracking_initialized = 1;
    state->last_reset_hour = -1;
    state->last_reset_check = 0;
    
    return 0;
}

#pragma IMAGINET_FRAGMENT_END

#pragma IMAGINET_FRAGMENT_BEGIN "object_detection_counter_f32"

//============================================================================
// CONSTANTS AND CONFIGURATION
//============================================================================

#define MAX_TRACKED_OBJECTS 200
#define POSITION_HISTORY_SIZE 10
#define EPSILON 1e-6f
#define CLEANUP_INTERVAL_FRAMES 10
#define TRACKER_EXPIRY_FRAMES 30

//============================================================================
// ENUMERATIONS
//============================================================================

typedef enum {
    DIRECTION_FROM_TOP_LEFT = 0,      // Objects entering from top or left count as IN
    DIRECTION_FROM_TOP_RIGHT = 1,     // Objects entering from top or right count as IN
    DIRECTION_FROM_BOTTOM_LEFT = 2,   // Objects entering from bottom or left count as IN
    DIRECTION_FROM_BOTTOM_RIGHT = 3   // Objects entering from bottom or right count as IN
} Direction;

typedef enum {
    CROSSING_NONE = 0,
    CROSSING_IN = 1,
    CROSSING_OUT = 2
} CrossingType;

// Region state tracking for objects
typedef enum {
    REGION_STATE_OUTSIDE = 0,
    REGION_STATE_INSIDE = 1,
    REGION_STATE_ENTERED_FROM_LEFT = 2,
    REGION_STATE_ENTERED_FROM_RIGHT = 3,
    REGION_STATE_ENTERED_FROM_TOP = 4,
    REGION_STATE_ENTERED_FROM_BOTTOM = 5
} RegionState;

//============================================================================
// DATA STRUCTURES
//============================================================================

// Structure to store object position history
typedef struct {
    float x_history[POSITION_HISTORY_SIZE];
    float y_history[POSITION_HISTORY_SIZE];
    int history_count;
    int track_id;
    int last_seen_frame;
    int is_active;
    RegionState region_state;  // Track object's state relative to counting region
} ObjectTracker;

// Structure to store counting region definition
typedef struct {
    float x1, y1, x2, y2;
    float min_x, max_x, min_y, max_y;  // Pre-calculated bounds
    Direction in_direction;
} CountingRegion;

// Object detection counter state structure (replaces static variables)
typedef struct {
    ObjectTracker tracked_objects[MAX_TRACKED_OBJECTS];
    int global_frame_counter;
    int global_in_count;
    int global_out_count;
    int tracking_initialized;
    int last_reset_hour;
    time_t last_reset_check;
} ObjectDetectionCounterState;

//============================================================================
// UTILITY FUNCTIONS
//============================================================================

// Helper function to get tracked detection value from input tensor
static float odc_get_detection_value(const float* restrict tracked_detections, 
                                    int max_detections, int confidence_count, 
                                    int conf_idx, int det_idx) {
    return tracked_detections[conf_idx * max_detections + det_idx];
}

// Initialize counting region with pre-calculated bounds
static void odc_init_region(CountingRegion* region, float x1, float y1, float x2, float y2, 
                           int in_direction) {
    region->x1 = x1;
    region->y1 = y1;
    region->x2 = x2;
    region->y2 = y2;
    region->min_x = fminf(x1, x2);
    region->max_x = fmaxf(x1, x2);
    region->min_y = fminf(y1, y2);
    region->max_y = fmaxf(y1, y2);
    region->in_direction = (Direction)in_direction;
}

// Find existing object tracker by track_id
static ObjectTracker* odc_find_tracker(ObjectDetectionCounterState* state, int track_id) {
    for (int i = 0; i < MAX_TRACKED_OBJECTS; i++) {
        if (state->tracked_objects[i].is_active && state->tracked_objects[i].track_id == track_id) {
            return &state->tracked_objects[i];
        }
    }
    return NULL;
}

// Find or create object tracker for track_id
static ObjectTracker* odc_get_tracker(ObjectDetectionCounterState* state, int track_id) {
    ObjectTracker* existing = odc_find_tracker(state, track_id);
    if (existing) {
        return existing;
    }
    
    // Find empty slot
    for (int i = 0; i < MAX_TRACKED_OBJECTS; i++) {
        if (!state->tracked_objects[i].is_active) {
            ObjectTracker* tracker = &state->tracked_objects[i];
            memset(tracker, 0, sizeof(ObjectTracker));
            tracker->track_id = track_id;
            tracker->is_active = 1;
            return tracker;
        }
    }
    
    return NULL; // No available slots
}

// Add position to object tracker history
static void odc_add_position(ObjectDetectionCounterState* state, ObjectTracker* tracker, float x, float y) {
    if (tracker->history_count < POSITION_HISTORY_SIZE) {
        tracker->x_history[tracker->history_count] = x;
        tracker->y_history[tracker->history_count] = y;
        tracker->history_count++;
    } else {
        // Shift history and add new position
        for (int i = 0; i < POSITION_HISTORY_SIZE - 1; i++) {
            tracker->x_history[i] = tracker->x_history[i + 1];
            tracker->y_history[i] = tracker->y_history[i + 1];
        }
        tracker->x_history[POSITION_HISTORY_SIZE - 1] = x;
        tracker->y_history[POSITION_HISTORY_SIZE - 1] = y;
    }
    tracker->last_seen_frame = state->global_frame_counter;
}

//============================================================================
// GEOMETRIC CALCULATIONS
//============================================================================

// Check if point is inside rectangular region using pre-calculated bounds
static int odc_point_in_rectangle(float x, float y, const CountingRegion* region) {
    return (x >= region->min_x && x <= region->max_x && 
            y >= region->min_y && y <= region->max_y);
}

// Determine which side of a rectangular region a point is closest to
static RegionState odc_get_entry_side(float x, float y, const CountingRegion* region) {
    // Calculate distances to each side using pre-calculated bounds
    const float dist_left = fabsf(x - region->min_x);
    const float dist_right = fabsf(x - region->max_x);
    const float dist_top = fabsf(y - region->min_y);
    const float dist_bottom = fabsf(y - region->max_y);
    
    // Find the minimum distance
    const float min_dist = fminf(fminf(dist_left, dist_right), fminf(dist_top, dist_bottom));
    
    if (min_dist == dist_left) return REGION_STATE_ENTERED_FROM_LEFT;
    if (min_dist == dist_right) return REGION_STATE_ENTERED_FROM_RIGHT;
    if (min_dist == dist_top) return REGION_STATE_ENTERED_FROM_TOP;
    return REGION_STATE_ENTERED_FROM_BOTTOM;
}

// Check if object traversal is valid (crosses completely through region)
static int odc_is_valid_traversal(RegionState entry_side, RegionState exit_side) {
    return ((entry_side == REGION_STATE_ENTERED_FROM_LEFT && exit_side == REGION_STATE_ENTERED_FROM_RIGHT) ||
            (entry_side == REGION_STATE_ENTERED_FROM_RIGHT && exit_side == REGION_STATE_ENTERED_FROM_LEFT) ||
            (entry_side == REGION_STATE_ENTERED_FROM_TOP && exit_side == REGION_STATE_ENTERED_FROM_BOTTOM) ||
            (entry_side == REGION_STATE_ENTERED_FROM_BOTTOM && exit_side == REGION_STATE_ENTERED_FROM_TOP));
}

// Determine if traversal direction should count as "IN" based on entry side and configured direction
static int odc_is_in_direction(RegionState entry_side, Direction in_direction) {
    switch (in_direction) {
        case DIRECTION_FROM_TOP_LEFT:
            return (entry_side == REGION_STATE_ENTERED_FROM_TOP || 
                    entry_side == REGION_STATE_ENTERED_FROM_LEFT);
        
        case DIRECTION_FROM_TOP_RIGHT:
            return (entry_side == REGION_STATE_ENTERED_FROM_TOP || 
                    entry_side == REGION_STATE_ENTERED_FROM_RIGHT);
        
        case DIRECTION_FROM_BOTTOM_LEFT:
            return (entry_side == REGION_STATE_ENTERED_FROM_BOTTOM || 
                    entry_side == REGION_STATE_ENTERED_FROM_LEFT);
        
        case DIRECTION_FROM_BOTTOM_RIGHT:
            return (entry_side == REGION_STATE_ENTERED_FROM_BOTTOM || 
                    entry_side == REGION_STATE_ENTERED_FROM_RIGHT);
        
        default:
            return 0; // Unknown direction
    }
}

//============================================================================
// CROSSING DETECTION
//============================================================================

static CrossingType odc_detect_crossing(ObjectTracker* tracker, const CountingRegion* region) {
    if (tracker->history_count < 2) {
        return CROSSING_NONE;
    }
    
    // Get last two positions
    const float prev_x = tracker->x_history[tracker->history_count - 2];
    const float prev_y = tracker->y_history[tracker->history_count - 2];
    const float curr_x = tracker->x_history[tracker->history_count - 1];
    const float curr_y = tracker->y_history[tracker->history_count - 1];
    
    const int prev_in_box = odc_point_in_rectangle(prev_x, prev_y, region);
    const int curr_in_box = odc_point_in_rectangle(curr_x, curr_y, region);
    
    // Handle region state transitions
    if (prev_in_box && !curr_in_box) {
        // Exiting region - check if we have a complete traversal
        if (tracker->region_state >= REGION_STATE_ENTERED_FROM_LEFT) {
            const RegionState exit_side = odc_get_entry_side(curr_x, curr_y, region);
            const RegionState entry_side = tracker->region_state;
            
            // Only count complete traversals across the region
            if (odc_is_valid_traversal(entry_side, exit_side)) {
                // Determine in/out direction based on entry side and configured direction
                const int counts_as_in = odc_is_in_direction(entry_side, region->in_direction);
                
                tracker->region_state = REGION_STATE_OUTSIDE;
                return counts_as_in ? CROSSING_IN : CROSSING_OUT;
            }
        }
        tracker->region_state = REGION_STATE_OUTSIDE;
    }
    else if (!prev_in_box && curr_in_box) {
        // Entering region - record entry side
        tracker->region_state = odc_get_entry_side(prev_x, prev_y, region);
    }
    
    return CROSSING_NONE;
}

// Clean up old inactive trackers
static void odc_cleanup_trackers(ObjectDetectionCounterState* state) {
    for (int i = 0; i < MAX_TRACKED_OBJECTS; i++) {
        if (state->tracked_objects[i].is_active) {
            // Remove trackers that haven't been seen for many frames
            if (state->global_frame_counter - state->tracked_objects[i].last_seen_frame > TRACKER_EXPIRY_FRAMES) {
                state->tracked_objects[i].is_active = 0;
            }
        }
    }
}

// Check if counters should be reset based on daily reset time
static void odc_check_daily_reset(ObjectDetectionCounterState* state, int reset_hour) {
    // If reset is disabled, do nothing
    if (reset_hour == -1) {
        return;
    }
    
    time_t current_time;
    time(&current_time);
    
    // Only check once per minute to avoid excessive calculations
    if (current_time - state->last_reset_check < 60) {
        return;
    }
    
    state->last_reset_check = current_time;
    
    // Get current local time
    struct tm* local_time = localtime(&current_time);
    if (!local_time) {
        return; // Error getting local time
    }
    
    int current_hour = local_time->tm_hour;
    
    // Check if we've crossed into the reset hour
    if (current_hour == reset_hour && state->last_reset_hour != reset_hour) {
        // Reset counters
        state->global_in_count = 0;
        state->global_out_count = 0;
        
        // Clear all active trackers to avoid counting objects that were 
        // already in the scene before reset
        for (int i = 0; i < MAX_TRACKED_OBJECTS; i++) {
            state->tracked_objects[i].is_active = 0;
        }
    }
    
    state->last_reset_hour = current_hour;
}

//============================================================================
// MAIN FUNCTION
//============================================================================

static void object_detection_counter_f32(
    const float* restrict tracked_detections,
    int8_t* restrict counter_state_bytes,
    int32_t* restrict in_count,
    int32_t* restrict out_count,
    int32_t* restrict total_count,
    int max_detections,
    int confidence_count,
    float region_x1,
    float region_y1,
    float region_x2,
    float region_y2,
    int in_direction,
    int reset_hour)
{
    // Cast the byte buffer to our state structure
    ObjectDetectionCounterState* state = (ObjectDetectionCounterState*)counter_state_bytes;
    
    // Check for daily reset
    odc_check_daily_reset(state, reset_hour);
    
    // Set up counting region with pre-calculated bounds
    CountingRegion region;
    odc_init_region(&region, region_x1, region_y1, region_x2, region_y2, in_direction);
    
    // Increment frame counter
    state->global_frame_counter++;
    
    // Process each tracked detection
    for (int i = 0; i < max_detections; i++) {
        // Get basic detection data (center_x, center_y, width, height)
        const float center_x = odc_get_detection_value(tracked_detections, max_detections, confidence_count, 0, i);
        const float center_y = odc_get_detection_value(tracked_detections, max_detections, confidence_count, 1, i);
        const float width = odc_get_detection_value(tracked_detections, max_detections, confidence_count, 2, i);
        const float height = odc_get_detection_value(tracked_detections, max_detections, confidence_count, 3, i);
        
        // Skip empty detections
        if (width <= 0 || height <= 0) continue;
        
        // Get track ID
        const int track_id = (int)odc_get_detection_value(tracked_detections, max_detections, confidence_count, confidence_count - 2, i);
        
        // Skip detections without valid track ID
        if (track_id <= 0) continue;
        
        // Get or create object tracker
        ObjectTracker* tracker = odc_get_tracker(state, track_id);
        if (!tracker) continue; // No available slots
        
        // Add current position to history
        odc_add_position(state, tracker, center_x, center_y);
        
        // Check for crossing
        const CrossingType crossing = odc_detect_crossing(tracker, &region);
        
        // Update counters based on crossing type
        if (crossing == CROSSING_IN) {
            state->global_in_count++;
        } else if (crossing == CROSSING_OUT) {
            state->global_out_count++;
        }
    }
    
    // Clean up inactive trackers periodically
    if (state->global_frame_counter % CLEANUP_INTERVAL_FRAMES == 0) {
        odc_cleanup_trackers(state);
    }
    
    // Set output values
    *in_count = state->global_in_count;
    *out_count = state->global_out_count;
    *total_count = state->global_in_count + state->global_out_count;
}

#pragma IMAGINET_FRAGMENT_END

#pragma IMAGINET_FRAGMENT_BEGIN "object_detection_counter_init_i8"

// Initialize counter state for int8 version - called once during Init phase
static int object_detection_counter_init_i8(void* restrict counter_state) {
    ObjectDetectionCounterState_i8* state = (ObjectDetectionCounterState_i8*)counter_state;
    
    memset(state->tracked_objects, 0, sizeof(state->tracked_objects));
    state->global_frame_counter = 0;
    state->global_in_count = 0;
    state->global_out_count = 0;
    state->tracking_initialized = 1;
    state->last_reset_hour = -1;
    state->last_reset_check = 0;
    
    return 0; // Success
}

#pragma IMAGINET_FRAGMENT_END

#pragma IMAGINET_FRAGMENT_BEGIN "object_detection_counter_i8"

//============================================================================
// CONSTANTS AND CONFIGURATION (INT8)
//============================================================================

#define MAX_TRACKED_OBJECTS_I8 200
#define POSITION_HISTORY_SIZE_I8 10
#define EPSILON_I8 1e-6f
#define CLEANUP_INTERVAL_FRAMES_I8 10
#define TRACKER_EXPIRY_FRAMES_I8 30

//============================================================================
// ENUMERATIONS (INT8)
//============================================================================

typedef enum {
    DIRECTION_FROM_TOP_LEFT_I8 = 0,      // Objects entering from top or left count as IN
    DIRECTION_FROM_TOP_RIGHT_I8 = 1,     // Objects entering from top or right count as IN
    DIRECTION_FROM_BOTTOM_LEFT_I8 = 2,   // Objects entering from bottom or left count as IN
    DIRECTION_FROM_BOTTOM_RIGHT_I8 = 3   // Objects entering from bottom or right count as IN
} Direction_i8;

typedef enum {
    CROSSING_NONE_I8 = 0,
    CROSSING_IN_I8 = 1,
    CROSSING_OUT_I8 = 2
} CrossingType_i8;

// Region state tracking for objects
typedef enum {
    REGION_STATE_OUTSIDE_I8 = 0,
    REGION_STATE_INSIDE_I8 = 1,
    REGION_STATE_ENTERED_FROM_LEFT_I8 = 2,
    REGION_STATE_ENTERED_FROM_RIGHT_I8 = 3,
    REGION_STATE_ENTERED_FROM_TOP_I8 = 4,
    REGION_STATE_ENTERED_FROM_BOTTOM_I8 = 5
} RegionState_i8;

//============================================================================
// DATA STRUCTURES (INT8)
//============================================================================

// Structure to store object position history (int8 version)
typedef struct {
    float x_history[POSITION_HISTORY_SIZE_I8];
    float y_history[POSITION_HISTORY_SIZE_I8];
    int history_count;
    int track_id;
    int last_seen_frame;
    int is_active;
    RegionState_i8 region_state;  // Track object's state relative to counting region
} ObjectTracker_i8;

// Structure to store counting region definition (int8 version)
typedef struct {
    float x1, y1, x2, y2;
    float min_x, max_x, min_y, max_y;  // Pre-calculated bounds
    Direction_i8 in_direction;
} CountingRegion_i8;

// Object detection counter state structure (int8 version)
typedef struct {
    ObjectTracker_i8 tracked_objects[MAX_TRACKED_OBJECTS_I8];
    int global_frame_counter;
    int global_in_count;
    int global_out_count;
    int tracking_initialized;
    int last_reset_hour;
    time_t last_reset_check;
} ObjectDetectionCounterState_i8;

//============================================================================
// INT8 CONVERSION UTILITIES
//============================================================================

// Helper function to convert int8 to float for reading detection values
static inline float odc_i8_to_float(int8_t value) {
    return ((float)value + 128.0f) / 255.0f;
}

//============================================================================
// UTILITY FUNCTIONS (INT8)
//============================================================================

// Helper function to get tracked detection value from int8 input tensor
static float odc_get_detection_value_i8(const int8_t* restrict tracked_detections, 
                                        int max_detections, int confidence_count, 
                                        int conf_idx, int det_idx) {
    return odc_i8_to_float(tracked_detections[conf_idx * max_detections + det_idx]);
}

// Initialize counting region with pre-calculated bounds (int8 version)
static void odc_init_region_i8(CountingRegion_i8* region, float x1, float y1, float x2, float y2, 
                               int in_direction) {
    region->x1 = x1;
    region->y1 = y1;
    region->x2 = x2;
    region->y2 = y2;
    region->min_x = fminf(x1, x2);
    region->max_x = fmaxf(x1, x2);
    region->min_y = fminf(y1, y2);
    region->max_y = fmaxf(y1, y2);
    region->in_direction = (Direction_i8)in_direction;
}

// Find existing object tracker by track_id (int8 version)
static ObjectTracker_i8* odc_find_tracker_i8(ObjectDetectionCounterState_i8* state, int track_id) {
    for (int i = 0; i < MAX_TRACKED_OBJECTS_I8; i++) {
        if (state->tracked_objects[i].is_active && state->tracked_objects[i].track_id == track_id) {
            return &state->tracked_objects[i];
        }
    }
    return NULL;
}

// Find or create object tracker for track_id (int8 version)
static ObjectTracker_i8* odc_get_tracker_i8(ObjectDetectionCounterState_i8* state, int track_id) {
    ObjectTracker_i8* existing = odc_find_tracker_i8(state, track_id);
    if (existing) {
        return existing;
    }
    
    // Find empty slot
    for (int i = 0; i < MAX_TRACKED_OBJECTS_I8; i++) {
        if (!state->tracked_objects[i].is_active) {
            ObjectTracker_i8* tracker = &state->tracked_objects[i];
            memset(tracker, 0, sizeof(ObjectTracker_i8));
            tracker->track_id = track_id;
            tracker->is_active = 1;
            return tracker;
        }
    }
    
    return NULL; // No available slots
}

// Add position to object tracker history (int8 version)
static void odc_add_position_i8(ObjectDetectionCounterState_i8* state, ObjectTracker_i8* tracker, float x, float y) {
    if (tracker->history_count < POSITION_HISTORY_SIZE_I8) {
        tracker->x_history[tracker->history_count] = x;
        tracker->y_history[tracker->history_count] = y;
        tracker->history_count++;
    } else {
        // Shift history and add new position
        for (int i = 0; i < POSITION_HISTORY_SIZE_I8 - 1; i++) {
            tracker->x_history[i] = tracker->x_history[i + 1];
            tracker->y_history[i] = tracker->y_history[i + 1];
        }
        tracker->x_history[POSITION_HISTORY_SIZE_I8 - 1] = x;
        tracker->y_history[POSITION_HISTORY_SIZE_I8 - 1] = y;
    }
    tracker->last_seen_frame = state->global_frame_counter;
}

//============================================================================
// GEOMETRIC CALCULATIONS (INT8)
//============================================================================

// Check if point is inside rectangular region using pre-calculated bounds (int8 version)
static int odc_point_in_rectangle_i8(float x, float y, const CountingRegion_i8* region) {
    return (x >= region->min_x && x <= region->max_x && 
            y >= region->min_y && y <= region->max_y);
}

// Determine which side of a rectangular region a point is closest to (int8 version)
static RegionState_i8 odc_get_entry_side_i8(float x, float y, const CountingRegion_i8* region) {
    // Calculate distances to each side using pre-calculated bounds
    const float dist_left = fabsf(x - region->min_x);
    const float dist_right = fabsf(x - region->max_x);
    const float dist_top = fabsf(y - region->min_y);
    const float dist_bottom = fabsf(y - region->max_y);
    
    // Find the minimum distance
    const float min_dist = fminf(fminf(dist_left, dist_right), fminf(dist_top, dist_bottom));
    
    if (min_dist == dist_left) return REGION_STATE_ENTERED_FROM_LEFT_I8;
    if (min_dist == dist_right) return REGION_STATE_ENTERED_FROM_RIGHT_I8;
    if (min_dist == dist_top) return REGION_STATE_ENTERED_FROM_TOP_I8;
    return REGION_STATE_ENTERED_FROM_BOTTOM_I8;
}

// Check if object traversal is valid (crosses completely through region) (int8 version)
static int odc_is_valid_traversal_i8(RegionState_i8 entry_side, RegionState_i8 exit_side) {
    return ((entry_side == REGION_STATE_ENTERED_FROM_LEFT_I8 && exit_side == REGION_STATE_ENTERED_FROM_RIGHT_I8) ||
            (entry_side == REGION_STATE_ENTERED_FROM_RIGHT_I8 && exit_side == REGION_STATE_ENTERED_FROM_LEFT_I8) ||
            (entry_side == REGION_STATE_ENTERED_FROM_TOP_I8 && exit_side == REGION_STATE_ENTERED_FROM_BOTTOM_I8) ||
            (entry_side == REGION_STATE_ENTERED_FROM_BOTTOM_I8 && exit_side == REGION_STATE_ENTERED_FROM_TOP_I8));
}

// Determine if traversal direction should count as "IN" based on entry side and configured direction (int8 version)
static int odc_is_in_direction_i8(RegionState_i8 entry_side, Direction_i8 in_direction) {
    switch (in_direction) {
        case DIRECTION_FROM_TOP_LEFT_I8:
            return (entry_side == REGION_STATE_ENTERED_FROM_TOP_I8 || 
                    entry_side == REGION_STATE_ENTERED_FROM_LEFT_I8);
        
        case DIRECTION_FROM_TOP_RIGHT_I8:
            return (entry_side == REGION_STATE_ENTERED_FROM_TOP_I8 || 
                    entry_side == REGION_STATE_ENTERED_FROM_RIGHT_I8);
        
        case DIRECTION_FROM_BOTTOM_LEFT_I8:
            return (entry_side == REGION_STATE_ENTERED_FROM_BOTTOM_I8 || 
                    entry_side == REGION_STATE_ENTERED_FROM_LEFT_I8);
        
        case DIRECTION_FROM_BOTTOM_RIGHT_I8:
            return (entry_side == REGION_STATE_ENTERED_FROM_BOTTOM_I8 || 
                    entry_side == REGION_STATE_ENTERED_FROM_RIGHT_I8);
        
        default:
            return 0; // Unknown direction
    }
}

//============================================================================
// CROSSING DETECTION (INT8)
//============================================================================

static CrossingType_i8 odc_detect_crossing_i8(ObjectTracker_i8* tracker, const CountingRegion_i8* region) {
    if (tracker->history_count < 2) {
        return CROSSING_NONE_I8;
    }
    
    // Get last two positions
    const float prev_x = tracker->x_history[tracker->history_count - 2];
    const float prev_y = tracker->y_history[tracker->history_count - 2];
    const float curr_x = tracker->x_history[tracker->history_count - 1];
    const float curr_y = tracker->y_history[tracker->history_count - 1];
    
    const int prev_in_box = odc_point_in_rectangle_i8(prev_x, prev_y, region);
    const int curr_in_box = odc_point_in_rectangle_i8(curr_x, curr_y, region);
    
    // Handle region state transitions
    if (prev_in_box && !curr_in_box) {
        // Exiting region - check if we have a complete traversal
        if (tracker->region_state >= REGION_STATE_ENTERED_FROM_LEFT_I8) {
            const RegionState_i8 exit_side = odc_get_entry_side_i8(curr_x, curr_y, region);
            const RegionState_i8 entry_side = tracker->region_state;
            
            // Only count complete traversals across the region
            if (odc_is_valid_traversal_i8(entry_side, exit_side)) {
                // Determine in/out direction based on entry side and configured direction
                const int counts_as_in = odc_is_in_direction_i8(entry_side, region->in_direction);
                
                tracker->region_state = REGION_STATE_OUTSIDE_I8;
                return counts_as_in ? CROSSING_IN_I8 : CROSSING_OUT_I8;
            }
        }
        tracker->region_state = REGION_STATE_OUTSIDE_I8;
    }
    else if (!prev_in_box && curr_in_box) {
        // Entering region - record entry side
        tracker->region_state = odc_get_entry_side_i8(prev_x, prev_y, region);
    }
    
    return CROSSING_NONE_I8;
}

// Clean up old inactive trackers (int8 version)
static void odc_cleanup_trackers_i8(ObjectDetectionCounterState_i8* state) {
    for (int i = 0; i < MAX_TRACKED_OBJECTS_I8; i++) {
        if (state->tracked_objects[i].is_active) {
            // Remove trackers that haven't been seen for many frames
            if (state->global_frame_counter - state->tracked_objects[i].last_seen_frame > TRACKER_EXPIRY_FRAMES_I8) {
                state->tracked_objects[i].is_active = 0;
            }
        }
    }
}

// Check if counters should be reset based on daily reset time (int8 version)
static void odc_check_daily_reset_i8(ObjectDetectionCounterState_i8* state, int reset_hour) {
    // If reset is disabled, do nothing
    if (reset_hour < 0 || reset_hour > 23) {
        return;
    }
    
    time_t current_time;
    time(&current_time);
    
    // Only check once per minute to avoid excessive calculations
    if (current_time - state->last_reset_check < 60) {
        return;
    }
    
    state->last_reset_check = current_time;
    
    // Get current local time
    struct tm* local_time = localtime(&current_time);
    if (!local_time) {
        return; // Error getting local time
    }
    
    int current_hour = local_time->tm_hour;
    
    // Check if we've crossed into the reset hour
    if (current_hour == reset_hour && state->last_reset_hour != reset_hour) {
        // Reset counters
        state->global_in_count = 0;
        state->global_out_count = 0;
        
        // Clear all active trackers to avoid counting objects that were 
        // already in the scene before reset
        for (int i = 0; i < MAX_TRACKED_OBJECTS_I8; i++) {
            state->tracked_objects[i].is_active = 0;
        }
    }
    
    state->last_reset_hour = current_hour;
}

//============================================================================
// MAIN FUNCTION (INT8)
//============================================================================

static void object_detection_counter_i8(
    const int8_t* restrict tracked_detections,
    int8_t* restrict counter_state_bytes,
    int32_t* restrict in_count,
    int32_t* restrict out_count,
    int32_t* restrict total_count,
    int max_detections,
    int confidence_count,
    float region_x1,
    float region_y1,
    float region_x2,
    float region_y2,
    int in_direction,
    int reset_hour)
{
    // Cast the byte buffer to our state structure
    ObjectDetectionCounterState_i8* state = (ObjectDetectionCounterState_i8*)counter_state_bytes;
    
    // Check for daily reset
    odc_check_daily_reset_i8(state, reset_hour);
    
    // Set up counting region with pre-calculated bounds
    CountingRegion_i8 region;
    odc_init_region_i8(&region, region_x1, region_y1, region_x2, region_y2, in_direction);
    
    // Increment frame counter
    state->global_frame_counter++;
    
    // Process each tracked detection
    for (int i = 0; i < max_detections; i++) {
        // Get basic detection data (center_x, center_y, width, height)
        const float center_x = odc_get_detection_value_i8(tracked_detections, max_detections, confidence_count, 0, i);
        const float center_y = odc_get_detection_value_i8(tracked_detections, max_detections, confidence_count, 1, i);
        const float width = odc_get_detection_value_i8(tracked_detections, max_detections, confidence_count, 2, i);
        const float height = odc_get_detection_value_i8(tracked_detections, max_detections, confidence_count, 3, i);
        
        // Skip empty detections
        if (width <= 0 || height <= 0) continue;
        
        // Get track ID - convert from normalized [0,1] range to actual ID
        const float track_id_normalized = odc_get_detection_value_i8(tracked_detections, max_detections, confidence_count, confidence_count - 2, i);
        const int track_id = (int)(track_id_normalized * 127.0f);  // Denormalize from int8 storage
        
        // Skip detections without valid track ID
        if (track_id <= 0) continue;
        
        // Get or create object tracker
        ObjectTracker_i8* tracker = odc_get_tracker_i8(state, track_id);
        if (!tracker) continue; // No available slots
        
        // Add current position to history
        odc_add_position_i8(state, tracker, center_x, center_y);
        
        // Check for crossing
        const CrossingType_i8 crossing = odc_detect_crossing_i8(tracker, &region);
        
        // Update counters based on crossing type
        if (crossing == CROSSING_IN_I8) {
            state->global_in_count++;
        } else if (crossing == CROSSING_OUT_I8) {
            state->global_out_count++;
        }
    }
    
    // Clean up inactive trackers periodically
    if (state->global_frame_counter % CLEANUP_INTERVAL_FRAMES_I8 == 0) {
        odc_cleanup_trackers_i8(state);
    }
    
    // Set output values
    *in_count = state->global_in_count;
    *out_count = state->global_out_count;
    *total_count = state->global_in_count + state->global_out_count;
}

#pragma IMAGINET_FRAGMENT_END 