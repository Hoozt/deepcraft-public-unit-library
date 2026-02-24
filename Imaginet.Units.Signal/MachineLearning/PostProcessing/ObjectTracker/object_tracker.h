#pragma IMAGINET_INCLUDES_BEGIN
#include <stdint.h>
#include <math.h>
#pragma IMAGINET_INCLUDES_END

#pragma IMAGINET_FRAGMENT_BEGIN "object_tracker_init_f32"

// Initialize tracker state - called once during Init phase
static int object_tracker_init_f32(void* restrict tracker_state, int max_tracks) {
    ObjectTrackerState* state = (ObjectTrackerState*)tracker_state;
    
    for (int i = 0; i < TRACKS_ARRAY_SIZE; i++) {
        state->tracks[i].active = 0;
        state->tracks[i].track_id = -1;
        state->tracks[i].age = 0;
        state->tracks[i].hits = 0;
    }
    state->next_track_id = 1;
    state->tracker_initialized = 1;
    
    return 0; // Success
}

#pragma IMAGINET_FRAGMENT_END

#pragma IMAGINET_FRAGMENT_BEGIN "object_tracker_f32"

//============================================================================
// CONSTANTS AND CONFIGURATION
//============================================================================

#define TRACKS_ARRAY_SIZE 50  // Maximum compile-time array size
#define TRACKING_CONFIDENCE_DECAY 0.9f
#define NEW_TRACK_CONFIDENCE 1.0f
#define DETECTION_DATA_SIZE 6  // x, y, w, h, confidence, class_id
#define MAX_TRACK_ID 127       // Maximum track ID before wrapping (1-127 for int8_t embedded systems)

//============================================================================
// DATA STRUCTURES
//============================================================================

// Track structure for maintaining object state across frames
typedef struct {
    float x, y, w, h;           // Bounding box (center x, y, width, height)
    float confidence;           // Detection confidence
    int class_id;              // Object class identifier
    int track_id;              // Unique track identifier
    int age;                   // Frames since last detection
    int hits;                  // Total number of detections
    int active;                // Whether track is currently active
    float tracking_confidence; // Confidence in tracking quality (IoU-based)
} ObjectTrack;

// Object tracker state structure (replaces static variables)
typedef struct {
    ObjectTrack tracks[TRACKS_ARRAY_SIZE];
    int next_track_id;
    int tracker_initialized;
} ObjectTrackerState;

//============================================================================
// UTILITY FUNCTIONS
//============================================================================

// Helper function to get detection data from input tensor
static float get_detection_value(const float* restrict detections, int max_detections, int confidence_count, int conf_idx, int det_idx) {
    return detections[conf_idx * max_detections + det_idx];
}

// Helper function to set output data
static void set_output_value(float* restrict output, int max_detections, int output_conf_count, int conf_idx, int det_idx, float value) {
    output[conf_idx * max_detections + det_idx] = value;
}

// Calculate Intersection over Union (IoU) between two bounding boxes
static float calculate_iou(float x1, float y1, float w1, float h1, 
                          float x2, float y2, float w2, float h2) {
    // Convert center coordinates to corner coordinates
    const float left1 = x1 - w1 * 0.5f;
    const float top1 = y1 - h1 * 0.5f;
    const float right1 = x1 + w1 * 0.5f;
    const float bottom1 = y1 + h1 * 0.5f;
    
    const float left2 = x2 - w2 * 0.5f;
    const float top2 = y2 - h2 * 0.5f;
    const float right2 = x2 + w2 * 0.5f;
    const float bottom2 = y2 + h2 * 0.5f;
    
    // Calculate intersection bounds
    const float inter_left = fmaxf(left1, left2);
    const float inter_top = fmaxf(top1, top2);
    const float inter_right = fminf(right1, right2);
    const float inter_bottom = fminf(bottom1, bottom2);
    
    // Check if there's no intersection
    if (inter_right <= inter_left || inter_bottom <= inter_top) {
        return 0.0f;
    }
    
    // Calculate areas
    const float inter_area = (inter_right - inter_left) * (inter_bottom - inter_top);
    const float area1 = w1 * h1;
    const float area2 = w2 * h2;
    const float union_area = area1 + area2 - inter_area;
    
    return union_area > 0 ? inter_area / union_area : 0.0f;
}

//============================================================================
// TRACKER MANAGEMENT
//============================================================================

//============================================================================
// DETECTION PROCESSING
//============================================================================

// Extract detections from input tensor (filtering already done upstream)
static int extract_detections(const float* restrict detections, int max_detections, 
                             int confidence_count, float detection_data[][DETECTION_DATA_SIZE]) {
    int valid_detections = 0;
    
    for (int i = 0; i < max_detections; i++) {
        float confidence = 0.0f;
        int class_id = -1;
        
        // Find the class with highest confidence (skip first 4 values: x, y, w, h)
        for (int j = 4; j < confidence_count; j++) {
            const float class_conf = get_detection_value(detections, max_detections, confidence_count, j, i);
            if (class_conf > confidence) {
                confidence = class_conf;
                class_id = j - 4;
            }
        }
        
        // Process all detections (filtering already done upstream)
        if (confidence > 0.0f || class_id >= 0) {  // Only skip completely empty detections
            detection_data[valid_detections][0] = get_detection_value(detections, max_detections, confidence_count, 0, i); // x
            detection_data[valid_detections][1] = get_detection_value(detections, max_detections, confidence_count, 1, i); // y
            detection_data[valid_detections][2] = get_detection_value(detections, max_detections, confidence_count, 2, i); // w
            detection_data[valid_detections][3] = get_detection_value(detections, max_detections, confidence_count, 3, i); // h
            detection_data[valid_detections][4] = confidence;
            detection_data[valid_detections][5] = (float)class_id;
            valid_detections++;
        }
    }
    
    return valid_detections;
}

//============================================================================
// TRACKING ALGORITHMS
//============================================================================

// Perform data association between detections and existing tracks
static void perform_tracking_association(ObjectTrackerState* state, const float detection_data[][DETECTION_DATA_SIZE], 
                                        int valid_detections, int max_tracks, float tracking_threshold,
                                        int* detection_matched, int* track_matched) {
    // Initialize matching arrays
    for (int i = 0; i < valid_detections; i++) detection_matched[i] = 0;
    for (int i = 0; i < max_tracks; i++) track_matched[i] = 0;
    
    // Greedy assignment based on IoU similarity
    for (int d = 0; d < valid_detections; d++) {
        if (detection_matched[d]) continue;
        
        float best_iou = 0.0f;
        int best_track = -1;
        
        for (int t = 0; t < max_tracks; t++) {
            if (!state->tracks[t].active || track_matched[t]) continue;
            
            // Only match detections of the same class
            if (state->tracks[t].class_id != (int)detection_data[d][5]) continue;
            
            const float iou = calculate_iou(
                detection_data[d][0], detection_data[d][1], detection_data[d][2], detection_data[d][3],
                state->tracks[t].x, state->tracks[t].y, state->tracks[t].w, state->tracks[t].h
            );
            
            if (iou > best_iou && iou > tracking_threshold) {
                best_iou = iou;
                best_track = t;
            }
        }
        
        if (best_track >= 0) {
            // Update existing track with new detection
            state->tracks[best_track].x = detection_data[d][0];
            state->tracks[best_track].y = detection_data[d][1];
            state->tracks[best_track].w = detection_data[d][2];
            state->tracks[best_track].h = detection_data[d][3];
            state->tracks[best_track].confidence = detection_data[d][4];
            state->tracks[best_track].age = 0;
            state->tracks[best_track].hits++;
            state->tracks[best_track].tracking_confidence = best_iou;
            
            detection_matched[d] = 1;
            track_matched[best_track] = 1;
        }
    }
}

// Create new tracks for unmatched detections
static void create_new_tracks(ObjectTrackerState* state, const float detection_data[][DETECTION_DATA_SIZE], 
                             int valid_detections, int max_tracks, const int* detection_matched) {
    for (int d = 0; d < valid_detections; d++) {
        if (detection_matched[d]) continue;
        
        // Find empty track slot
        int empty_slot = -1;
        for (int t = 0; t < max_tracks; t++) {
            if (!state->tracks[t].active) {
                empty_slot = t;
                break;
            }
        }
        
        if (empty_slot >= 0) {
            // Assign track ID using circular counter (1 to MAX_TRACK_ID)
            int new_track_id = state->next_track_id;
            state->next_track_id++;
            if (state->next_track_id > MAX_TRACK_ID) {
                state->next_track_id = 1;  // Wrap around to 1
            }
            
            // Initialize new track
            state->tracks[empty_slot].x = detection_data[d][0];
            state->tracks[empty_slot].y = detection_data[d][1];
            state->tracks[empty_slot].w = detection_data[d][2];
            state->tracks[empty_slot].h = detection_data[d][3];
            state->tracks[empty_slot].confidence = detection_data[d][4];
            state->tracks[empty_slot].class_id = (int)detection_data[d][5];
            state->tracks[empty_slot].track_id = new_track_id;
            state->tracks[empty_slot].age = 0;
            state->tracks[empty_slot].hits = 1;
            state->tracks[empty_slot].active = 1;
            state->tracks[empty_slot].tracking_confidence = NEW_TRACK_CONFIDENCE;
        }
    }
}

// Age unmatched tracks and remove expired ones
static void update_track_lifecycle(ObjectTrackerState* state, int max_tracks, int max_age, const int* track_matched) {
    for (int t = 0; t < max_tracks; t++) {
        if (!state->tracks[t].active) continue;
        
        if (!track_matched[t]) {
            state->tracks[t].age++;
            state->tracks[t].tracking_confidence *= TRACKING_CONFIDENCE_DECAY;
            
            if (state->tracks[t].age > max_age) {
                // Clear all track history
                state->tracks[t].active = 0;
                state->tracks[t].track_id = -1;
                state->tracks[t].age = 0;
                state->tracks[t].hits = 0;
                state->tracks[t].x = 0.0f;
                state->tracks[t].y = 0.0f;
                state->tracks[t].w = 0.0f;
                state->tracks[t].h = 0.0f;
                state->tracks[t].confidence = 0.0f;
                state->tracks[t].class_id = -1;
                state->tracks[t].tracking_confidence = 0.0f;
            }
        }
    }
}

// Generate output tensor with tracked detections
static void generate_tracking_output(ObjectTrackerState* state, float* restrict output, int max_detections, 
                                    int confidence_count, int max_tracks, int min_hits) {
    int output_count = 0;
    const int output_conf_count = confidence_count + 2; // +2 for track_id and tracking_confidence
    
    // Initialize output buffer to zeros
    for (int i = 0; i < max_detections * output_conf_count; i++) {
        output[i] = 0.0f;
    }
    
    // Output confirmed tracks
    for (int t = 0; t < max_tracks && output_count < max_detections; t++) {
        if (!state->tracks[t].active || state->tracks[t].hits < min_hits) continue;
        
        // Copy bounding box coordinates
        set_output_value(output, max_detections, output_conf_count, 0, output_count, state->tracks[t].x);
        set_output_value(output, max_detections, output_conf_count, 1, output_count, state->tracks[t].y);
        set_output_value(output, max_detections, output_conf_count, 2, output_count, state->tracks[t].w);
        set_output_value(output, max_detections, output_conf_count, 3, output_count, state->tracks[t].h);
        
        // Set class confidence scores
        for (int j = 4; j < confidence_count; j++) {
            const float conf_value = (j - 4 == state->tracks[t].class_id) ? state->tracks[t].confidence : 0.0f;
            set_output_value(output, max_detections, output_conf_count, j, output_count, conf_value);
        }
        
        // Add tracking metadata
        set_output_value(output, max_detections, output_conf_count, confidence_count, output_count, (float)state->tracks[t].track_id);
        set_output_value(output, max_detections, output_conf_count, confidence_count + 1, output_count, state->tracks[t].tracking_confidence);
        
        output_count++;
    }
}

//============================================================================
// MAIN TRACKING FUNCTION
//============================================================================

// Main object tracking function - maintains object identities across frames
static void object_tracker_f32(
    const float* restrict detections,
    int8_t* restrict tracker_state_bytes,
    float* restrict output,
    int max_detections,
    int confidence_count,
    float tracking_threshold,
    int max_tracks,
    int max_age,
    int min_hits)
{
    // Cast the byte buffer to our state structure
    ObjectTrackerState* state = (ObjectTrackerState*)tracker_state_bytes;
    
    // Step 1: Extract detections from input tensor
    float detection_data[max_detections][DETECTION_DATA_SIZE];
    const int valid_detections = extract_detections(detections, max_detections, confidence_count, detection_data);
    
    // Step 2: Associate detections with existing tracks
    int detection_matched[max_detections];
    int track_matched[max_tracks];
    perform_tracking_association(state, detection_data, valid_detections, max_tracks, tracking_threshold, 
                                detection_matched, track_matched);
    
    // Step 3: Create new tracks for unmatched detections
    create_new_tracks(state, detection_data, valid_detections, max_tracks, detection_matched);
    
    // Step 4: Age unmatched tracks and remove expired ones
    update_track_lifecycle(state, max_tracks, max_age, track_matched);
    
    // Step 5: Generate output with tracked detections
    generate_tracking_output(state, output, max_detections, confidence_count, max_tracks, min_hits);
}

#pragma IMAGINET_FRAGMENT_END

#pragma IMAGINET_FRAGMENT_BEGIN "object_tracker_init_i8"

// Initialize tracker state for int8 version - called once during Init phase
static int object_tracker_init_i8(void* restrict tracker_state, int max_tracks) {
    ObjectTrackerState_i8* state = (ObjectTrackerState_i8*)tracker_state;
    
    for (int i = 0; i < TRACKS_ARRAY_SIZE_I8; i++) {
        state->tracks[i].active = 0;
        state->tracks[i].track_id = -1;
        state->tracks[i].age = 0;
        state->tracks[i].hits = 0;
    }
    state->next_track_id = 1;
    state->tracker_initialized = 1;
    
    return 0; // Success
}

#pragma IMAGINET_FRAGMENT_END

#pragma IMAGINET_FRAGMENT_BEGIN "object_tracker_i8"

//============================================================================
// CONSTANTS AND CONFIGURATION (INT8)
//============================================================================

#define TRACKS_ARRAY_SIZE_I8 50  // Maximum compile-time array size
#define TRACKING_CONFIDENCE_DECAY_I8 0.9f
#define NEW_TRACK_CONFIDENCE_I8 1.0f
#define DETECTION_DATA_SIZE_I8 6  // x, y, w, h, confidence, class_id
#define MAX_TRACK_ID_I8 127       // Maximum track ID before wrapping (1-127 for int8_t embedded systems)

//============================================================================
// DATA STRUCTURES (INT8)
//============================================================================

// Track structure for maintaining object state across frames (int8 version)
typedef struct {
    float x, y, w, h;           // Bounding box (center x, y, width, height)
    float confidence;           // Detection confidence
    int class_id;              // Object class identifier
    int track_id;              // Unique track identifier
    int age;                   // Frames since last detection
    int hits;                  // Total number of detections
    int active;                // Whether track is currently active
    float tracking_confidence; // Confidence in tracking quality (IoU-based)
} ObjectTrack_i8;

// Object tracker state structure (int8 version)
typedef struct {
    ObjectTrack_i8 tracks[TRACKS_ARRAY_SIZE_I8];
    int next_track_id;
    int tracker_initialized;
} ObjectTrackerState_i8;

//============================================================================
// INT8 CONVERSION UTILITIES
//============================================================================

// Helper function to convert int8 to float for reading detection values
static inline float get_detection_value_i8(const int8_t* restrict detections, int max_detections, int confidence_count, int conf_idx, int det_idx) {
    return ((float)detections[conf_idx * max_detections + det_idx] + 128.0f) / 255.0f;
}

// Helper function to convert float to int8 for output values
static inline int8_t float_to_int8(float value) {
    int8_t int8_val = (int8_t)(value * 255.0f - 128.0f);
    if (int8_val > 127) int8_val = 127;
    else if (int8_val < -128) int8_val = -128;
    return int8_val;
}

// Helper function to set output data in int8 format
static void set_output_value_i8(int8_t* restrict output, int max_detections, int output_conf_count, int conf_idx, int det_idx, float value) {
    output[conf_idx * max_detections + det_idx] = float_to_int8(value);
}

//============================================================================
// UTILITY FUNCTIONS (INT8)
//============================================================================

// Calculate Intersection over Union (IoU) between two bounding boxes (int8 version)
static float calculate_iou_i8(float x1, float y1, float w1, float h1, 
                               float x2, float y2, float w2, float h2) {
    // Convert center coordinates to corner coordinates
    const float left1 = x1 - w1 * 0.5f;
    const float top1 = y1 - h1 * 0.5f;
    const float right1 = x1 + w1 * 0.5f;
    const float bottom1 = y1 + h1 * 0.5f;
    
    const float left2 = x2 - w2 * 0.5f;
    const float top2 = y2 - h2 * 0.5f;
    const float right2 = x2 + w2 * 0.5f;
    const float bottom2 = y2 + h2 * 0.5f;
    
    // Calculate intersection bounds
    const float inter_left = fmaxf(left1, left2);
    const float inter_top = fmaxf(top1, top2);
    const float inter_right = fminf(right1, right2);
    const float inter_bottom = fminf(bottom1, bottom2);
    
    // Check if there's no intersection
    if (inter_right <= inter_left || inter_bottom <= inter_top) {
        return 0.0f;
    }
    
    // Calculate areas
    const float inter_area = (inter_right - inter_left) * (inter_bottom - inter_top);
    const float area1 = w1 * h1;
    const float area2 = w2 * h2;
    const float union_area = area1 + area2 - inter_area;
    
    return union_area > 0 ? inter_area / union_area : 0.0f;
}

//============================================================================
// INT8 DETECTION PROCESSING
//============================================================================

// Extract detections from int8 input tensor (filtering already done upstream)
static int extract_detections_i8(const int8_t* restrict detections, int max_detections, 
                                 int confidence_count, float detection_data[][DETECTION_DATA_SIZE_I8]) {
    int valid_detections = 0;
    
    for (int i = 0; i < max_detections; i++) {
        float confidence = 0.0f;
        int class_id = -1;
        
        // Find the class with highest confidence (skip first 4 values: x, y, w, h)
        for (int j = 4; j < confidence_count; j++) {
            const float class_conf = get_detection_value_i8(detections, max_detections, confidence_count, j, i);
            if (class_conf > confidence) {
                confidence = class_conf;
                class_id = j - 4;
            }
        }
        
        // Process all detections (filtering already done upstream)
        if (confidence > 0.0f || class_id >= 0) {  // Only skip completely empty detections
            detection_data[valid_detections][0] = get_detection_value_i8(detections, max_detections, confidence_count, 0, i); // x
            detection_data[valid_detections][1] = get_detection_value_i8(detections, max_detections, confidence_count, 1, i); // y
            detection_data[valid_detections][2] = get_detection_value_i8(detections, max_detections, confidence_count, 2, i); // w
            detection_data[valid_detections][3] = get_detection_value_i8(detections, max_detections, confidence_count, 3, i); // h
            detection_data[valid_detections][4] = confidence;
            detection_data[valid_detections][5] = (float)class_id;
            valid_detections++;
        }
    }
    
    return valid_detections;
}

//============================================================================
// INT8 TRACKING ALGORITHMS
//============================================================================

// Perform data association between detections and existing tracks (int8 version)
static void perform_tracking_association_i8(ObjectTrackerState_i8* state, const float detection_data[][DETECTION_DATA_SIZE_I8], 
                                            int valid_detections, int max_tracks, float tracking_threshold,
                                            int* detection_matched, int* track_matched) {
    // Initialize matching arrays
    for (int i = 0; i < valid_detections; i++) detection_matched[i] = 0;
    for (int i = 0; i < max_tracks; i++) track_matched[i] = 0;
    
    // Greedy assignment based on IoU similarity
    for (int d = 0; d < valid_detections; d++) {
        if (detection_matched[d]) continue;
        
        float best_iou = 0.0f;
        int best_track = -1;
        
        for (int t = 0; t < max_tracks; t++) {
            if (!state->tracks[t].active || track_matched[t]) continue;
            
            // Only match detections of the same class
            if (state->tracks[t].class_id != (int)detection_data[d][5]) continue;
            
            const float iou = calculate_iou_i8(
                detection_data[d][0], detection_data[d][1], detection_data[d][2], detection_data[d][3],
                state->tracks[t].x, state->tracks[t].y, state->tracks[t].w, state->tracks[t].h
            );
            
            if (iou > best_iou && iou > tracking_threshold) {
                best_iou = iou;
                best_track = t;
            }
        }
        
        if (best_track >= 0) {
            // Update existing track with new detection
            state->tracks[best_track].x = detection_data[d][0];
            state->tracks[best_track].y = detection_data[d][1];
            state->tracks[best_track].w = detection_data[d][2];
            state->tracks[best_track].h = detection_data[d][3];
            state->tracks[best_track].confidence = detection_data[d][4];
            state->tracks[best_track].age = 0;
            state->tracks[best_track].hits++;
            state->tracks[best_track].tracking_confidence = best_iou;
            
            detection_matched[d] = 1;
            track_matched[best_track] = 1;
        }
    }
}

// Create new tracks for unmatched detections (int8 version)
static void create_new_tracks_i8(ObjectTrackerState_i8* state, const float detection_data[][DETECTION_DATA_SIZE_I8], 
                                 int valid_detections, int max_tracks, const int* detection_matched) {
    for (int d = 0; d < valid_detections; d++) {
        if (detection_matched[d]) continue;
        
        // Find empty track slot
        int empty_slot = -1;
        for (int t = 0; t < max_tracks; t++) {
            if (!state->tracks[t].active) {
                empty_slot = t;
                break;
            }
        }
        
        if (empty_slot >= 0) {
            // Assign track ID using circular counter (1 to MAX_TRACK_ID)
            int new_track_id = state->next_track_id;
            state->next_track_id++;
            if (state->next_track_id > MAX_TRACK_ID_I8) {
                state->next_track_id = 1;  // Wrap around to 1
            }
            
            // Initialize new track
            state->tracks[empty_slot].x = detection_data[d][0];
            state->tracks[empty_slot].y = detection_data[d][1];
            state->tracks[empty_slot].w = detection_data[d][2];
            state->tracks[empty_slot].h = detection_data[d][3];
            state->tracks[empty_slot].confidence = detection_data[d][4];
            state->tracks[empty_slot].class_id = (int)detection_data[d][5];
            state->tracks[empty_slot].track_id = new_track_id;
            state->tracks[empty_slot].age = 0;
            state->tracks[empty_slot].hits = 1;
            state->tracks[empty_slot].active = 1;
            state->tracks[empty_slot].tracking_confidence = NEW_TRACK_CONFIDENCE_I8;
        }
    }
}

// Age unmatched tracks and remove expired ones (int8 version)
static void update_track_lifecycle_i8(ObjectTrackerState_i8* state, int max_tracks, int max_age, const int* track_matched) {
    for (int t = 0; t < max_tracks; t++) {
        if (!state->tracks[t].active) continue;
        
        if (!track_matched[t]) {
            state->tracks[t].age++;
            state->tracks[t].tracking_confidence *= TRACKING_CONFIDENCE_DECAY_I8;
            
            if (state->tracks[t].age > max_age) {
                // Clear all track history
                state->tracks[t].active = 0;
                state->tracks[t].track_id = -1;
                state->tracks[t].age = 0;
                state->tracks[t].hits = 0;
                state->tracks[t].x = 0.0f;
                state->tracks[t].y = 0.0f;
                state->tracks[t].w = 0.0f;
                state->tracks[t].h = 0.0f;
                state->tracks[t].confidence = 0.0f;
                state->tracks[t].class_id = -1;
                state->tracks[t].tracking_confidence = 0.0f;
            }
        }
    }
}

//============================================================================
// INT8 OUTPUT GENERATION
//============================================================================

// Generate int8 output tensor with tracked detections
static void generate_tracking_output_i8(ObjectTrackerState_i8* state, int8_t* restrict output, int max_detections, 
                                        int confidence_count, int max_tracks, int min_hits) {
    int output_count = 0;
    const int output_conf_count = confidence_count + 2; // +2 for track_id and tracking_confidence
    
    // Initialize output buffer to -128 (converts to 0.0 when read)
    for (int i = 0; i < max_detections * output_conf_count; i++) {
        output[i] = -128;
    }
    
    // Output confirmed tracks
    for (int t = 0; t < max_tracks && output_count < max_detections; t++) {
        if (!state->tracks[t].active || state->tracks[t].hits < min_hits) continue;
        
        // Copy bounding box coordinates
        set_output_value_i8(output, max_detections, output_conf_count, 0, output_count, state->tracks[t].x);
        set_output_value_i8(output, max_detections, output_conf_count, 1, output_count, state->tracks[t].y);
        set_output_value_i8(output, max_detections, output_conf_count, 2, output_count, state->tracks[t].w);
        set_output_value_i8(output, max_detections, output_conf_count, 3, output_count, state->tracks[t].h);
        
        // Set class confidence scores
        for (int j = 4; j < confidence_count; j++) {
            const float conf_value = (j - 4 == state->tracks[t].class_id) ? state->tracks[t].confidence : 0.0f;
            set_output_value_i8(output, max_detections, output_conf_count, j, output_count, conf_value);
        }
        
        // Add tracking metadata
        // Track ID: normalize to [0,1] range for int8 storage (1-127 -> stored as values)
        float normalized_track_id = (float)state->tracks[t].track_id / (float)MAX_TRACK_ID_I8;
        set_output_value_i8(output, max_detections, output_conf_count, confidence_count, output_count, normalized_track_id);
        set_output_value_i8(output, max_detections, output_conf_count, confidence_count + 1, output_count, state->tracks[t].tracking_confidence);
        
        output_count++;
    }
}

//============================================================================
// MAIN INT8 TRACKING FUNCTION
//============================================================================

// Main object tracking function for int8 - maintains object identities across frames
static void object_tracker_i8(
    const int8_t* restrict detections,
    int8_t* restrict tracker_state_bytes,
    int8_t* restrict output,
    int max_detections,
    int confidence_count,
    float tracking_threshold,
    int max_tracks,
    int max_age,
    int min_hits)
{
    // Cast the byte buffer to our state structure
    ObjectTrackerState_i8* state = (ObjectTrackerState_i8*)tracker_state_bytes;
    
    // Step 1: Extract detections from int8 input tensor
    float detection_data[max_detections][DETECTION_DATA_SIZE_I8];
    const int valid_detections = extract_detections_i8(detections, max_detections, confidence_count, detection_data);
    
    // Step 2: Associate detections with existing tracks
    int detection_matched[max_detections];
    int track_matched[max_tracks];
    perform_tracking_association_i8(state, detection_data, valid_detections, max_tracks, tracking_threshold, 
                                    detection_matched, track_matched);
    
    // Step 3: Create new tracks for unmatched detections
    create_new_tracks_i8(state, detection_data, valid_detections, max_tracks, detection_matched);
    
    // Step 4: Age unmatched tracks and remove expired ones
    update_track_lifecycle_i8(state, max_tracks, max_age, track_matched);
    
    // Step 5: Generate int8 output with tracked detections
    generate_tracking_output_i8(state, output, max_detections, confidence_count, max_tracks, min_hits);
}

#pragma IMAGINET_FRAGMENT_END 