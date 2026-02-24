#pragma IMAGINET_INCLUDES_BEGIN
#include <stdint.h>
#include <math.h>
#pragma IMAGINET_INCLUDES_END

#pragma IMAGINET_FRAGMENT_BEGIN "draw_line_f32"

//============================================================================
// CONSTANTS AND COLOR PALETTE
//============================================================================

// Comprehensive design system color palette (R, G, B values)
static const float DRAW_LINE_COLORS[][3] = {
    {0.0f, 0.0f, 0.0f},       // 0: Black
    {1.0f, 1.0f, 1.0f},       // 1: White
    {0.161f, 0.439f, 1.0f},   // 2: BlueDark500 #2970ff
    {0.937f, 0.408f, 0.125f}, // 3: Orange500 #ef6820
    {0.082f, 0.718f, 0.62f},  // 4: Teal500 #15b79e
    {0.365f, 0.42f, 0.596f},  // 5: GrayCool500 #5d6b98
    {0.831f, 0.267f, 0.945f}, // 6: Fuchsia500 #d444f1
    {0.4f, 0.776f, 0.11f},    // 7: GreenLight500 #66c61c
    {0.965f, 0.239f, 0.408f}, // 8: Rose500 #f63d68
    {0.475f, 0.443f, 0.42f},  // 9: GrayWarm500 #79716b
    {0.38f, 0.447f, 0.953f},  // 10: Indigo500 #6172f3
    {0.969f, 0.565f, 0.035f}, // 11: Warning500 #F79009
    {0.0f, 0.525f, 0.788f},   // 12: BlueLight500 #0086c9
    {0.424f, 0.451f, 0.498f}, // 13: GrayNeutral500 #6c737f
    {0.09f, 0.698f, 0.416f},  // 14: Success500 #17B26A
    {0.478f, 0.353f, 0.973f}, // 15: Purple500 #7a5af8
    {0.918f, 0.667f, 0.031f}, // 16: Yellow500 #eaaa08
    {0.451f, 0.451f, 0.451f}, // 17: GrayTrue500 #737373
    {0.529f, 0.357f, 0.969f}, // 18: Violet500 #875bf7
    {1.0f, 0.267f, 0.02f},    // 19: OrangeDark500 #FF4405
    {0.024f, 0.682f, 0.831f}, // 20: Cyan500 #06aed4
    {0.439f, 0.439f, 0.463f}, // 21: GrayIron500 #707076
    {0.941f, 0.267f, 0.22f},  // 22: Error500 #F04438
    {0.18f, 0.565f, 0.98f},   // 23: Blue500 #2e90fa
    {0.086f, 0.388f, 0.392f}, // 24: Green500 #166364
    {0.412f, 0.459f, 0.525f}, // 25: GrayModern500 #697586
    {0.231f, 0.608f, 0.569f}, // 26: InfineonBrand500 #3B9B91
    {0.306f, 0.357f, 0.651f}, // 27: GrayBlue500 #4e5ba6
    {0.933f, 0.275f, 0.737f}  // 28: Pink500 #ee46bc
};

#define DRAW_LINE_COLOR_COUNT 29

//============================================================================
// UTILITY FUNCTIONS
//============================================================================

// Helper function to get RGB values from color index using array lookup
static void get_line_draw_color(int color_index, float* r, float* g, float* b) {
    const int idx = (color_index >= 0 && color_index < DRAW_LINE_COLOR_COUNT) ? color_index : 0;
    *r = DRAW_LINE_COLORS[idx][0];
    *g = DRAW_LINE_COLORS[idx][1]; 
    *b = DRAW_LINE_COLORS[idx][2];
}

//============================================================================
// PIXEL SETTING SYSTEM  
//============================================================================

// Unified pixel setting function to eliminate code duplication
static void draw_line_set_pixel(float* restrict image, int image_width, int image_height, int channels,
                                int x, int y, float r, float g, float b) {
    if (x < 0 || x >= image_width || y < 0 || y >= image_height) return;
    
    if (channels == 3) {
        const int idx = (y * image_width + x) * channels;
        image[idx] = r;
        image[idx + 1] = g;
        image[idx + 2] = b;
    } else {
        // Use proper luminance formula for grayscale conversion
        const int idx = y * image_width + x;
        image[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

//============================================================================
// LINE DRAWING FUNCTIONS
//============================================================================

// Helper function to draw a line using Bresenham's algorithm
static void draw_unit_line(float* restrict image, int image_width, int image_height, int channels,
                          int x1, int y1, int x2, int y2, float r, float g, float b) {
    // Calculate absolute differences and step directions
    const int delta_x = abs(x2 - x1);
    const int delta_y = abs(y2 - y1);
    const int step_x = (x1 < x2) ? 1 : -1;
    const int step_y = (y1 < y2) ? 1 : -1;
    
    // Initialize error term for Bresenham's algorithm
    int error = delta_x - delta_y;
    
    // Current position starts at line start point
    int current_x = x1;
    int current_y = y1;
    
    // Draw line pixel by pixel
    while (1) {
        // Draw current pixel
        draw_line_set_pixel(image, image_width, image_height, channels, 
                           current_x, current_y, r, g, b);
        
        // Check if we've reached the end point
        if (current_x == x2 && current_y == y2) {
            break;
        }
        
        // Calculate error adjustment for next step
        const int error_doubled = 2 * error;
        
        // Step in X direction if needed
        if (error_doubled > -delta_y) {
            error -= delta_y;
            current_x += step_x;
        }
        
        // Step in Y direction if needed  
        if (error_doubled < delta_x) {
            error += delta_x;
            current_y += step_y;
        }
    }
}

// Helper function to draw a thick line
static void draw_unit_thick_line(float* restrict image, int image_width, int image_height, int channels,
                                int x1, int y1, int x2, int y2, int thickness, float r, float g, float b) {
    if (thickness <= 1) {
        draw_unit_line(image, image_width, image_height, channels, x1, y1, x2, y2, r, g, b);
        return;
    }
    
    // Calculate line direction to determine perpendicular offset
    const int dx = x2 - x1;
    const int dy = y2 - y1;
    const float length = sqrtf((float)(dx * dx + dy * dy));
    
    // Calculate perpendicular unit vector
    const float perp_x = -(float)dy / length;
    const float perp_y = (float)dx / length;
    
    // Draw multiple parallel lines
    const int half_thickness = thickness / 2;
    for (int offset = -half_thickness; offset <= half_thickness; offset++) {
        const int offset_x1 = x1 + (int)(perp_x * offset);
        const int offset_y1 = y1 + (int)(perp_y * offset);
        const int offset_x2 = x2 + (int)(perp_x * offset);
        const int offset_y2 = y2 + (int)(perp_y * offset);
        
        draw_unit_line(image, image_width, image_height, channels, 
                      offset_x1, offset_y1, offset_x2, offset_y2, r, g, b);
    }
}

//============================================================================
// MAIN DRAW LINE FUNCTION
//============================================================================

static void draw_line_f32(
    const float* restrict image,
    float* restrict output,
    int image_height,
    int image_width,
    int channels,
    double x1,
    double y1,
    double x2,
    double y2,
    int thickness,
    int color)
{
    // First, copy input image to output
    const int total_pixels = image_height * image_width * channels;
    for (int i = 0; i < total_pixels; ++i) {
        output[i] = image[i];
    }
    
    // Convert normalized coordinates (0-1) to pixel coordinates, handling edge case
    int pixel_x1 = x1 >= 1.0 ? image_width - 1 : (int)(x1 * image_width);
    int pixel_y1 = y1 >= 1.0 ? image_height - 1 : (int)(y1 * image_height);
    int pixel_x2 = x2 >= 1.0 ? image_width - 1 : (int)(x2 * image_width);
    int pixel_y2 = y2 >= 1.0 ? image_height - 1 : (int)(y2 * image_height);
    
    // Get RGB values from color index
    float r, g, b;
    get_line_draw_color(color, &r, &g, &b);
    
    // Draw line
    draw_unit_thick_line(output, image_width, image_height, channels,
                        pixel_x1, pixel_y1, pixel_x2, pixel_y2, thickness, r, g, b);
}

#pragma IMAGINET_FRAGMENT_END
