#include <math.h>
#include <string.h>

static inline float safe_expf(float x) {
    return expf(x < 11.0f ? x : 11.0f);
}

static inline void sigmoid_inplace(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 1.0f / (1.0f + safe_expf(-data[i]));
    }
}

static inline void exp_inplace(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = safe_expf(data[i]);
    }
}

static inline void softmax_inplace(float* data, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float* row = data + r * cols;
        float mx = row[0];
        for (int c = 1; c < cols; c++) if (row[c] > mx) mx = row[c];
        float sum = 0;
        for (int c = 0; c < cols; c++) { row[c] = safe_expf(row[c] - mx); sum += row[c]; }
        float inv = 1.0f / sum;
        for (int c = 0; c < cols; c++) row[c] *= inv;
    }
}

/* Process vision output array in-place:
   - MDN entries: exp() the std half
   - sigmoid entries: sigmoid() in-place
   - softmax entries: softmax() in-place
   
   Slice layout (1576 floats):
     meta:            [0:55]     55  sigmoid
     desire_pred:     [55:87]    32  softmax (4 rows x 8 cols)
     pose:            [87:99]    12  MDN (mu=6, std=exp(6))
     wide_from_device_euler: [99:105] 6 MDN (mu=3, std=exp(3))
     road_transform:  [105:117]  12  MDN (mu=6, std=exp(6))
     lane_lines:      [117:645]  528 MDN (mu=264, std=exp(264))
     lane_lines_prob: [645:653]  8   sigmoid
     road_edges:      [653:917]  264 MDN (mu=132, std=exp(132))
     lead:            [917:1061] 144 MDN (mu=72, std=exp(72))
     lead_prob:       [1061:1064] 3  sigmoid
     hidden_state:    [1064:1576] 512 passthrough
*/
void fast_parse_vision(float* v) {
    /* meta: sigmoid */
    sigmoid_inplace(v + 0, 55);
    
    /* desire_pred: softmax 4x8 */
    softmax_inplace(v + 55, 4, 8);
    
    /* pose: exp std half (87+6=93..99) */
    exp_inplace(v + 93, 6);
    
    /* wide_from_device_euler: exp std half (99+3=102..105) */
    exp_inplace(v + 102, 3);
    
    /* road_transform: exp std half (105+6=111..117) */
    exp_inplace(v + 111, 6);
    
    /* lane_lines: exp std half (117+264=381..645) */
    exp_inplace(v + 381, 264);
    
    /* lane_lines_prob: sigmoid */
    sigmoid_inplace(v + 645, 8);
    
    /* road_edges: exp std half (653+132=785..917) */
    exp_inplace(v + 785, 132);
    
    /* lead: exp std half (917+72=989..1061) */
    exp_inplace(v + 989, 72);
    
    /* lead_prob: sigmoid */
    sigmoid_inplace(v + 1061, 3);
}

/* Process policy output in-place:
   plan: [0:990] 990 floats - MHP with in_N=5, out_N=1
     reshape to (1, 5, 198), n_values=98.5... wait that's wrong.
     Actually: is_mhp check: shape = IDX_N * PLAN_WIDTH = 33*15 = 495
     outs['plan'].shape[1] = 990
     2 * 495 = 990 → NOT MHP (returns False)
     So plan uses in_N=0, out_N=0, out_shape=(33, 15)
     n_values = 990/2 = 495, mu=first 495, std=exp(last 495)
   
   desire_state: [990:998] 8 floats - softmax (1x8)
*/
void fast_parse_policy(float* p) {
    /* plan: exp std half (0+495=495..990) */
    exp_inplace(p + 495, 495);
    
    /* desire_state: softmax 1x8 */
    softmax_inplace(p + 990, 1, 8);
}
