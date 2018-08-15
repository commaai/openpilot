//UI Overlay

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <cutils/properties.h>

#include <GLES3/gl3.h>
#include <EGL/egl.h>

#include <json.h>
#include <czmq.h>

#include "nanovg.h"
#define NANOVG_GLES3_IMPLEMENTATION
#include "nanovg_gl.h"
#include "nanovg_gl_utils.h"

#include "common/timing.h"
#include "common/util.h"
#include "common/swaglog.h"
#include "common/mat.h"
#include "common/glutil.h"

#include "common/touch.h"
#include "common/framebuffer.h"
#include "common/visionipc.h"
#include "common/visionimg.h"
#include "common/modeldata.h"
#include "common/params.h"

#include "cereal/gen/c/log.capnp.h"
//BB Cereal for UI
#include "cereal/gen/c/ui.capnp.h"


// Calibration status values from controlsd.py
#define CALIBRATION_UNCALIBRATED 0
#define CALIBRATION_CALIBRATED 1
#define CALIBRATION_INVALID 2

#define STATUS_STOPPED 0
#define STATUS_DISENGAGED 1
#define STATUS_ENGAGED 2
#define STATUS_WARNING 3
#define STATUS_ALERT 4
#define STATUS_MAX 5

#define ALERTSIZE_NONE 0
#define ALERTSIZE_SMALL 1
#define ALERTSIZE_MID 2
#define ALERTSIZE_FULL 3

#define UI_BUF_COUNT 4

const int vwp_w = 1920;
const int vwp_h = 1080;
const int nav_w = 640;
const int nav_ww= 760;
const int sbr_w = 300;
const int bdr_s = 30;
const int box_x = sbr_w+bdr_s;
const int box_y = bdr_s;
const int box_w = vwp_w-sbr_w-(bdr_s*2);
const int box_h = vwp_h-(bdr_s*2);
const int viz_w = vwp_w-(bdr_s*2);
const int header_h = 420;

const uint8_t bg_colors[][4] = {
  [STATUS_STOPPED] = {0x07, 0x23, 0x39, 0xff},
  [STATUS_DISENGAGED] = {0x17, 0x33, 0x49, 0xff},
  [STATUS_ENGAGED] = {0x17, 0x86, 0x44, 0xff},
  [STATUS_WARNING] = {0xDA, 0x6F, 0x25, 0xff},
  [STATUS_ALERT] = {0xC9, 0x22, 0x31, 0xff},
};

const uint8_t alert_colors[][4] = {
  [STATUS_STOPPED] = {0x07, 0x23, 0x39, 0xf1},
  [STATUS_DISENGAGED] = {0x17, 0x33, 0x49, 0xc8},
  [STATUS_ENGAGED] = {0x17, 0x86, 0x44, 0xf1},
  [STATUS_WARNING] = {0xDA, 0x6F, 0x25, 0xf1},
  [STATUS_ALERT] = {0xC9, 0x22, 0x31, 0xf1},
};

const int alert_sizes[] = {
  [ALERTSIZE_NONE] = 0,
  [ALERTSIZE_SMALL] = 241,
  [ALERTSIZE_MID] = 390,
  [ALERTSIZE_FULL] = vwp_h,
};

typedef struct UICstmButton {
  char *btn_name;
  char *btn_label;
  char *btn_label2;
} UICstmButton;

typedef struct UIScene {
  int frontview;

  int transformed_width, transformed_height;

  uint64_t model_ts;
  ModelData model;

  float mpc_x[50];
  float mpc_y[50];

  bool world_objects_visible;
  mat3 warp_matrix;           // transformed box -> frame.
  mat4 extrinsic_matrix;      // Last row is 0 so we can use mat4.

  float v_cruise;
  uint64_t v_cruise_update_ts;
  float v_ego;
  float curvature;
  int engaged;
  bool engageable;

  bool uilayout_sidebarcollapsed;
  bool uilayout_mapenabled;
  // responsive layout
  int ui_viz_rx;
  int ui_viz_rw;
  int ui_viz_ro;

  int lead_status;
  float lead_d_rel, lead_y_rel, lead_v_rel;

  int front_box_x, front_box_y, front_box_width, front_box_height;

  uint64_t alert_ts;
  char alert_text1[1024];
  char alert_text2[1024];
  uint8_t alert_size;
  float alert_blinkingrate;

  float awareness_status;

  uint64_t started_ts;
  
  // Used to display calibration progress
  int cal_status;
  int cal_perc;
  // Used to show gps planner status
  bool gps_planner_active;

} UIScene;

typedef struct UIState {
  pthread_mutex_t lock;
  pthread_cond_t bg_cond;

  FramebufferState *fb;
  int fb_w, fb_h;
  EGLDisplay display;
  EGLSurface surface;

  NVGcontext *vg;

  int font_courbd;
  int font_sans_regular;
  int font_sans_semibold;
  int font_sans_bold;
  int img_wheel;

  zsock_t *thermal_sock;
  void *thermal_sock_raw;
  zsock_t *model_sock;
  void *model_sock_raw;
  zsock_t *live100_sock;
  void *live100_sock_raw;
  zsock_t *livecalibration_sock;
  void *livecalibration_sock_raw;
  zsock_t *live20_sock;
  void *live20_sock_raw;
  zsock_t *livempc_sock;
  void *livempc_sock_raw;
  zsock_t *plus_sock;
  void *plus_sock_raw;
  zsock_t *gps_sock;
  void *gps_sock_raw;

  zsock_t *uilayout_sock;
  void *uilayout_sock_raw;

  int plus_state;

  // vision state
  bool vision_connected;
  bool vision_connect_firstrun;
  int ipc_fd;

  VIPCBuf bufs[UI_BUF_COUNT];
  VIPCBuf front_bufs[UI_BUF_COUNT];
  int cur_vision_idx;
  int cur_vision_front_idx;

  GLuint frame_program;
  GLuint frame_texs[UI_BUF_COUNT];
  GLuint frame_front_texs[UI_BUF_COUNT];

  GLint frame_pos_loc, frame_texcoord_loc;
  GLint frame_texture_loc, frame_transform_loc;

  GLuint line_program;
  GLint line_pos_loc, line_color_loc;
  GLint line_transform_loc;

  unsigned int rgb_width, rgb_height, rgb_stride;
  size_t rgb_buf_len;
  mat4 rgb_transform;

  unsigned int rgb_front_width, rgb_front_height, rgb_front_stride;
  size_t rgb_front_buf_len;

  bool intrinsic_matrix_loaded;
  mat3 intrinsic_matrix;

  UIScene scene;

  bool awake;
  int awake_timeout;

  int status;
  bool is_metric;
  bool passive;
  int alert_size;
  float alert_blinking_alpha;
  bool alert_blinked;
  bool acc_enabled;

  float light_sensor;
} UIState;


typedef struct BBUIVals {
  //BB
  UICstmButton btns[6];
  char btns_status[6];
  char *car_model;
  char *car_folder;
  char *custom_message;
  int custom_message_status;
  zsock_t *uiButtonInfo_sock;
  void *uiButtonInfo_sock_raw;
  zsock_t *uiCustomAlert_sock;
  void *uiCustomAlert_sock_raw;
  zsock_t *uiSetCar_sock;
  void *uiSetCar_sock_raw;
  zsock_t *uiPlaySound_sock;
  void *uiPlaySound_sock_raw;
  zsock_t *uiButtonStatus_sock;
  void *uiButtonStatus_sock_raw;
  zsock_t *uiUpdate_sock;
  void *uiUpdate_sock_raw; 
  int btns_x[6];
  int btns_y[6];
  int btns_r[6];
  time_t label_last_modified;
  time_t status_last_modified;
  uint16_t maxCpuTemp;
  uint32_t maxBatTemp;
  float gpsAccuracy ;
  float freeSpace;
  float angleSteers;
  float angleSteersDes;
  //BB END
} BBUIVals;


static const char frame_vertex_shader[] =
  "attribute vec4 aPosition;\n"
  "attribute vec4 aTexCoord;\n"
  "uniform mat4 uTransform;\n"
  "varying vec4 vTexCoord;\n"
  "void main() {\n"
  "  gl_Position = uTransform * aPosition;\n"
  "  vTexCoord = aTexCoord;\n"
  "}\n";

static const char frame_fragment_shader[] =
  "precision mediump float;\n"
  "uniform sampler2D uTexture;\n"
  "varying vec4 vTexCoord;\n"
  "void main() {\n"
  "  gl_FragColor = texture2D(uTexture, vTexCoord.xy);\n"
  "}\n";

static const char line_vertex_shader[] =
  "attribute vec4 aPosition;\n"
  "attribute vec4 aColor;\n"
  "uniform mat4 uTransform;\n"
  "varying vec4 vColor;\n"
  "void main() {\n"
  "  gl_Position = uTransform * aPosition;\n"
  "  vColor = aColor;\n"
  "}\n";

static const char line_fragment_shader[] =
  "precision mediump float;\n"
  "uniform sampler2D uTexture;\n"
  "varying vec4 vColor;\n"
  "void main() {\n"
  "  gl_FragColor = vColor;\n"
  "}\n";



//BB START: functions added for the display of various items

static int bb_ui_draw_measure(UIState *s, BBUIVals *v,  const char* bb_value, const char* bb_uom, const char* bb_label, 
		int bb_x, int bb_y, int bb_uom_dx,
		NVGcolor bb_valueColor, NVGcolor bb_labelColor, NVGcolor bb_uomColor, 
		int bb_valueFontSize, int bb_labelFontSize, int bb_uomFontSize )  {
  const UIScene *scene = &s->scene;	
  nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_BASELINE);
  int dx = 0;
  if (strlen(bb_uom) > 0) {
  	dx = (int)(bb_uomFontSize*2.5/2);
   }
  //print value
  nvgFontFace(s->vg, "sans-semibold");
  nvgFontSize(s->vg, bb_valueFontSize*2.5);
  nvgFillColor(s->vg, bb_valueColor);
  nvgText(s->vg, bb_x-dx/2, bb_y+ (int)(bb_valueFontSize*2.5)+5, bb_value, NULL);
  //print label
  nvgFontFace(s->vg, "sans-regular");
  nvgFontSize(s->vg, bb_labelFontSize*2.5);
  nvgFillColor(s->vg, bb_labelColor);
  nvgText(s->vg, bb_x, bb_y + (int)(bb_valueFontSize*2.5)+5 + (int)(bb_labelFontSize*2.5)+5, bb_label, NULL);
  //print uom
  if (strlen(bb_uom) > 0) {
      nvgSave(s->vg);
	  int rx =bb_x + bb_uom_dx + bb_valueFontSize -3;
	  int ry = bb_y + (int)(bb_valueFontSize*2.5/2)+25;
	  nvgTranslate(s->vg,rx,ry);
	  nvgRotate(s->vg, -1.5708); //-90deg in radians
	  nvgFontFace(s->vg, "sans-regular");
	  nvgFontSize(s->vg, (int)(bb_uomFontSize*2.5));
	  nvgFillColor(s->vg, bb_uomColor);
	  nvgText(s->vg, 0, 0, bb_uom, NULL);
	  nvgRestore(s->vg);
  }
  return (int)((bb_valueFontSize + bb_labelFontSize)*2.5) + 5;
}

static bool bb_handle_ui_touch(UIState *s, BBUIVals *v, int touch_x, int touch_y) {
  char *out_status_file = malloc(90);
  sprintf(out_status_file,"/data/openpilot/selfdrive/car/%s/buttons.ui.msg",v->car_model);
  char temp_stats[6];
  int oFile;
  for(int i=0; i<6; i++) {
    if (v->btns_r[i] > 0) {
      if ((abs(touch_x - v->btns_x[i]) < v->btns_r[i]) && (abs(touch_y - v->btns_y[i]) < v->btns_r[i])) {
        //found it; change the status and write to file
        if (v->btns_status[i] > 0) {
          v->btns_status[i] = 0;
        } else {
          v->btns_status[i] = 1;
        }
        //now write to file
        for (int j=0; j<6; j++) {
          if (v->btns_status[j] ==0) {
            temp_stats[j]='0';
          } else {
            temp_stats[j]='1';
          }
        }
        oFile = open(out_status_file,O_WRONLY);
        if (oFile != -1) {
          write(oFile,&temp_stats,6);
        }
        close(oFile);
        //done, return true
        return true;
      }
    }
  }
  return false;
};

static int bb_get_button_status(UIState *s, BBUIVals *v, char *btn_name) {
  int ret_status = -1;
  for (int i = 0; i< 6; i++) {
    if (strcmp(v->btns[i].btn_name,btn_name)==0) {
      ret_status = v->btns_status[i];
    }
  }
  return ret_status;
}

static void bb_draw_button(UIState *s, BBUIVals *v, int btn_id) {
  const UIScene *scene = &s->scene;

  int viz_button_x = 0;
  const int viz_button_y = (box_y + (bdr_s*1.5)) + 20;
  const int viz_button_w = 140;
  const int viz_button_h = 140;

  char *btn_text, *btn_text2;
  
  const int delta_x = viz_button_w * 1.1;
  
  if (btn_id >2) {
    viz_button_x = scene->ui_viz_rx + scene->ui_viz_rw - (bdr_s*2) -190;
    viz_button_x -= (6-btn_id) * delta_x ;
  } else {
    viz_button_x = scene->ui_viz_rx + (bdr_s*2) + 200;
    viz_button_x +=  (btn_id) * delta_x;
  }

  btn_text = v->btns[btn_id].btn_label;
  btn_text2 = v->btns[btn_id].btn_label2;
  
  if (strcmp(btn_text,"")==0) {
    v->btns_r[btn_id] = 0;
  } else {
    v->btns_r[btn_id]= (int)((viz_button_w + viz_button_h)/4);
  }
  v->btns_x[btn_id]=viz_button_x + v->btns_r[btn_id];
  v->btns_y[btn_id]=viz_button_y + v->btns_r[btn_id];
  if (v->btns_r[btn_id] == 0) {
    return;
  }
  
  nvgBeginPath(s->vg);
  nvgRoundedRect(s->vg, viz_button_x, viz_button_y, viz_button_w, viz_button_h, 80);
  nvgStrokeWidth(s->vg, 12);

  
  if (v->btns_status[btn_id] ==0) {
    //disabled - red
    nvgStrokeColor(s->vg, nvgRGBA(255, 0, 0, 200));
    if (strcmp(btn_text2,"")==0) {
      btn_text2 = "Off";
    }
  } else
  if (v->btns_status[btn_id] ==1) {
    //enabled - white
    nvgStrokeColor(s->vg, nvgRGBA(255,255,255,200));
    nvgStrokeWidth(s->vg, 4);
    if (strcmp(btn_text2,"")==0) {
      btn_text2 = "Ready";
    }
  } else
  if (v->btns_status[btn_id] ==2) {
    //active - green
    nvgStrokeColor(s->vg, nvgRGBA(28, 204,98,200));
    if (strcmp(btn_text2,"")==0) {
      btn_text2 = "Active";
    }
  } else
  if (v->btns_status[btn_id] ==9) {
    //available - thin white
    nvgStrokeColor(s->vg, nvgRGBA(200,200,200,40));
    nvgStrokeWidth(s->vg, 4);
    if (strcmp(btn_text2,"")==0) {
      btn_text2 = "";
    }
  } else {
    //others - orange
    nvgStrokeColor(s->vg, nvgRGBA(255, 188, 3, 200));
    if (strcmp(btn_text2,"")==0) {
      btn_text2 = "Alert";
    }
  }

  nvgStroke(s->vg);

  nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_BASELINE);
  nvgFontFace(s->vg, "sans-regular");
  nvgFontSize(s->vg, 14*2.5);
  nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 200));
  nvgText(s->vg, viz_button_x+viz_button_w/2, 210, btn_text2, NULL);

  nvgFontFace(s->vg, "sans-semibold");
  nvgFontSize(s->vg, 28*2.5);
  nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 255));
  nvgText(s->vg, viz_button_x+viz_button_w/2, 183, btn_text, NULL);
}

static void bb_draw_buttons(UIState *s, BBUIVals *v) {
  const UIScene *scene = &s->scene;
  char *labels_file = malloc(90);
  char *in_status_file = malloc(90);
  
  sprintf(labels_file,"/data/openpilot/selfdrive/car/%s/buttons.msg",v->car_model);
  sprintf(in_status_file,"/data/openpilot/selfdrive/car/%s/buttons.cc.msg",v->car_model);
  
  int lFile;
  int sFile;
  char temp_stats[6];
  struct stat filestat;
  int file_status;
  bool changes_present;

  changes_present = false;
  file_status = stat(labels_file, &filestat);
  //read only if modified after last read
  if ((filestat.st_mtime > v->label_last_modified) || (v->label_last_modified ==0)) {
    lFile = open (labels_file, O_RDONLY);
    if (lFile != -1) {
      int rd = read(lFile, &(v->btns), 6*sizeof(struct UICstmButton));
      close(lFile);
      v->label_last_modified = filestat.st_mtime;
      changes_present = true;
    }
  }
  file_status = stat(in_status_file, &filestat);
  //read only if modified after last read
  if ((filestat.st_mtime > v->status_last_modified) || (v->status_last_modified ==0)) {
    sFile = open(in_status_file, O_RDONLY);
    if (sFile != -1) {
      int rd = read(sFile, &(temp_stats),6*sizeof(char));
      if (rd == 6) {
        for (int i = 0; i < 6; i++) {
          v->btns_status[i] = temp_stats[i]-'0';
        }
      }
      close(sFile);
      v->status_last_modified = filestat.st_mtime;
      changes_present = true;
    }
  }
  for (int i = 0; i < 6; i++) {
    bb_draw_button(s,v,i);
  }
}

static void bb_ui_draw_custom_alert(UIState *s, BBUIVals *v) {
  const UIScene *scene = &s->scene;
  char *filepath = malloc(90);
  sprintf(filepath,"/data/openpilot/selfdrive/car/%s/alert.msg",v->car_model);
  //get 3-state switch position
  int alert_msg_fd;
  char alert_msg[1000];
  if (strlen(s->scene.alert_text1) > 0) {
    //already a system alert, ignore ours
    return;
  }
  alert_msg_fd = open (filepath, O_RDONLY);
  //if we can't open then done
  if (alert_msg_fd == -1) {
    return;
  } else {
    int rd = read(alert_msg_fd, &(s->scene.alert_text1), 1000);
    s->scene.alert_text1[rd] = '\0';
    close(alert_msg_fd);
    if(!strcmp(s->scene.alert_text1, "ACC Enabled")){
      s->acc_enabled = true;
    }
    if(!strcmp(s->scene.alert_text1, "ACC Disabled")){
      s->acc_enabled = false;
    }
    if (strlen(s->scene.alert_text1) > 0) {
      s->scene.alert_size = ALERTSIZE_SMALL;
    } else {
      s->scene.alert_size = ALERTSIZE_NONE;
      s->scene.alert_text1[0]=0;
    }
  }
}



static void bb_ui_draw_measures_left(UIState *s, BBUIVals *v, int bb_x, int bb_y, int bb_w ) {
	const UIScene *scene = &s->scene;		
	int bb_rx = bb_x + (int)(bb_w/2);
	int bb_ry = bb_y;
	int bb_h = 5; 
	NVGcolor lab_color = nvgRGBA(255, 255, 255, 200);
	NVGcolor uom_color = nvgRGBA(255, 255, 255, 200);
	int value_fontSize=30;
	int label_fontSize=15;
	int uom_fontSize = 15;
	int bb_uom_dx =  (int)(bb_w /2 - uom_fontSize*2.5) ;
	
	//add CPU temperature
	if (true) {
	    char val_str[16];
		char uom_str[6];
		NVGcolor val_color = nvgRGBA(255, 255, 255, 200);
        if((int)(v->maxCpuTemp/10) > 80) {
            val_color = nvgRGBA(255, 188, 3, 200);
        }
        if((int)(v->maxCpuTemp/10) > 92) {
            val_color = nvgRGBA(255, 0, 0, 200);
        }
        // temp is alway in C * 10
        if (s->is_metric) {
                snprintf(val_str, sizeof(val_str), "%d C", (int)(v->maxCpuTemp/10));
        } else {
                snprintf(val_str, sizeof(val_str), "%d F", (int)(32+9*(v->maxCpuTemp/10)/5));
        }
		snprintf(uom_str, sizeof(uom_str), "");
		bb_h +=bb_ui_draw_measure(s, v, val_str, uom_str, "CPU TEMP", 
				bb_rx, bb_ry, bb_uom_dx,
				val_color, lab_color, uom_color, 
				value_fontSize, label_fontSize, uom_fontSize );
		bb_ry = bb_y + bb_h;
	}

   //add battery temperature
	if (true) {
		char val_str[16];
		char uom_str[6];
		NVGcolor val_color = nvgRGBA(255, 255, 255, 200);
		if((int)(v->maxBatTemp/1000) > 40) {
			val_color = nvgRGBA(255, 188, 3, 200);
		}
		if((int)(v->maxBatTemp/1000) > 50) {
			val_color = nvgRGBA(255, 0, 0, 200);
		}
		// temp is alway in C * 1000
		if (s->is_metric) {
			 snprintf(val_str, sizeof(val_str), "%d C", (int)(v->maxBatTemp/1000));
		} else {
			 snprintf(val_str, sizeof(val_str), "%d F", (int)(32+9*(v->maxBatTemp/1000)/5));
		}
		snprintf(uom_str, sizeof(uom_str), "");
		bb_h +=bb_ui_draw_measure(s, v, val_str, uom_str, "BAT TEMP", 
				bb_rx, bb_ry, bb_uom_dx,
				val_color, lab_color, uom_color, 
				value_fontSize, label_fontSize, uom_fontSize );
		bb_ry = bb_y + bb_h;
	}
	
	//add grey panda GPS accuracy
	if (true) {
		char val_str[16];
		char uom_str[3];
		NVGcolor val_color = nvgRGBA(255, 255, 255, 200);
		//show red/orange if gps accuracy is high
	    if(v->gpsAccuracy > 0.59) {
	       val_color = nvgRGBA(255, 188, 3, 200);
	    }
	    if(v->gpsAccuracy > 0.8) {
	       val_color = nvgRGBA(255, 0, 0, 200);
	    }


		// gps accuracy is always in meters
		if (s->is_metric) {
			 snprintf(val_str, sizeof(val_str), "%d", (int)(v->gpsAccuracy*100.0));
		} else {
			 snprintf(val_str, sizeof(val_str), "%.1f", v->gpsAccuracy * 3.28084 * 12);
		}
		if (s->is_metric) {
			snprintf(uom_str, sizeof(uom_str), "cm");;
		} else {
			snprintf(uom_str, sizeof(uom_str), "in");
		}
		bb_h +=bb_ui_draw_measure(s, v, val_str, uom_str, "GPS PREC", 
				bb_rx, bb_ry, bb_uom_dx,
				val_color, lab_color, uom_color, 
				value_fontSize, label_fontSize, uom_fontSize );
		bb_ry = bb_y + bb_h;
	}
  //add free space - from bthaler1
	if (true) {
		char val_str[16];
		char uom_str[3];
		NVGcolor val_color = nvgRGBA(255, 255, 255, 200);

		//show red/orange if free space is low
		if(v->freeSpace < 0.4) {
			val_color = nvgRGBA(255, 188, 3, 200);
		}
		if(v->freeSpace < 0.2) {
			val_color = nvgRGBA(255, 0, 0, 200);
		}

		snprintf(val_str, sizeof(val_str), "%.1f", v->freeSpace* 100);
		snprintf(uom_str, sizeof(uom_str), "%%");

		bb_h +=bb_ui_draw_measure(s, v, val_str, uom_str, "FREE", 
			bb_rx, bb_ry, bb_uom_dx,
			val_color, lab_color, uom_color, 
			value_fontSize, label_fontSize, uom_fontSize );
		bb_ry = bb_y + bb_h;
	}	
	//finally draw the frame
	bb_h += 20;
	nvgBeginPath(s->vg);
  	nvgRoundedRect(s->vg, bb_x, bb_y, bb_w, bb_h, 20);
  	nvgStrokeColor(s->vg, nvgRGBA(255,255,255,80));
  	nvgStrokeWidth(s->vg, 6);
  	nvgStroke(s->vg);
}


static void bb_ui_draw_measures_right(UIState *s, BBUIVals *v, int bb_x, int bb_y, int bb_w ) {
	const UIScene *scene = &s->scene;		
	int bb_rx = bb_x + (int)(bb_w/2);
	int bb_ry = bb_y;
	int bb_h = 5; 
	NVGcolor lab_color = nvgRGBA(255, 255, 255, 200);
	NVGcolor uom_color = nvgRGBA(255, 255, 255, 200);
	int value_fontSize=30;
	int label_fontSize=15;
	int uom_fontSize = 15;
	int bb_uom_dx =  (int)(bb_w /2 - uom_fontSize*2.5) ;
	
	//add visual radar relative distance
	if (true) {
		char val_str[16];
		char uom_str[6];
		NVGcolor val_color = nvgRGBA(255, 255, 255, 200);
		if (scene->lead_status) {
			//show RED if less than 5 meters
			//show orange if less than 15 meters
			if((int)(scene->lead_d_rel) < 15) {
				val_color = nvgRGBA(255, 188, 3, 200);
			}
			if((int)(scene->lead_d_rel) < 5) {
				val_color = nvgRGBA(255, 0, 0, 200);
			}
			// lead car relative distance is always in meters
			if (s->is_metric) {
				 snprintf(val_str, sizeof(val_str), "%d", (int)scene->lead_d_rel);
			} else {
				 snprintf(val_str, sizeof(val_str), "%d", (int)(scene->lead_d_rel * 3.28084));
			}
		} else {
		   snprintf(val_str, sizeof(val_str), "-");
		}
		if (s->is_metric) {
			snprintf(uom_str, sizeof(uom_str), "m   ");
		} else {
			snprintf(uom_str, sizeof(uom_str), "ft");
		}
		bb_h +=bb_ui_draw_measure(s, v, val_str, uom_str, "REL DIST", 
				bb_rx, bb_ry, bb_uom_dx,
				val_color, lab_color, uom_color, 
				value_fontSize, label_fontSize, uom_fontSize );
		bb_ry = bb_y + bb_h;
	}
	
	//add visual radar relative speed
	if (true) {
		char val_str[16];
		char uom_str[6];
		NVGcolor val_color = nvgRGBA(255, 255, 255, 200);
		if (scene->lead_status) {
			//show Orange if negative speed (approaching)
			//show Orange if negative speed faster than 5mph (approaching fast)
			if((int)(scene->lead_v_rel) < 0) {
				val_color = nvgRGBA(255, 188, 3, 200);
			}
			if((int)(scene->lead_v_rel) < -5) {
				val_color = nvgRGBA(255, 0, 0, 200);
			}
			// lead car relative speed is always in meters
			if (s->is_metric) {
				 snprintf(val_str, sizeof(val_str), "%d", (int)(scene->lead_v_rel * 3.6 + 0.5));
			} else {
				 snprintf(val_str, sizeof(val_str), "%d", (int)(scene->lead_v_rel * 2.2374144 + 0.5));
			}
		} else {
		   snprintf(val_str, sizeof(val_str), "-");
		}
		if (s->is_metric) {
			snprintf(uom_str, sizeof(uom_str), "km/h");;
		} else {
			snprintf(uom_str, sizeof(uom_str), "mph");
		}
		bb_h +=bb_ui_draw_measure(s, v, val_str, uom_str, "REL SPD", 
				bb_rx, bb_ry, bb_uom_dx,
				val_color, lab_color, uom_color, 
				value_fontSize, label_fontSize, uom_fontSize );
		bb_ry = bb_y + bb_h;
	}
	
	//add  steering angle
	if (true) {
		char val_str[16];
		char uom_str[6];
		NVGcolor val_color = nvgRGBA(255, 255, 255, 200);
			//show Orange if more than 6 degrees
			//show red if  more than 12 degrees
			if(((int)(v->angleSteers) < -6) || ((int)(v->angleSteers) > 6)) {
				val_color = nvgRGBA(255, 188, 3, 200);
			}
			if(((int)(v->angleSteers) < -12) || ((int)(v->angleSteers) > 12)) {
				val_color = nvgRGBA(255, 0, 0, 200);
			}
			// steering is in degrees
			snprintf(val_str, sizeof(val_str), "%.1f",(v->angleSteers));

	    snprintf(uom_str, sizeof(uom_str), "deg");
		bb_h +=bb_ui_draw_measure(s, v,  val_str, uom_str, "STEER", 
				bb_rx, bb_ry, bb_uom_dx,
				val_color, lab_color, uom_color, 
				value_fontSize, label_fontSize, uom_fontSize );
		bb_ry = bb_y + bb_h;
	}
	
	//add  desired steering angle
	if (true) {
		char val_str[16];
		char uom_str[6];
		NVGcolor val_color = nvgRGBA(255, 255, 255, 200);
			//show Orange if more than 6 degrees
			//show red if  more than 12 degrees
			if(((int)(v->angleSteersDes) < -6) || ((int)(v->angleSteersDes) > 6)) {
				val_color = nvgRGBA(255, 188, 3, 200);
			}
			if(((int)(v->angleSteersDes) < -12) || ((int)(v->angleSteersDes) > 12)) {
				val_color = nvgRGBA(255, 0, 0, 200);
			}
			// steering is in degrees
			snprintf(val_str, sizeof(val_str), "%.1f",(v->angleSteersDes));

	    snprintf(uom_str, sizeof(uom_str), "deg");
		bb_h +=bb_ui_draw_measure(s, v, val_str, uom_str, "DES STEER", 
				bb_rx, bb_ry, bb_uom_dx,
				val_color, lab_color, uom_color, 
				value_fontSize, label_fontSize, uom_fontSize );
		bb_ry = bb_y + bb_h;
	}
	
	
	//finally draw the frame
	bb_h += 20;
	nvgBeginPath(s->vg);
  	nvgRoundedRect(s->vg, bb_x, bb_y, bb_w, bb_h, 20);
  	nvgStrokeColor(s->vg, nvgRGBA(255,255,255,80));
  	nvgStrokeWidth(s->vg, 6);
  	nvgStroke(s->vg);
}

//draw grid from wiki
static void ui_draw_vision_grid(UIState *s, BBUIVals *v) {
  const UIScene *scene = &s->scene;
  bool is_cruise_set = (s->scene.v_cruise != 0 && s->scene.v_cruise != 255);
  if (!is_cruise_set) {
    const int grid_spacing = 30;

    int ui_viz_rx = scene->ui_viz_rx;
    int ui_viz_rw = scene->ui_viz_rw;

    nvgSave(s->vg);

    // path coords are worked out in rgb-box space
    nvgTranslate(s->vg, 240.0f, 0.0);

    // zooom in 2x
    nvgTranslate(s->vg, -1440.0f / 2, -1080.0f / 2);
    nvgScale(s->vg, 2.0, 2.0);

    nvgScale(s->vg, 1440.0f / s->rgb_width, 1080.0f / s->rgb_height);

    nvgBeginPath(s->vg);
    nvgStrokeColor(s->vg, nvgRGBA(255,255,255,128));
    nvgStrokeWidth(s->vg, 1);

    for (int i=box_y; i < box_h; i+=grid_spacing) {
      nvgMoveTo(s->vg, ui_viz_rx, i);
      //nvgLineTo(s->vg, ui_viz_rx, i);
      nvgLineTo(s->vg, ((ui_viz_rw + ui_viz_rx) / 2)+ 10, i);
    }

    for (int i=ui_viz_rx + 12; i <= ui_viz_rw; i+=grid_spacing) {
      nvgMoveTo(s->vg, i, 0);
      nvgLineTo(s->vg, i, 1000);
    }
    nvgStroke(s->vg);
    nvgRestore(s->vg);
  }
}

static void bb_ui_draw_UI(UIState *s, BBUIVals *v) {
    //get 3-state switch position
    int tri_state_fd;
    int tri_state_switch;
    char buffer[10];
    tri_state_switch = 0;
    tri_state_fd = open ("/sys/devices/virtual/switch/tri-state-key/state", O_RDONLY);
    //if we can't open then switch should be considered in the middle, nothing done
    if (tri_state_fd == -1) {
        tri_state_switch = 2;
    } else {
        read (tri_state_fd, &buffer, 10);
        tri_state_switch = buffer[0] -48;
        close(tri_state_fd);
    }
  
    if (tri_state_switch == 1) {
        const UIScene *scene = &s->scene;
        const int bb_dml_w = 180;
        const int bb_dml_x =  (scene->ui_viz_rx + (bdr_s*2));
        const int bb_dml_y = (box_y + (bdr_s*1.5))+220;
        
        const int bb_dmr_w = 180;
        const int bb_dmr_x = scene->ui_viz_rx + scene->ui_viz_rw - bb_dmr_w - (bdr_s*2) ; 
        const int bb_dmr_y = (box_y + (bdr_s*1.5))+220;
        bb_ui_draw_measures_left(s,v,bb_dml_x, bb_dml_y, bb_dml_w );
        bb_ui_draw_measures_right(s,v,bb_dmr_x, bb_dmr_y, bb_dmr_w );
        bb_draw_buttons(s,v);
        bb_ui_draw_custom_alert(s,v);
	}
    if (tri_state_switch ==2) {
	 	bb_ui_draw_custom_alert(s,v);
	}
	if (tri_state_switch ==3) {
	    ui_draw_vision_grid(s,v);
	}
 }

 static void bb_ui_play_sound(UIState *s, BBUIVals *v, int sound) {
   char* snd_command;
   int bts = bb_get_button_status(s, v,"sound");
   if ((bts > 0) || (bts == -1)) {
    asprintf(&snd_command, "python /data/openpilot/selfdrive/car/modules/snd/playsound.py %d &", sound);
    system(snd_command);
   }
 }

 static void bb_ui_set_car(UIState *s, BBUIVals *v, char *model, char *folder) {
   v->car_model = model;
   v->car_folder = folder;
 }

 static void  bb_ui_poll_update(UIState *s, BBUIVals *v) {
    int err;
    while (true) {
        zmq_pollitem_t bb_polls[8] = {{0}};
        bb_polls[0].socket = v->uiButtonInfo_sock_raw;
        bb_polls[0].events = ZMQ_POLLIN;
        bb_polls[1].socket = v->uiCustomAlert_sock_raw;
        bb_polls[1].events = ZMQ_POLLIN;
        bb_polls[2].socket = v->uiSetCar_sock_raw;
        bb_polls[2].events = ZMQ_POLLIN;
        bb_polls[3].socket = v->uiPlaySound_sock_raw;
        bb_polls[3].events = ZMQ_POLLIN;
        bb_polls[4].socket = v->uiUpdate_sock_raw;
        bb_polls[4].events = ZMQ_POLLIN;
        bb_polls[5].socket = s->gps_sock_raw;
        bb_polls[5].events = ZMQ_POLLIN;
        bb_polls[6].socket = s->live100_sock_raw;
        bb_polls[6].events = ZMQ_POLLIN;
        bb_polls[7].socket = s->thermal_sock_raw;
        bb_polls[7].events = ZMQ_POLLIN;
        

        int ret = zmq_poll(bb_polls, 8, 0);
        if (ret < 0) {
        LOGW("bb poll failed (%d)", ret);
            break;
        }
        if (ret == 0) {
            break;
        }

        if (bb_polls[5].revents) 
        {
            // gps socket
            zmq_msg_t msg;
            err = zmq_msg_init(&msg);
            assert(err == 0);
            err = zmq_msg_recv(&msg, s->gps_sock_raw, 0);
            assert(err >= 0);

            struct capn ctx;
            capn_init_mem(&ctx, zmq_msg_data(&msg), zmq_msg_size(&msg), 0);

            cereal_Event_ptr eventp;
            eventp.p = capn_getp(capn_root(&ctx), 0, 1);
            struct cereal_Event eventd;
            cereal_read_Event(&eventd, eventp);

            struct cereal_GpsLocationData datad;
            cereal_read_GpsLocationData(&datad, eventd.gpsLocation);

            v->gpsAccuracy= datad.accuracy;
            if (v->gpsAccuracy>100)
            {
                v->gpsAccuracy=99.99;
            }
            else if (v->gpsAccuracy==0)
            {
                v->gpsAccuracy=99.8;
            }
            capn_free(&ctx);
            zmq_msg_close(&msg);
        } else if (bb_polls[6].revents || bb_polls[7].revents) 
        {
            // zmq messages
            void* which = NULL;
            for (int i=6; i<8; i++) {
                if (bb_polls[i].revents) {
                    which = bb_polls[i].socket;
                    break;
                }
            }
            if (which == NULL) {
                continue;
            }

            zmq_msg_t msg;
            err = zmq_msg_init(&msg);
            assert(err == 0);
            err = zmq_msg_recv(&msg, which, 0);
            assert(err >= 0);

            struct capn ctx;
            capn_init_mem(&ctx, zmq_msg_data(&msg), zmq_msg_size(&msg), 0);

            cereal_Event_ptr eventp;
            eventp.p = capn_getp(capn_root(&ctx), 0, 1);
            struct cereal_Event eventd;
            cereal_read_Event(&eventd, eventp);

            if (eventd.which == cereal_Event_thermal) {
                struct cereal_ThermalData datad;
                cereal_read_ThermalData(&datad, eventd.thermal);
                v->maxCpuTemp=datad.cpu0;
                if (v->maxCpuTemp<datad.cpu1)
                {
                    v->maxCpuTemp=datad.cpu1;
                }
                else if (v->maxCpuTemp<datad.cpu2)
                {
                    v->maxCpuTemp=datad.cpu2;
                }
                else if (v->maxCpuTemp<datad.cpu3)
                {
                    v->maxCpuTemp=datad.cpu3;
                }
                v->maxBatTemp=datad.bat;
                v->freeSpace=datad.freeSpace;

            } else if (eventd.which == cereal_Event_live100) {
                struct cereal_Live100Data datad;
                cereal_read_Live100Data(&datad, eventd.live100);
                v->angleSteers = datad.angleSteers;
		        v->angleSteersDes = datad.angleSteersDes;
            }
            capn_free(&ctx);
            zmq_msg_close(&msg);
        } else {
            // zmq messages
            void* which = NULL;
            for (int i=0; i<5; i++) {
                if (bb_polls[i].revents) {
                    which = bb_polls[i].socket;
                    break;
                }
            }
            if (which == NULL) {
                continue;
            }

            zmq_msg_t msg;
            err = zmq_msg_init(&msg);
            assert(err == 0);
            err = zmq_msg_recv(&msg, which, 0);
            assert(err >= 0);

            struct capn ctx;
            capn_init_mem(&ctx, zmq_msg_data(&msg), zmq_msg_size(&msg), 0);

            cereal_UIEvent_ptr eventp;
            eventp.p = capn_getp(capn_root(&ctx), 0, 1);
            struct cereal_UIEvent eventd;
            cereal_read_UIEvent(&eventd, eventp);

            if (eventd.which == cereal_UIEvent_uiPlaySound) {
                struct cereal_UIPlaySound datad;
                cereal_read_UIPlaySound(&datad, eventd.uiPlaySound);
                int snd = datad.sndSound;
                bb_ui_play_sound(s,v,snd);
            } else if (eventd.which == cereal_UIEvent_uiButtonInfo) {
                struct cereal_UIButtonInfo datad;
                cereal_read_UIButtonInfo(&datad, eventd.uiButtonInfo);
                int id = datad.btnId;
                
                v->btns[id].btn_name = (char *) datad.btnName.str;
                v->btns[id].btn_label = (char *) datad.btnLabel.str;
                v->btns[id].btn_label2 = (char *) datad.btnLabel2.str;
                v->btns_status[id] = datad.btnStatus;
            } else if (eventd.which == cereal_UIEvent_uiCustomAlert) {
                struct cereal_UICustomAlert datad;
                cereal_read_UICustomAlert(&datad, eventd.uiCustomAlert);
                v->custom_message = (char *) datad.caText.str;
                v->custom_message_status = datad.caStatus;
            } else if (eventd.which == cereal_UIEvent_uiSetCar) {
                struct cereal_UISetCar datad;
                cereal_read_UISetCar(&datad, eventd.uiSetCar);
                v->car_model = (char *) datad.icCarName.str;
                v->car_folder = (char *) datad.icCarFolder.str;
            }
            capn_free(&ctx);
            zmq_msg_close(&msg);
        }
    }

 }

 static void bb_ui_init(UIState *s, BBUIVals *v) {
    memset(v, 0, sizeof(BBUIVals));
    v->car_model = "Tesla";
    v->car_folder = "tesla";
    v->label_last_modified = 0;
    v->status_last_modified = 0;
    //BB Define CAPNP sock
    v->uiButtonInfo_sock = zsock_new_sub(">tcp://127.0.0.1:8201", "");
    assert(v->uiButtonInfo_sock);
    v->uiButtonInfo_sock_raw = zsock_resolve(v->uiButtonInfo_sock);

    v->uiCustomAlert_sock = zsock_new_sub(">tcp://127.0.0.1:8202", "");
    assert(v->uiCustomAlert_sock);
    v->uiCustomAlert_sock_raw = zsock_resolve(v->uiCustomAlert_sock);

    v->uiSetCar_sock = zsock_new_sub(">tcp://127.0.0.1:8203", "");
    assert(v->uiSetCar_sock);
    v->uiSetCar_sock_raw = zsock_resolve(v->uiSetCar_sock);

    v->uiPlaySound_sock = zsock_new_sub(">tcp://127.0.0.1:8205", "");
    assert(v->uiPlaySound_sock);
    v->uiPlaySound_sock_raw = zsock_resolve(v->uiPlaySound_sock);

    v->uiButtonStatus_sock = zsock_new_pub(">tcp://127.0.0.1:8204");
    assert(v->uiButtonStatus_sock);
    v->uiButtonStatus_sock_raw = zsock_resolve(v->uiButtonStatus_sock);

    v->uiUpdate_sock = zsock_new_sub(">tcp://127.0.0.1:8206", "");
    assert(v->uiUpdate_sock);
    v->uiUpdate_sock_raw = zsock_resolve(v->uiUpdate_sock);
 }
 
//BB END: functions added for the display of various items

volatile int do_exit = 0;
static void set_do_exit(int sig) {
  do_exit = 1;
}

static void ui_init(UIState *s) {
  memset(s, 0, sizeof(UIState));

  pthread_mutex_init(&s->lock, NULL);
  pthread_cond_init(&s->bg_cond, NULL);
  //
  s->status = STATUS_DISENGAGED;
  // init connections

  s->thermal_sock = zsock_new_sub(">tcp://127.0.0.1:8005", "");
  assert(s->thermal_sock);
  s->thermal_sock_raw = zsock_resolve(s->thermal_sock);

  s->gps_sock = zsock_new_sub(">tcp://127.0.0.1:8032", "");
  assert(s->gps_sock);
  s->gps_sock_raw = zsock_resolve(s->gps_sock);
  

  s->live100_sock = zsock_new_sub(">tcp://127.0.0.1:8007", "");
  assert(s->live100_sock);
  s->live100_sock_raw = zsock_resolve(s->live100_sock);

  s->ipc_fd = -1;

  // init display
  s->fb = framebuffer_init("ui", 0x00010000, true,
                           &s->display, &s->surface, &s->fb_w, &s->fb_h);
  assert(s->fb);


  // init drawing
  s->vg = nvgCreateGLES3(NVG_ANTIALIAS | NVG_STENCIL_STROKES | NVG_DEBUG);
  assert(s->vg);

  s->font_courbd = nvgCreateFont(s->vg, "courbd", "../assets/courbd.ttf");
  assert(s->font_courbd >= 0);
  s->font_sans_regular = nvgCreateFont(s->vg, "sans-regular", "../assets/OpenSans-Regular.ttf");
  assert(s->font_sans_regular >= 0);
  s->font_sans_semibold = nvgCreateFont(s->vg, "sans-semibold", "../assets/OpenSans-SemiBold.ttf");
  assert(s->font_sans_semibold >= 0);
  s->font_sans_bold = nvgCreateFont(s->vg, "sans-bold", "../assets/OpenSans-Bold.ttf");
  assert(s->font_sans_bold >= 0);

  assert(s->img_wheel >= 0);
  s->img_wheel = nvgCreateImage(s->vg, "../assets/img_chffr_wheel.png", 1);

  // init gl
  s->frame_program = load_program(frame_vertex_shader, frame_fragment_shader);
  assert(s->frame_program);

  s->frame_pos_loc = glGetAttribLocation(s->frame_program, "aPosition");
  s->frame_texcoord_loc = glGetAttribLocation(s->frame_program, "aTexCoord");

  s->frame_texture_loc = glGetUniformLocation(s->frame_program, "uTexture");
  s->frame_transform_loc = glGetUniformLocation(s->frame_program, "uTransform");

  s->line_program = load_program(line_vertex_shader, line_fragment_shader);
  assert(s->line_program);

  s->line_pos_loc = glGetAttribLocation(s->line_program, "aPosition");
  s->line_color_loc = glGetAttribLocation(s->line_program, "aColor");
  s->line_transform_loc = glGetUniformLocation(s->line_program, "uTransform");

  glViewport(0, 0, s->fb_w, s->fb_h);

  glDisable(GL_DEPTH_TEST);

  assert(glGetError() == GL_NO_ERROR);


  {
    char *value;
    const int result = read_db_value(NULL, "Passive", &value, NULL);
    if (result == 0) {
      s->passive = value[0] == '1';
      free(value);
    }
  }
}

int main() {
    int err;
    bool s_received = false;
    setpriority(PRIO_PROCESS, 0, -14);

    zsys_handler_set(NULL);
    signal(SIGINT, (sighandler_t)set_do_exit);

    BBUIVals bbuivals;
    BBUIVals *v = &bbuivals;
    UIState uistate;
    UIState *s = &uistate;
    ui_init(s);
    bb_ui_init(s,v);

    TouchState touch = {0};
    touch_init(&touch);

    
    while (!do_exit) {

        bb_ui_poll_update(s,v);

        int touch_x = -1, touch_y = -1;
        int touched = touch_poll(&touch, &touch_x, &touch_y, s->awake ? 0 : 100);
        if (touched == 1) {
            // BB check touch area
            bb_handle_ui_touch(s,v,touch_x,touch_y);
        }
    }

    return 0;

}
