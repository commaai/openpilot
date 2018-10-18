#include "cereal/gen/c/ui.capnp.h"




vec3 bb_car_space_to_full_frame(const  UIState *s, vec4 car_space_projective) {
  const UIScene *scene = &s->scene;

  // We'll call the car space point p.
  // First project into normalized image coordinates with the extrinsics matrix.
  const vec4 Ep4 = matvecmul(scene->extrinsic_matrix, car_space_projective);

  // The last entry is zero because of how we store E (to use matvecmul).
  const vec3 Ep = {{Ep4.v[0], Ep4.v[1], Ep4.v[2]}};
  const vec3 KEp = matvecmul3(intrinsic_matrix, Ep);

  // Project.
  const vec3 p_image = {{KEp.v[0] / KEp.v[2], KEp.v[1] / KEp.v[2], 1.}};
  return p_image;
}


void bb_ui_draw_vision_alert( UIState *s, int va_size, int va_color,
                                  const char* va_text1, const char* va_text2) {
  const UIScene *scene = &s->scene;
  int ui_viz_rx = scene->ui_viz_rx;
  int ui_viz_rw = scene->ui_viz_rw;
  bool hasSidebar = !s->scene.uilayout_sidebarcollapsed;
  bool mapEnabled = s->scene.uilayout_mapenabled;
  bool longAlert1 = strlen(va_text1) > 15;

  const uint8_t *color = alert_colors[va_color];
  const int alr_s = alert_sizes[va_size];
  const int alr_x = ui_viz_rx-(mapEnabled?(hasSidebar?nav_w:(nav_ww)):0)-bdr_s;
  const int alr_w = ui_viz_rw+(mapEnabled?(hasSidebar?nav_w:(nav_ww)):0)+(bdr_s*2);
  const int alr_h = alr_s+(va_size==ALERTSIZE_NONE?0:bdr_s);
  const int alr_y = vwp_h-alr_h;

  nvgBeginPath(s->vg);
  nvgRect(s->vg, alr_x, alr_y, alr_w, alr_h);
  nvgFillColor(s->vg, nvgRGBA(color[0],color[1],color[2],(color[3]*s->alert_blinking_alpha)));
  nvgFill(s->vg);

  nvgBeginPath(s->vg);
  NVGpaint gradient = nvgLinearGradient(s->vg, alr_x, alr_y, alr_x, alr_y+alr_h,
                        nvgRGBAf(0.0,0.0,0.0,0.05), nvgRGBAf(0.0,0.0,0.0,0.35));
  nvgFillPaint(s->vg, gradient);
  nvgRect(s->vg, alr_x, alr_y, alr_w, alr_h);
  nvgFill(s->vg);

  nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 255));
  nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_BASELINE);

  if (va_size == ALERTSIZE_SMALL) {
    nvgFontFace(s->vg, "sans-semibold");
    nvgFontSize(s->vg, 40*2.5);
    nvgText(s->vg, alr_x+alr_w/2, alr_y+alr_h/2+15, va_text1, NULL);
  } else if (va_size== ALERTSIZE_MID) {
    nvgFontFace(s->vg, "sans-bold");
    nvgFontSize(s->vg, 48*2.5);
    nvgText(s->vg, alr_x+alr_w/2, alr_y+alr_h/2-45, va_text1, NULL);
    nvgFontFace(s->vg, "sans-regular");
    nvgFontSize(s->vg, 36*2.5);
    nvgText(s->vg, alr_x+alr_w/2, alr_y+alr_h/2+75, va_text2, NULL);
  } else if (va_size== ALERTSIZE_FULL) {
    nvgFontSize(s->vg, (longAlert1?72:96)*2.5);
    nvgFontFace(s->vg, "sans-bold");
    nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
    nvgTextBox(s->vg, alr_x, alr_y+(longAlert1?360:420), alr_w-60, va_text1, NULL);
    nvgFontSize(s->vg, 48*2.5);
    nvgFontFace(s->vg, "sans-regular");
    nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_BOTTOM);
    nvgTextBox(s->vg, alr_x, alr_h-(longAlert1?300:360), alr_w-60, va_text2, NULL);
  }
}



void bb_ui_draw_car(  UIState *s) {
  // replaces the draw_chevron function when button in mid position
  //static void draw_chevron(UIState *s, float x_in, float y_in, float sz,
  //                        NVGcolor fillColor, NVGcolor glowColor) {
  if (!s->scene.lead_status) {
    //no lead car to draw
    return;
  }
  const UIScene *scene = &s->scene;
  float x_in = scene->lead_d_rel+2.7;
  float y_in = scene->lead_y_rel;
  float sz = 1.0;

  nvgSave(s->vg);
  nvgTranslate(s->vg, 240.0f, 0.0);
  nvgTranslate(s->vg, -1440.0f / 2, -1080.0f / 2);
  nvgScale(s->vg, 2.0, 2.0);
  nvgScale(s->vg, 1440.0f / s->rgb_width, 1080.0f / s->rgb_height);

  const vec4 p_car_space = (vec4){{x_in, y_in, 0., 1.}};
  const vec3 p_full_frame = bb_car_space_to_full_frame(s, p_car_space);

  float x = p_full_frame.v[0];
  float y = p_full_frame.v[1];

  // glow

  float bbsz =0.1*  0.1 + 0.6 * (180-x_in*3.3-20)/180;
  if (bbsz < 0.1) bbsz = 0.1;
  if (bbsz > 0.6) bbsz = 0.6;
  float car_alpha = .8;
  float car_w = 750 * bbsz;
  float car_h = 531 * bbsz;
  float car_y = y - car_h/2;
  float car_x = x - car_w/2;
  
  nvgBeginPath(s->vg);
  NVGpaint imgPaint = nvgImagePattern(s->vg, car_x, car_y,
  car_w, car_h, 0, s->b.img_car, car_alpha);
  nvgRect(s->vg, car_x, car_y, car_w, car_h);
  nvgFillPaint(s->vg, imgPaint);
  nvgFill(s->vg);
  nvgRestore(s->vg);
}

void bb_draw_lane_fill ( UIState *s) {

  const UIScene *scene = &s->scene;

  nvgSave(s->vg);
  nvgTranslate(s->vg, 240.0f, 0.0); // rgb-box space
  nvgTranslate(s->vg, -1440.0f / 2, -1080.0f / 2); // zoom 2x
  nvgScale(s->vg, 2.0, 2.0);
  nvgScale(s->vg, 1440.0f / s->rgb_width, 1080.0f / s->rgb_height);
  nvgBeginPath(s->vg);

  //BB let the magic begin
  //define variables
  PathData path;
  float *points;
  float off;
  NVGcolor bb_color = nvgRGBA(138, 140, 142,180);
  bool started = false;
  bool is_ghost = true;
  //left lane, first init
  path = scene->model.left_lane;
  points = path.points;
  off = 0.025*path.prob;
  //draw left
  for (int i=0; i<49; i++) {
    float px = (float)i;
    float py = points[i] - off;
    vec4 p_car_space = (vec4){{px, py, 0., 1.}};
    vec3 p_full_frame = bb_car_space_to_full_frame(s, p_car_space);
    float x = p_full_frame.v[0];
    float y = p_full_frame.v[1];
    if (x < 0 || y < 0.) {
      continue;
    }
    if (!started) {
      nvgMoveTo(s->vg, x, y);
      started = true;
    } else {
      nvgLineTo(s->vg, x, y);
    }
  }
  //right lane, first init
  path = scene->model.right_lane;
  points = path.points;
  off = 0.025*path.prob;
  //draw right
  for (int i=49; i>0; i--) {
    float px = (float)i;
    float py = is_ghost?(points[i]-off):(points[i]+off);
    vec4 p_car_space = (vec4){{px, py, 0., 1.}};
    vec3 p_full_frame = bb_car_space_to_full_frame(s, p_car_space);
    float x = p_full_frame.v[0];
    float y = p_full_frame.v[1];
    if (x < 0 || y < 0.) {
      continue;
    }
    nvgLineTo(s->vg, x, y);
  }

  nvgClosePath(s->vg);
  nvgFillColor(s->vg, bb_color);
  nvgFill(s->vg);
  nvgRestore(s->vg);

}





long bb_currentTimeInMilis() {
    struct timespec res;
    clock_gettime(CLOCK_MONOTONIC, &res);
    return (res.tv_sec * 1000) + res.tv_nsec/1000000;
}

int bb_get_status( UIState *s) {
    //BB select status based on main s->status w/ priority over s->b.custom_message_status
    int bb_status = -1;
    if ((s->status == STATUS_ENGAGED) && (s->b.custom_message_status > STATUS_ENGAGED)) {
      bb_status = s->b.custom_message_status;
    } else {
      bb_status = s->status;
    }
    return bb_status;
}

int bb_ui_draw_measure( UIState *s,  const char* bb_value, const char* bb_uom, const char* bb_label, 
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


bool bb_handle_ui_touch( UIState *s, int touch_x, int touch_y) {
  for(int i=0; i<6; i++) {
    if (s->b.btns_r[i] > 0) {
      if ((abs(touch_x - s->b.btns_x[i]) < s->b.btns_r[i]) && (abs(touch_y - s->b.btns_y[i]) < s->b.btns_r[i])) {
        //found it; change the status
        if (s->b.btns_status[i] > 0) {
          s->b.btns_status[i] = 0;
        } else {
          s->b.btns_status[i] = 1;
        }
        //now let's send the cereal
        
        struct capn c;
        capn_init_malloc(&c);
        struct capn_ptr cr = capn_root(&c);
        struct capn_segment *cs = cr.seg;
        
        struct cereal_UIButtonStatus btn_d;
        btn_d.btnId = i;
        btn_d.btnStatus = s->b.btns_status[i];

        cereal_UIButtonStatus_ptr btn_p = cereal_new_UIButtonStatus(cs);
        
        cereal_write_UIButtonStatus(&btn_d, btn_p);
        int setp_ret = capn_setp(capn_root(&c), 0, btn_p.p);
        assert(setp_ret == 0);

        uint8_t buf[4096];
        ssize_t rs = capn_write_mem(&c, buf, sizeof(buf), 0);

        capn_free(&c);
        
        zmq_send(s->b.uiButtonStatus_sock_raw, buf,rs,0);
        
        return true;
      }
    }
  }
  return false;
};

int bb_get_button_status( UIState *s, char *btn_name) {
  int ret_status = -1;
  for (int i = 0; i< 6; i++) {
    if (strcmp(s->b.btns[i].btn_name,btn_name)==0) {
      ret_status = s->b.btns_status[i];
    }
  }
  return ret_status;
}

void bb_draw_button( UIState *s, int btn_id) {
  const UIScene *scene = &s->scene;

  int viz_button_x = 0;
  int viz_button_y = (box_y + (bdr_s*1.5)) + 20;
  int viz_button_w = 140;
  int viz_button_h = 140;

  char *btn_text, *btn_text2;
  
  int delta_x = viz_button_w * 1.1;
  int delta_y = 0;
  int dx1 = 200;
  int dx2 = 190;
  int dy = 0;

  if (s->b.tri_state_switch ==2) {
    delta_y = delta_x;
    delta_x = 0;
    dx1 = 20;
    dx2 = 160;
    dy = 200;
  }

  if (btn_id >2) {
    viz_button_x = scene->ui_viz_rx + scene->ui_viz_rw - (bdr_s*2) -dx2;
    viz_button_x -= (6-btn_id) * delta_x ;
    viz_button_y += (btn_id-3) * delta_y + dy;
    
  } else {
    viz_button_x = scene->ui_viz_rx + (bdr_s*2) + dx1;
    viz_button_x +=  (btn_id) * delta_x;
    viz_button_y += btn_id * delta_y + dy;
  }
  

  btn_text = s->b.btns[btn_id].btn_label;
  btn_text2 = s->b.btns[btn_id].btn_label2;
  
  if (strcmp(btn_text,"")==0) {
    s->b.btns_r[btn_id] = 0;
  } else {
    s->b.btns_r[btn_id]= (int)((viz_button_w + viz_button_h)/4);
  }
  s->b.btns_x[btn_id]=viz_button_x + s->b.btns_r[btn_id];
  s->b.btns_y[btn_id]=viz_button_y + s->b.btns_r[btn_id];
  if (s->b.btns_r[btn_id] == 0) {
    return;
  }
  
  nvgBeginPath(s->vg);
  nvgRoundedRect(s->vg, viz_button_x, viz_button_y, viz_button_w, viz_button_h, 80);
  nvgStrokeWidth(s->vg, 12);

  
  if (s->b.btns_status[btn_id] ==0) {
    //disabled - red
    nvgStrokeColor(s->vg, nvgRGBA(255, 0, 0, 200));
    if (strcmp(btn_text2,"")==0) {
      btn_text2 = "Off";
    }
  } else
  if (s->b.btns_status[btn_id] ==1) {
    //enabled - white
    nvgStrokeColor(s->vg, nvgRGBA(255,255,255,200));
    nvgStrokeWidth(s->vg, 4);
    if (strcmp(btn_text2,"")==0) {
      btn_text2 = "Ready";
    }
  } else
  if (s->b.btns_status[btn_id] ==2) {
    //active - green
    nvgStrokeColor(s->vg, nvgRGBA(28, 204,98,200));
    if (strcmp(btn_text2,"")==0) {
      btn_text2 = "Active";
    }
  } else
  if (s->b.btns_status[btn_id] ==9) {
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
  nvgText(s->vg, viz_button_x+viz_button_w/2, viz_button_y + 112, btn_text2, NULL);

  nvgFontFace(s->vg, "sans-semibold");
  nvgFontSize(s->vg, 28*2.5);
  nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 255));
  nvgText(s->vg, viz_button_x+viz_button_w/2, viz_button_y + 85,btn_text, NULL);
}

void bb_draw_buttons( UIState *s) {
  for (int i = 0; i < 6; i++) {
    bb_draw_button(s,i);
  }
}

void bb_ui_draw_custom_alert( UIState *s) {
    if ((strlen(s->b.custom_message) > 0) && (strlen(s->scene.alert_text1)==0)){
      if (!((bb_get_button_status(s,"msg") == 0) && (s->b.custom_message_status<=3))) {
        bb_ui_draw_vision_alert(s, ALERTSIZE_SMALL, s->b.custom_message_status,
                              s->b.custom_message,"");
      }
    } 
}


void bb_ui_draw_measures_left( UIState *s, int bb_x, int bb_y, int bb_w ) {
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
			if((int)(s->b.maxCpuTemp/10) > 65) {
				val_color = nvgRGBA(255, 188, 3, 200);
			}
			if((int)(s->b.maxCpuTemp/10) > 85) {
				val_color = nvgRGBA(255, 0, 0, 200);
			}
			// temp is alway in C * 10
			if (s->is_metric) {
				 snprintf(val_str, sizeof(val_str), "%d C", (int)(s->b.maxCpuTemp/10));
			} else {
				 snprintf(val_str, sizeof(val_str), "%d F", (int)(32+9*(s->b.maxCpuTemp/10)/5));
			}
		snprintf(uom_str, sizeof(uom_str), "");
		bb_h +=bb_ui_draw_measure(s,  val_str, uom_str, "CPU TEMP", 
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
		if((int)(s->b.maxBatTemp/1000) > 40) {
			val_color = nvgRGBA(255, 188, 3, 200);
		}
		if((int)(s->b.maxBatTemp/1000) > 50) {
			val_color = nvgRGBA(255, 0, 0, 200);
		}
		// temp is alway in C * 1000
		if (s->is_metric) {
			 snprintf(val_str, sizeof(val_str), "%d C", (int)(s->b.maxBatTemp/1000));
		} else {
			 snprintf(val_str, sizeof(val_str), "%d F", (int)(32+9*(s->b.maxBatTemp/1000)/5));
		}
		snprintf(uom_str, sizeof(uom_str), "");
		bb_h +=bb_ui_draw_measure(s,  val_str, uom_str, "BAT TEMP", 
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
	    if(s->b.gpsAccuracy > 0.59) {
	       val_color = nvgRGBA(255, 188, 3, 200);
	    }
	    if(s->b.gpsAccuracy > 0.99) {
	       val_color = nvgRGBA(255, 0, 0, 200);
	    }


		// gps accuracy is always in meters
		if (s->is_metric) {
			 snprintf(val_str, sizeof(val_str), "%d", (int)(s->b.gpsAccuracy*100.0));
		} else {
			 snprintf(val_str, sizeof(val_str), "%.1f", s->b.gpsAccuracy * 3.28084 * 12);
		}
		if (s->is_metric) {
			snprintf(uom_str, sizeof(uom_str), "cm");;
		} else {
			snprintf(uom_str, sizeof(uom_str), "in");
		}
		bb_h +=bb_ui_draw_measure(s,  val_str, uom_str, "GPS PREC", 
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
		if(s->b.freeSpace < 0.4) {
			val_color = nvgRGBA(255, 188, 3, 200);
		}
		if(s->b.freeSpace < 0.2) {
			val_color = nvgRGBA(255, 0, 0, 200);
		}

		snprintf(val_str, sizeof(val_str), "%.1f", s->b.freeSpace* 100);
		snprintf(uom_str, sizeof(uom_str), "%%");

		bb_h +=bb_ui_draw_measure(s, val_str, uom_str, "FREE", 
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


void bb_ui_draw_measures_right( UIState *s, int bb_x, int bb_y, int bb_w ) {
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
		bb_h +=bb_ui_draw_measure(s,  val_str, uom_str, "REL DIST", 
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
		bb_h +=bb_ui_draw_measure(s,  val_str, uom_str, "REL SPD", 
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
			if(((int)(s->b.angleSteers) < -6) || ((int)(s->b.angleSteers) > 6)) {
				val_color = nvgRGBA(255, 188, 3, 200);
			}
			if(((int)(s->b.angleSteers) < -12) || ((int)(s->b.angleSteers) > 12)) {
				val_color = nvgRGBA(255, 0, 0, 200);
			}
			// steering is in degrees
			snprintf(val_str, sizeof(val_str), "%.1f",(s->b.angleSteers));

	    snprintf(uom_str, sizeof(uom_str), "deg");
		bb_h +=bb_ui_draw_measure(s,  val_str, uom_str, "STEER", 
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
			if(((int)(s->b.angleSteersDes) < -6) || ((int)(s->b.angleSteersDes) > 6)) {
				val_color = nvgRGBA(255, 188, 3, 200);
			}
			if(((int)(s->b.angleSteersDes) < -12) || ((int)(s->b.angleSteersDes) > 12)) {
				val_color = nvgRGBA(255, 0, 0, 200);
			}
			// steering is in degrees
			snprintf(val_str, sizeof(val_str), "%.1f",(s->b.angleSteersDes));

	    snprintf(uom_str, sizeof(uom_str), "deg");
		bb_h +=bb_ui_draw_measure(s,  val_str, uom_str, "DES STEER", 
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
void ui_draw_vision_grid( UIState *s) {
  const UIScene *scene = &s->scene;
  bool is_cruise_set = false;//(s->scene.v_cruise != 0 && s->scene.v_cruise != 255);
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

void bb_ui_draw_logo( UIState *s) {
  if ((s->status != STATUS_DISENGAGED) && (s->status != STATUS_STOPPED)) { //(s->status != STATUS_DISENGAGED) {//
    return;
  }
  int rduration = 8000;
  int logot = (bb_currentTimeInMilis() % rduration);
  int logoi = s->b.img_logo;
  if ((logot > (int)(rduration/4)) && (logot < (int)(3*rduration/4))) {
    logoi = s->b.img_logo2;
  }
  if (logot < (int)(rduration/2)) {
    logot = logot - (int)(rduration/4);
  } else {
    logot = logot - (int)(3*rduration/4);
  }
  float logop = fabs(4.0*logot/rduration);
  const UIScene *scene = &s->scene;
  const int ui_viz_rx = scene->ui_viz_rx;
  const int ui_viz_rw = scene->ui_viz_rw;
  const int viz_event_w = (int)(820 * logop);
  const int viz_event_h = 820;
  const int viz_event_x = (ui_viz_rx + (ui_viz_rw - viz_event_w - bdr_s*2)/2);
  const int viz_event_y = 200;
  bool is_engageable = scene->engageable;
  float viz_event_alpha = 1.0f;
  nvgBeginPath(s->vg);
  NVGpaint imgPaint = nvgImagePattern(s->vg, viz_event_x, viz_event_y,
  viz_event_w, viz_event_h, 0, logoi, viz_event_alpha);
  nvgRect(s->vg, viz_event_x, viz_event_y, (int)viz_event_w, viz_event_h);
  nvgFillPaint(s->vg, imgPaint);
  nvgFill(s->vg);
}



void bb_ui_draw_UI( UIState *s) {
  //get 3-state switch position
  int tri_state_fd;
  char buffer[10];
  if  (bb_currentTimeInMilis() - s->b.tri_state_switch_last_read > 2000)  {
    tri_state_fd = open ("/sys/devices/virtual/switch/tri-state-key/state", O_RDONLY);
    //if we can't open then switch should be considered in the middle, nothing done
    if (tri_state_fd == -1) {
      s->b.tri_state_switch = 2;
    } else {
      read (tri_state_fd, &buffer, 10);
      s->b.tri_state_switch = buffer[0] -48;
      close(tri_state_fd);
      s->b.tri_state_switch_last_read = bb_currentTimeInMilis();
    }
  }
  
  draw_date_time(s);
	
  if (s->b.tri_state_switch == 1) {
	  const UIScene *scene = &s->scene;
	  const int bb_dml_w = 180;
	  const int bb_dml_x =  (scene->ui_viz_rx + (bdr_s*2));
	  const int bb_dml_y = (box_y + (bdr_s*1.5))+220;
	  
	  const int bb_dmr_w = 180;
	  const int bb_dmr_x = scene->ui_viz_rx + scene->ui_viz_rw - bb_dmr_w - (bdr_s*2) ; 
	  const int bb_dmr_y = (box_y + (bdr_s*1.5))+220;
    bb_ui_draw_measures_left(s,bb_dml_x, bb_dml_y, bb_dml_w );
    bb_ui_draw_measures_right(s,bb_dmr_x, bb_dmr_y, bb_dmr_w );
    bb_draw_buttons(s);
    bb_ui_draw_custom_alert(s);
    bb_ui_draw_logo(s);
	 }

   if (s->b.tri_state_switch ==2) {
	 	const UIScene *scene = &s->scene;
	  const int bb_dml_w = 180;
	  const int bb_dml_x =  (scene->ui_viz_rx + (bdr_s*2));
	  const int bb_dml_y = (box_y + (bdr_s*1.5))+220;
	  
	  const int bb_dmr_w = 180;
	  const int bb_dmr_x = scene->ui_viz_rx + scene->ui_viz_rw - bb_dmr_w - (bdr_s*2) ; 
	  const int bb_dmr_y = (box_y + (bdr_s*1.5))+220;
    bb_draw_buttons(s);
    bb_ui_draw_custom_alert(s);
    bb_ui_draw_logo(s);
    //bb_ui_draw_car(s);
	 }
	 if (s->b.tri_state_switch ==3) {
	 	ui_draw_vision_grid(s);
	 }
}



void bb_ui_init(UIState *s) {

    //BB INIT
    s->b.shouldDrawFrame = true;
    s->status = STATUS_DISENGAGED;
    strcpy(s->b.car_model,"Tesla");
    strcpy(s->b.car_folder,"tesla");
    s->b.tri_state_switch = -1;
    s->b.tri_state_switch_last_read = 0;

    //BB Define CAPNP sock
    s->b.uiButtonInfo_sock = zsock_new_sub(">tcp://127.0.0.1:8201", "");
    assert(s->b.uiButtonInfo_sock);
    s->b.uiButtonInfo_sock_raw = zsock_resolve(s->b.uiButtonInfo_sock);

    s->b.uiCustomAlert_sock = zsock_new_sub(">tcp://127.0.0.1:8202", "");
    assert(s->b.uiCustomAlert_sock);
    s->b.uiCustomAlert_sock_raw = zsock_resolve(s->b.uiCustomAlert_sock);

    s->b.uiSetCar_sock = zsock_new_sub(">tcp://127.0.0.1:8203", "");
    assert(s->b.uiSetCar_sock);
    s->b.uiSetCar_sock_raw = zsock_resolve(s->b.uiSetCar_sock);

    s->b.uiPlaySound_sock = zsock_new_sub(">tcp://127.0.0.1:8205", "");
    assert(s->b.uiPlaySound_sock);
    s->b.uiPlaySound_sock_raw = zsock_resolve(s->b.uiPlaySound_sock);

    s->b.uiButtonStatus_sock = zsock_new_pub("@tcp://127.0.0.1:8204");
    assert(s->b.uiButtonStatus_sock);
    s->b.uiButtonStatus_sock_raw = zsock_resolve(s->b.uiButtonStatus_sock);

    s->b.gps_sock = zsock_new_sub(">tcp://127.0.0.1:8032", "");
    assert(s->b.gps_sock);
    s->b.gps_sock_raw = zsock_resolve(s->b.gps_sock);

    //BB Load Images
    s->b.img_logo = nvgCreateImage(s->vg, "../assets/img_spinner_comma.png", 1);
    s->b.img_logo2 = nvgCreateImage(s->vg, "../assets/img_spinner_comma2.png", 1);
    s->b.img_car = nvgCreateImage(s->vg, "../assets/img_car_tesla.png", 1);
}

void bb_ui_play_sound( UIState *s, int sound) {
    char* snd_command;
    int bts = bb_get_button_status(s,"sound");
    if ((bts > 0) || (bts == -1)) {
        asprintf(&snd_command, "python /data/openpilot/selfdrive/car/modules/snd/playsound.py %d &", sound);
        system(snd_command);
    }
}

void bb_ui_set_car( UIState *s, char *model, char *folder) {
    strcpy(s->b.car_model,model);
    strcpy(s->b.car_folder, folder);
}

void  bb_ui_poll_update( UIState *s) {
    int err;
    zmq_pollitem_t bb_polls[8] = {{0}};
    bb_polls[0].socket = s->b.uiButtonInfo_sock_raw;
    bb_polls[0].events = ZMQ_POLLIN;
    bb_polls[1].socket = s->b.uiCustomAlert_sock_raw;
    bb_polls[1].events = ZMQ_POLLIN;
    bb_polls[2].socket = s->b.uiSetCar_sock_raw;
    bb_polls[2].events = ZMQ_POLLIN;
    bb_polls[3].socket = s->b.uiPlaySound_sock_raw;
    bb_polls[3].events = ZMQ_POLLIN;
    bb_polls[4].socket = s->b.gps_sock_raw;
    bb_polls[4].events = ZMQ_POLLIN;
	
    while (true) {
	    
	    
        int ret = zmq_poll(bb_polls, 5, 0);
        if (ret < 0) {
          LOGW("bb poll failed (%d)", ret);
          break;
        }
        if (ret == 0) {
          //LOGW("poll empty");
          break;
        }

        if (bb_polls[0].revents) {
          //button info socket
          zmq_msg_t msg;
          err = zmq_msg_init(&msg);
          assert(err == 0);
          err = zmq_msg_recv(&msg, s->b.uiButtonInfo_sock_raw, 0);
          assert(err >= 0);

          struct capn ctx;
          capn_init_mem(&ctx, zmq_msg_data(&msg), zmq_msg_size(&msg), 0);

          cereal_UIButtonInfo_ptr stp;
          stp.p = capn_getp(capn_root(&ctx), 0, 1);
          struct cereal_UIButtonInfo datad;
          cereal_read_UIButtonInfo(&datad, stp);

          int id = datad.btnId;
          //LOGW("got button info: ID = (%d)", id);
          strcpy(s->b.btns[id].btn_name,(char *)datad.btnName.str);
          strcpy(s->b.btns[id].btn_label, (char *)datad.btnLabel.str);
          strcpy(s->b.btns[id].btn_label2, (char *)datad.btnLabel2.str);
          s->b.btns_status[id] = datad.btnStatus;
          
          capn_free(&ctx);
          zmq_msg_close(&msg);
        }
	if (bb_polls[1].revents) {
          //custom alert socket
          zmq_msg_t msg;
          err = zmq_msg_init(&msg);
          assert(err == 0);
          err = zmq_msg_recv(&msg, s->b.uiCustomAlert_sock_raw, 0);
          assert(err >= 0);

          struct capn ctx;
          capn_init_mem(&ctx, zmq_msg_data(&msg), zmq_msg_size(&msg), 0);

          cereal_UICustomAlert_ptr stp;
          stp.p = capn_getp(capn_root(&ctx), 0, 1);
          struct cereal_UICustomAlert  datad;
          cereal_read_UICustomAlert(&datad, stp);

          strcpy(s->b.custom_message,datad.caText.str);
          s->b.custom_message_status = datad.caStatus;
          
          capn_free(&ctx);
          zmq_msg_close(&msg);
          // wakeup bg thread since status changed
          pthread_cond_signal(&s->bg_cond);
        }
	if (bb_polls[2].revents) {
          //set car model socket
          zmq_msg_t msg;
          err = zmq_msg_init(&msg);
          assert(err == 0);
          err = zmq_msg_recv(&msg, s->b.uiSetCar_sock_raw, 0);
          assert(err >= 0);

          struct capn ctx;
          capn_init_mem(&ctx, zmq_msg_data(&msg), zmq_msg_size(&msg), 0);

          cereal_UISetCar_ptr stp;
          stp.p = capn_getp(capn_root(&ctx), 0, 1);
          struct cereal_UISetCar datad;
          cereal_read_UISetCar(&datad, stp);

          if ((strcmp(s->b.car_model,(char *) datad.icCarName.str) != 0) || (strcmp(s->b.car_folder, (char *) datad.icCarFolder.str) !=0)) {
            strcpy(s->b.car_model, (char *) datad.icCarName.str);
            strcpy(s->b.car_folder, (char *) datad.icCarFolder.str);
            LOGW("Car folder set (%s)", s->b.car_folder);

            if (strcmp(s->b.car_folder,"tesla")==0) {
              s->b.img_logo = nvgCreateImage(s->vg, "../assets/img_spinner_comma.png", 1);
              s->b.img_logo2 = nvgCreateImage(s->vg, "../assets/img_spinner_comma2.png", 1);
              LOGW("Spinning logo set for Tesla");
            } else if (strcmp(s->b.car_folder,"honda")==0) {
              s->b.img_logo = nvgCreateImage(s->vg, "../assets/img_spinner_comma.honda.png", 1);
              s->b.img_logo2 = nvgCreateImage(s->vg, "../assets/img_spinner_comma.honda2.png", 1);
              LOGW("Spinning logo set for Honda");
            } else if (strcmp(s->b.car_folder,"toyota")==0) {
              s->b.img_logo = nvgCreateImage(s->vg, "../assets/img_spinner_comma.toyota.png", 1);
              s->b.img_logo2 = nvgCreateImage(s->vg, "../assets/img_spinner_comma.toyota2.png", 1);
              LOGW("Spinning logo set for Toyota");
            };
          }
          capn_free(&ctx);
          zmq_msg_close(&msg);
        }
	if (bb_polls[3].revents) {
          //play sound socket
          zmq_msg_t msg;
          err = zmq_msg_init(&msg);
          assert(err == 0);
          err = zmq_msg_recv(&msg, s->b.uiPlaySound_sock_raw, 0);
          assert(err >= 0);

          struct capn ctx;
          capn_init_mem(&ctx, zmq_msg_data(&msg), zmq_msg_size(&msg), 0);

          cereal_UIPlaySound_ptr stp;
          stp.p = capn_getp(capn_root(&ctx), 0, 1);
          struct cereal_UIPlaySound datad;
          cereal_read_UIPlaySound(&datad, stp);

          int snd = datad.sndSound;
          bb_ui_play_sound(s,snd);
          
          capn_free(&ctx);
          zmq_msg_close(&msg);
        }
	if (bb_polls[4].revents) {
            // gps socket
            zmq_msg_t msg;
            err = zmq_msg_init(&msg);
            assert(err == 0);
            err = zmq_msg_recv(&msg, s->b.gps_sock_raw, 0);
            assert(err >= 0);

            struct capn ctx;
            capn_init_mem(&ctx, zmq_msg_data(&msg), zmq_msg_size(&msg), 0);

            cereal_Event_ptr eventp;
            eventp.p = capn_getp(capn_root(&ctx), 0, 1);
            struct cereal_Event eventd;
            cereal_read_Event(&eventd, eventp);

            struct cereal_GpsLocationData datad;
            cereal_read_GpsLocationData(&datad, eventd.gpsLocation);

            s->b.gpsAccuracy = datad.accuracy;
            if (s->b.gpsAccuracy>100)
            {
                s->b.gpsAccuracy=99.99;
            }
            else if (s->b.gpsAccuracy==0)
            {
                s->b.gpsAccuracy=99.8;
            }
            capn_free(&ctx);
            zmq_msg_close(&msg);
        }
            
    }
}

