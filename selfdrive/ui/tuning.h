/**************************************
  Tuning UI
  Author: pjlao307

  This OpenPilot mod allows you to dynamically modify variables used by OpenPilot.
  The purpose of this mod is to make it easier to tweak certain variables instead of
  having to modify code, recompile, reboot after every change.

  To use this mod you need to do 2 things:

  1. Create a file called /sdcard/tuning/params.txt

     Copy the file called params.example.txt into /sdcard/tuning (create the directory if needed).  
     You will need to specify which variables you want the Tuning mod to manage by adding them
     to that file.

  2. Modify OpenPilot code that uses the variable so that it is read from this file instead of hard coded.  This is left for the user to figure out and implement.

  For questions or info about this mod, visit the comma slack channel #mod-tuning

  CHANGE LOG:

  v0.0.4.1 - Add ability to use named presets
  v0.0.4 - Add ability to specify "presets" for the parameter list
           Presets need to be labeled as "PRESET=#" where # starts with 1
  v0.0.3 - Increase precision to 6 decimal places
  v0.0.2 - Added 'press and hold' on the +/- buttons
         - Swapped positions of +/- buttons
  v0.0.1 - Initial version

**************************************/

#include "custom_touch.h"

#define VERSION "0.0.4.1"

#define ERROR_NO_FILE 1
#define BTN_NONE 0
#define BTN_INCREASE 1
#define BTN_DECREASE 2
#define BTN_STEP_INCREASE 3
#define BTN_STEP_DECREASE 4
#define BTN_LEFT_ARROW 5
#define BTN_RIGHT_ARROW 6
#define BTN_TUNE 7
#define BTN_STEP 8
#define MAX_NUM_PARAMS 10 // max number of params we can track
#define MAX_FILE_BYTES 100000 // max bytes to write to file
#define MAX_PRESETS 10 // max number of presets we can track

bool debug = false;
int status = 0; // Status code to tell us if something went wrong

typedef struct ui_element {
  char name[50];
  int pos_x;
  int pos_y;
  int width;
  int height;
} ui_element;

int left_arrow_icon;
int right_arrow_icon;
int tune_icon;
int current_button = BTN_NONE;
bool tune_enabled = false;

int last_x = 0;
int last_y = 0;
int touch_frame_delay = 5; // Delay this many frames before reading touch events to slow down the reading
int touch_frame= 0; // Current frame

ui_element increase_button;
ui_element decrease_button;
ui_element property_button;
ui_element tune_values;

ui_element step_increase;
ui_element step_decrease;
ui_element step_text;
ui_element left_arrow_button;
ui_element right_arrow_button;
ui_element tune_button;
ui_element preset_button;

char rootdir[50] = "/sdcard/tuning";

char params_file[256] = "params.txt";
char tune_file[256] = "tune.txt"; // Stores the current preset values in a format that can be used by openpilot (python)
char python_file[256]; // stores values as pyton variable
bool init_tune = false; // Used to initialize stuff
int num_lines = 0;

// Preset variables
int preset = 0; // Preset index
int num_presets = 0; // Number of presets in the file
int num_preset_params[MAX_PRESETS]; // Number of presets in the file
char *preset_labels[MAX_PRESETS]; // Store preset names

double angles[MAX_PRESETS][MAX_NUM_PARAMS][3]; // List of angle values from the file
char property[50] = "CL_MAXD_A"; // What property to tune
char properties[50][MAX_NUM_PARAMS]; // List of property names
int current_property = -1; // Which property value to adjust when clicking the increase/decrease buttons
ui_element *property_buttons[MAX_NUM_PARAMS]; // Store the buttons that can be selected
double step = 0.000001; // Steps to adjust on each click
float delta_step = 0.000001; // Change step by this amount
int step_toggle = 0; // Change to preset toggle step

int param_index = 0; // Index to track which param label we're working with
int param_value_count[MAX_PRESETS][MAX_NUM_PARAMS]; // Store the number of elements for each param
char *param_labels[MAX_NUM_PARAMS]; // Store the names of the params

// Used to delay button highlighting
int frame_num = 0;
int frame_delay = 15; // delay for X frames

char *str_remove(const char *s, char ch) {
  int counter;
  char *new_str = malloc(strlen(s)+1);
  new_str[0] = '\0';
  while(s[counter] != '\0') {
    if (s[counter] != ch) {
      size_t str_len = strlen(new_str);
      new_str[str_len] = s[counter];
      new_str[str_len+1] = '\0';
    }
    counter++;
  }
  //printf("new_str: %s\n",new_str);

  return new_str;
}

char *get_full_path(char *fname) {
  if (debug) { printf("get_full_path\n"); }
  char *filename = malloc(sizeof(char *) * 128);
  char fullpath[128];
  snprintf(fullpath,sizeof(fullpath),"%s/%s", rootdir, fname);
  strcpy(filename, fullpath);
  return filename;
}

char **readfile(char *filename) {
  if (debug) { printf("readfile\n"); }
  FILE* file = fopen(filename, "r");

  if (!file) {
    //printf("Could not open file!\n");
    status = ERROR_NO_FILE;
    return NULL;
  }

  char line[256];
  char **lines = malloc(sizeof(char *) * MAX_NUM_PARAMS);
  num_lines = 0;
  size_t ln;

  while(fgets(line, sizeof(line), file)) {
    //printf("line: %s\n", line);
    ln = strlen(line) - 1;
    int cmp = strncmp(line,"#",1);
    if (cmp != 0 && ln > 0) {
      line[ln] = '\0';
      lines[num_lines] = malloc(256+1);
      strcpy(lines[num_lines],line);
      num_lines++;
    }
    else if (ln > 0) {
      printf("[TUNING MOD] Ignoring comment: %s",line);
    }
    else {
      printf("[TUNING MOD] Ignoring blank line\n");
    }
  }
  fclose(file);
  //printf("num_lines: %d\n", num_lines);

  /*
  for (int i=0; i < num_lines; i++) {
    printf("line %d: %s\n", i, lines[i]);
  }
  */
  return lines;
}

void parse_file(char *filename) {
  if (debug) { printf("parse_file\n"); }
  char **lines = readfile(filename);
  int param_num = 0;
  int sawPreset = 0;

  if (status == ERROR_NO_FILE) {
    return;
  }

  //printf("num_lines: %d\n",num_lines);

  for (int i=0; i < num_lines; i++) {
    int angle_index = 0;
    char* token_value;
    char* token_value2;
    char* token2;
    param_labels[param_num] = malloc(sizeof(char *));

    //printf("line %d: %s\n", i, lines[i]);
    char* token = strtok(lines[i], "=");
    //printf("token1: %s\n", token);
    int cmp = strncmp(token,"PRESET",6);
    if (cmp != 0) {
      param_labels[param_num] = token;
    }
    else {
      sawPreset++;
    }
    while (token != NULL ) {
      token = strtok(NULL, "=");
      char *end;
      if (token != NULL) {
        token_value = str_remove(str_remove(token,'['),']');
        //printf("token_value: %s\n", token_value);
        char* token2 = strtok(token_value,",");

        if (cmp == 0) {
          //preset = atoi(token2)-1;
          num_presets++;
          if (sawPreset > 1) {
            preset++;
          }
          preset_labels[preset] = token_value;
          //printf("%s: %d\n",token_value,preset);
          param_num = 0;
        }
        else {
          //printf("token2: (%d) %s\n", preset, token2);
          angles[preset][param_num][angle_index++] = strtod(token2,&end);
          while (token2 != NULL) {
            token2 = strtok(NULL, ",");
            if (token2 != NULL) {
              token_value2 = token2;
              angles[preset][param_num][angle_index++] = strtod(token_value2,&end);
              //printf("token_value2: %s\n", token_value2);
            }
          }
        }
      }
    }
    param_value_count[preset][param_num] = angle_index;
    if (cmp != 0) {
      param_num++;
      num_preset_params[preset] = param_num;
    }
  }

  preset = 0; // Reset preset index back to default

/*
  for (int i=0; i < num_presets; i++) {
    int num_params = num_preset_params[i];
    for (int j=0; j < num_params; j++) {
      printf("Preset %d - %s: %s: %d indexes\n",i,preset_labels[i],param_labels[j],param_value_count[i][j]);
    }
  }
  for (int i=0; i < num_presets; i++) {
    int num_params = num_preset_params[i];
    for (int j=0; j < num_params; j++) {
      int num_elements = param_value_count[i][j];
      for (int k=0; k < num_elements; k++) {
        printf("Preset %d: %d, %d: %.7f\n", i, j, k, angles[i][j][k]);
      }
    }
  }
*/
}

int numChars(char *str) {
  int i=0;

  while(str[i] != '\0') {
    i++;
  }

  return i;
}

void read_file_values(char *fname) {
  if (debug) { printf("read_file_values\n"); }
  char *filename = get_full_path(fname);
  //printf("file: %s\n",filename);
  parse_file(filename);
} 

void write_current_params() {
  char content[MAX_FILE_BYTES] = "";
  char line[128] = "";
  int num_params = num_preset_params[preset];

  for (int i=0; i < num_params; i++) {
    int num_elements = param_value_count[preset][i];
    char temp[128] = "";
    for (int j=0; j < num_elements; j++) {
      char value_str[50];
      snprintf(value_str,sizeof(value_str),"%.6f",angles[preset][i][j]);
      strcat(temp, value_str);
      if (j < num_elements-1) {
        strcat(temp, ",");
      }
      //printf("%d, %d: %.36\n", i, j, angles[preset][i][j]);
    }
    char line[128] = "";
    snprintf(line,sizeof(line),"%s=[%s]\n",param_labels[i],temp);
    //printf("line: %s\n",line);
    strcat(content,line);
  }

  char *filename = get_full_path(tune_file);
  FILE *out_file = fopen(filename, "w");

  if (out_file == NULL) {
    printf("Error: Could not open file: %s", filename);
    return;
  } 

  fprintf(out_file, content, sizeof(content));
  //printf("python: %s\n",content);

  fclose(out_file); 
}

void update_params() {
  if (debug) { printf("update_params\n"); }
  // update the params file with the current values
  char content[MAX_FILE_BYTES] = "";
  char preset_label[128] = "PRESET";

  for (int k=0; k < num_presets; k++) {
    char line[128] = "";
    int num_params = num_preset_params[k];
    snprintf(line,sizeof(line),"%s=%s\n",preset_label,preset_labels[k]);
    strcat(content,line);
    for (int i=0; i < num_params; i++) {
      int num_elements = param_value_count[k][i];
      char temp[128] = "";
      for (int j=0; j < num_elements; j++) {
        char value_str[50];
        snprintf(value_str,sizeof(value_str),"%.6f",angles[k][i][j]);
        strcat(temp, value_str);
        if (j < num_elements-1) {
          strcat(temp, ",");
        }
        //printf("%d, %d: %.36\n", i, j, angles[k][i][j]);
      }
      char line[128] = "";
      snprintf(line,sizeof(line),"%s=[%s]\n",param_labels[i],temp);
      //printf("line: %s\n",line);
      strcat(content,line);
    }
  }
  //printf("Contents:\n\n");
  //printf("%s", content);

  char *filename = get_full_path(params_file);
  FILE *out_file = fopen(filename, "w");

  if (out_file == NULL) {
    printf("Error: Could not open file: %s", filename);
    return;
  } 

  fprintf(out_file, content, sizeof(content));
  //printf("python: %s\n",content);

  fclose(out_file); 

  write_current_params();
} 

void init_tuning(UIState *s) {
  if (debug) { printf("init_tuning\n"); }
  if (init_tune) {
    return; 
  } 

  left_arrow_icon = nvgCreateImage(s->vg, "../assets/left_arrow.png", 1);
  right_arrow_icon = nvgCreateImage(s->vg, "../assets/right_arrow.png", 1);
  tune_icon = nvgCreateImage(s->vg, "../assets/tune_icon.png", 1);

  read_file_values(params_file);

  init_tune = true;

  increase_button = (ui_element){
    .name = "increase_button",
    .pos_x = (1920/2)+40,
    .pos_y = 1080-240,
    .width = 120,
    .height = 120
  };

  decrease_button = (ui_element){
    .name = "decrease_button",
    .pos_x = (1920/2)-140,
    .pos_y = 1080-240,
    .width = 120,
    .height = 120
  };

  tune_values = (ui_element){
    .name = "tune_values",
    .pos_x = (1920/2)-650,
    .pos_y = 1080-490,
    .width = 1350,
    .height = 220
  };

  property_button = (ui_element){
    .name = "property_button",
    .pos_x = (1920/2)-600,
    .pos_y = 1080-400,
    .width = 390,
    .height = 95
  };

  //for (int j=0; j < num_lines; j++) {
  for (int i=0; i < 3; i++) {
    property_buttons[i] = malloc(sizeof(ui_element) * 3);
    property_buttons[0][i] = property_button;
    property_buttons[0][i].pos_x += i*(property_button.width+45);
    char valuename[50];
    snprintf(valuename,sizeof(valuename),"value_%d",i);
    snprintf(property_buttons[0][i].name,sizeof(property_buttons[0][i].name),"%s",valuename);
  }

  step_text = (ui_element){
    .name = "step_text",
    .pos_x = (1920/2)-580,
    .pos_y = 1080-635,
    .width = 450,
    .height = 125
  };

  step_decrease = (ui_element){
    .name = "step_decrease",
    .pos_x = (1920/2)-90,
    .pos_y = 1080-640,
    .width = 120,
    .height = 120
  };

  step_increase = (ui_element){
    .name = "step_increase",
    .pos_x = step_decrease.pos_x + step_decrease.width + 15,
    .pos_y = 1080-640,
    .width = 120,
    .height = 120
  };

  left_arrow_button = (ui_element) {
    .name = "left_arrow",
    .pos_x = (1920/2)-400,
    .pos_y = 1080-240,
    .width = 85,
    .height = 85
  };

  right_arrow_button = (ui_element) {
    .name = "right_arrow",
    .pos_x = (1920/2)+335,
    .pos_y = 1080-240,
    .width = 85,
    .height = 85
  };

  tune_button = (ui_element) {
    .name = "tune_button",
    .pos_x = (1920/2)-610,
    .pos_y = 1080-240,
    .width = 120,
    .height = 120
  };

  preset_button = (ui_element) {
    .name = "preset_button",
    .pos_x = (1920/2)-630,
    .pos_y = 1080-490,
    .width = 310,
    .height = 85
  };

}

void draw_error( UIState *s) {
  //printf("No file!\n");

  nvgBeginPath(s->vg);
    nvgRoundedRect(s->vg,350,400,1350,380,40);
    nvgStrokeColor(s->vg, nvgRGBA(255,255,255,80));
    nvgStrokeWidth(s->vg, 6);
    nvgStroke(s->vg);
    nvgFillColor(s->vg, nvgRGBA(0, 0, 0, 100));
    nvgFill(s->vg);
    nvgTextAlign(s->vg, NVG_ALIGN_LEFT | NVG_ALIGN_BASELINE);
    nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 200));
    nvgFontSize(s->vg, 70);
    nvgText(s->vg,400,500,"ERROR - File not found:",NULL);
    nvgText(s->vg,400,600,"  /sdcard/tuning/params.txt",NULL);
    nvgText(s->vg,400,700,"Please check that this file exists",NULL);
}

void screen_draw_tuning(UIState *s) {
  if (debug) { printf("screen_draw_tuning\n"); }

  float alpha = 1.0f;
  NVGpaint imgPaint;

  if (!init_tune) {
    return;
  }

  if (status == ERROR_NO_FILE && tune_enabled) {
    draw_error(s);
  }
  
  if (tune_enabled && status != ERROR_NO_FILE) {

  if (current_property != -1) {

    // increase button
    nvgBeginPath(s->vg);
    nvgRoundedRect(s->vg, increase_button.pos_x, increase_button.pos_y, increase_button.width, increase_button.height, 100);
    nvgStrokeColor(s->vg, nvgRGBA(255,255,255,80));
    nvgStrokeWidth(s->vg, 6);
    nvgStroke(s->vg);
    nvgFillColor(s->vg, nvgRGBA(0, 0, 0, 100));
    if (current_button == BTN_INCREASE) {
      nvgFillColor(s->vg, nvgRGBA(0, 162, 255, 100));
    }
    nvgFill(s->vg);

    nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 200));
    nvgFontSize(s->vg, 120);
    nvgText(s->vg,increase_button.pos_x+30,increase_button.pos_y+90,"+",NULL);

    // decrease button
    nvgBeginPath(s->vg);
    nvgRoundedRect(s->vg, decrease_button.pos_x, decrease_button.pos_y, decrease_button.width, decrease_button.height, 100);
    nvgStrokeColor(s->vg, nvgRGBA(255,255,255,80));
    nvgStrokeWidth(s->vg, 6);
    nvgStroke(s->vg);
    nvgFillColor(s->vg, nvgRGBA(0, 0, 0, 100));
    if (current_button == BTN_DECREASE) {
      nvgFillColor(s->vg, nvgRGBA(0, 162, 255, 100));
    }
    nvgFill(s->vg);

    nvgFontSize(s->vg, 120);
    nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 200));
    nvgText(s->vg,decrease_button.pos_x+30,decrease_button.pos_y+90,"-",NULL);
  }

  // Draw current tune values
  nvgBeginPath(s->vg);
  nvgTextAlign(s->vg, NVG_ALIGN_LEFT | NVG_ALIGN_BASELINE);
  nvgRoundedRect(s->vg, tune_values.pos_x, tune_values.pos_y, tune_values.width, tune_values.height, 10);
  nvgStrokeColor(s->vg, nvgRGBA(255,255,255,80));
  nvgStrokeWidth(s->vg, 6);
  nvgStroke(s->vg);
  nvgFillColor(s->vg, nvgRGBA(0, 0, 0, 100));
  nvgFill(s->vg);

  nvgFontSize(s->vg, 65);
  nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 200));

  // Preset button
  char label[128];
  nvgBeginPath(s->vg);
  nvgRoundedRect(s->vg, preset_button.pos_x, preset_button.pos_y, preset_button.width, preset_button.height, 10);
  /*  
  nvgStrokeColor(s->vg, nvgRGBA(255,255,255,80));
  nvgStrokeWidth(s->vg, 6);
  nvgStroke(s->vg);
  nvgFillColor(s->vg, nvgRGBA(0, 0, 0, 100));
  nvgFill(s->vg);
  */
  snprintf(label,sizeof(label),"%s:", preset_labels[preset]);
  nvgFillColor(s->vg, nvgRGBA(0, 255, 255, 200));
  nvgText(s->vg,tune_values.pos_x+20,tune_values.pos_y+50,label,NULL);

  nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 200));
  nvgText(s->vg,tune_values.pos_x+340+(10*numChars(preset_labels[preset])),tune_values.pos_y+50,param_labels[param_index],NULL);

  int pos_x;
  int pos_y;
  char thisValue[12];
  for (int i=0; i < param_value_count[preset][param_index]; i++) {
    int pos_x = 0;
    switch(i) {
      case 0:
        pos_x = 30;
        break;
      case 1:
        pos_x = 470;
        break;
      case 2:
        pos_x = 900;
        break;
      default:
        break;
    }
    if (angles[preset][param_index][i] < 1) {
      snprintf(thisValue,sizeof(thisValue),"%.6f",angles[preset][param_index][i]);
    }
    else {
      snprintf(thisValue,sizeof(thisValue),"%.1f",angles[preset][param_index][i]);
    }
    nvgFontSize(s->vg, 70);
    nvgText(s->vg,tune_values.pos_x+pos_x+60,tune_values.pos_y+160,thisValue,NULL);
  }

  // Render property buttons based on the number of elements in the params 
  int num_params = param_value_count[preset][param_index];
  for (int i=0; i < num_params; i++) {
    nvgBeginPath(s->vg);
    nvgRoundedRect(s->vg, property_buttons[0][i].pos_x, property_buttons[0][i].pos_y, property_buttons[0][i].width, property_buttons[0][i].height, 15);
    if (i == current_property) {
      nvgStrokeColor(s->vg, nvgRGBA(0,162,255,200));
    }
    else {
      nvgStrokeColor(s->vg, nvgRGBA(255,255,255,200));
    }
    nvgStrokeWidth(s->vg, 9);
    nvgStroke(s->vg);
  }

  // Step increment buttons
  nvgBeginPath(s->vg);
  nvgRoundedRect(s->vg, step_text.pos_x, step_text.pos_y, step_text.width, step_text.height, 15);
  nvgStrokeColor(s->vg, nvgRGBA(255,255,255,80));
  nvgStrokeWidth(s->vg, 6);
  nvgStroke(s->vg);
  nvgFillColor(s->vg, nvgRGBA(0, 0, 0, 100));
  nvgFill(s->vg);

  nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 200));
  if (current_button == BTN_STEP) {
    nvgFillColor(s->vg, nvgRGBA(0, 162, 255, 100));
  }
  char step_str[12];
  if (step >= 1) {
    snprintf(step_str,sizeof(step_str),"%.1f",step);
  }
  else {
    snprintf(step_str,sizeof(step_str),"%.6f",step);
  }
  nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_BASELINE);
  nvgFontSize(s->vg, 70);
  nvgText(s->vg,step_text.pos_x+120,step_increase.pos_y+55,"Steps",NULL);
  nvgFontSize(s->vg, 80);
  nvgText(s->vg,step_text.pos_x+220,step_increase.pos_y+112,step_str,NULL);

  nvgTextAlign(s->vg, NVG_ALIGN_LEFT | NVG_ALIGN_BASELINE);
  nvgBeginPath(s->vg);
  nvgRoundedRect(s->vg, step_increase.pos_x, step_increase.pos_y, step_increase.width, step_increase.height, 100);
  nvgStrokeColor(s->vg, nvgRGBA(255,255,255,80));
  nvgStrokeWidth(s->vg, 6);
  nvgStroke(s->vg);
  nvgFillColor(s->vg, nvgRGBA(0, 0, 0, 100));
  if (current_button == BTN_STEP_INCREASE) {
    nvgFillColor(s->vg, nvgRGBA(0, 162, 255, 100));
  }
  nvgFill(s->vg);

  nvgFontSize(s->vg, 120);
  nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 200));
  nvgText(s->vg,step_increase.pos_x+25,step_increase.pos_y+90,"+",NULL);

  nvgBeginPath(s->vg);
  nvgRoundedRect(s->vg, step_decrease.pos_x, step_decrease.pos_y, step_decrease.width, step_decrease.height, 100);
  nvgStrokeColor(s->vg, nvgRGBA(255,255,255,80));
  nvgStrokeWidth(s->vg, 6);
  nvgStroke(s->vg);
  nvgFillColor(s->vg, nvgRGBA(0, 0, 0, 100));
  if (current_button == BTN_STEP_DECREASE) {
    nvgFillColor(s->vg, nvgRGBA(0, 162, 255, 100));
  }
  nvgFill(s->vg);

  nvgFontSize(s->vg, 150);
  nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 200));
  nvgText(s->vg,step_decrease.pos_x+25,step_increase.pos_y+100,"-",NULL);

  // left arrow
  nvgBeginPath(s->vg);
  nvgRoundedRect(s->vg, left_arrow_button.pos_x, left_arrow_button.pos_y, left_arrow_button.width+35, left_arrow_button.height+35, 100);
  nvgStrokeColor(s->vg, nvgRGBA(255,255,255,80));
  nvgStrokeWidth(s->vg, 6);
  nvgStroke(s->vg);
  nvgFillColor(s->vg, nvgRGBA(0, 0, 0, 100));
  if (current_button == BTN_LEFT_ARROW) {
    nvgFillColor(s->vg, nvgRGBA(0, 162, 255, 100));
  }
  nvgFill(s->vg);

  nvgBeginPath(s->vg);
    imgPaint = nvgImagePattern(s->vg, left_arrow_button.pos_x+15, left_arrow_button.pos_y+15, left_arrow_button.width, left_arrow_button.height, 0, left_arrow_icon, 1.0f);
    nvgRoundedRect(s->vg, left_arrow_button.pos_x+15, left_arrow_button.pos_y+15, left_arrow_button.width, left_arrow_button.height, 100);
    nvgFillPaint(s->vg, imgPaint);
    nvgFill(s->vg);

  // right arrow
  nvgBeginPath(s->vg);
  nvgRoundedRect(s->vg, right_arrow_button.pos_x, right_arrow_button.pos_y, right_arrow_button.width+35, right_arrow_button.height+35, 100);
  nvgStrokeColor(s->vg, nvgRGBA(255,255,255,80));
  nvgStrokeWidth(s->vg, 6);
  nvgStroke(s->vg);
  nvgFillColor(s->vg, nvgRGBA(0, 0, 0, 100));
  if (current_button == BTN_RIGHT_ARROW) {
    nvgFillColor(s->vg, nvgRGBA(0, 162, 255, 100));
  }
  nvgFill(s->vg);

  nvgBeginPath(s->vg);
    imgPaint = nvgImagePattern(s->vg, right_arrow_button.pos_x+15, right_arrow_button.pos_y+15, right_arrow_button.width, right_arrow_button.height, 0, right_arrow_icon, 1.0f);
    nvgRoundedRect(s->vg, right_arrow_button.pos_x+15, right_arrow_button.pos_y+15, right_arrow_button.width, right_arrow_button.height, 100);
    nvgFillPaint(s->vg, imgPaint);
    nvgFill(s->vg);

  }

  if (s->vision_connected) {
    //printf("vision connected\n");

    // tune button
    nvgBeginPath(s->vg);
    nvgRoundedRect(s->vg, tune_button.pos_x, tune_button.pos_y, tune_button.width+35, tune_button.height+35, 100);
    nvgStrokeColor(s->vg, nvgRGBA(255,255,255,80));
    nvgStrokeWidth(s->vg, 6);
    nvgStroke(s->vg);
    nvgFillColor(s->vg, nvgRGBA(0, 0, 0, 100));
    if (current_button == BTN_TUNE) {
      nvgFillColor(s->vg, nvgRGBA(0, 162, 255, 100));
    }
    nvgFill(s->vg);

    nvgBeginPath(s->vg);
      if (!tune_enabled) {
        alpha = 0.3f;
      }
      imgPaint = nvgImagePattern(s->vg, tune_button.pos_x+15, tune_button.pos_y+15, tune_button.width, tune_button.height, 0, tune_icon, alpha);
      nvgRoundedRect(s->vg, tune_button.pos_x+15, tune_button.pos_y+15, tune_button.width, tune_button.height, 100);
      nvgFillPaint(s->vg, imgPaint);
      nvgFill(s->vg);
  }

}

void next_param(int direction) {
  if (debug) { printf("next_param\n"); }
  // Load the next parameter to adjust
  param_index += direction;

  int num_params = num_preset_params[preset];
  
  if (param_index >= num_params) {
    param_index = 0;
  }
  else if (param_index < 0) {
    param_index = num_params-1;
  }
}

void toggle_tune() {
  if (debug) { printf("toggle_tune\n"); }
  if (!tune_enabled) {
    tune_enabled = true;
  }
  else {
    tune_enabled = false;
  }
}

bool ui_element_clicked(int touch_x, int touch_y, ui_element el) {
  int padding = 20; // Increase touch region by this many pixels so it's not as precise
  if (touch_x >= (el.pos_x-padding) && touch_x <= (int)(el.pos_x+el.width+padding)) {
    if (touch_y >= (el.pos_y-padding) && touch_y <= (int)(el.pos_y+el.height+padding)) {
      return true;
    }
  }
  return false;
}

void toggle_presets() {
  preset++;
  if (preset >= num_presets) {
    preset = 0;
  }
  write_current_params();
}

void toggle_step() {
  //printf("step_toggle: %d\n",step_toggle);
  step_toggle++;
  if (step_toggle == 0) {
    delta_step = 0.000001;
  }
  else if (step_toggle == 1) {
    delta_step = 0.00001;
  }
  else if (step_toggle == 2) {
    delta_step = 0.0001;
  }
  else if (step_toggle == 3) {
    delta_step = 0.001;
  }
  else if (step_toggle == 4) {
    delta_step = 0.01;
  }
  else if (step_toggle == 5) {
    delta_step = 0.1;
  }
  else if (step_toggle == 6) {
    delta_step = 1;
  }
  else if (step_toggle == 7) {
    delta_step = 5;
  }
  else {
    step_toggle = 0;
    delta_step = 0.000001;
  }
  step = delta_step;
}

void tuning( UIState *s, int touch_x, int touch_y, int key_up) {
  init_tuning(s);

  bool hold = false; // Track whether we're pressing and holding

  if (debug) { printf("tuning\n"); }

  if (status == ERROR_NO_FILE) {
    return;
  }

  if (ui_element_clicked(touch_x,touch_y,tune_button) && key_up) {
    toggle_tune();
    current_button = BTN_TUNE;
  }

  screen_draw_tuning(s);

/*
  if (touch_x > 0) {
    printf("touch: %d, %d keyup: %d\n", touch_x, touch_y, key_up);
  }
*/

  if (touch_x >= 0 && touch_y >= 0 && last_x == touch_x && last_y == touch_y && touch_frame >= touch_frame_delay) {
    hold = true;
  }

  if (current_button != BTN_NONE && ((last_x != touch_x && last_y != touch_y) || (touch_x < 0 && touch_y < 0))) {
    // Delay turning off the current button highlight
    frame_num++;
    if (frame_num >= frame_delay && !hold) {
      current_button = BTN_NONE;
      frame_num = 0;
    }
  }

  touch_frame++;
  if (touch_frame > touch_frame_delay) {
    touch_frame = 0;
  }

  last_x = touch_x;
  last_y = touch_y;

  if (ui_element_clicked(touch_x,touch_y,increase_button) && current_property != -1 && hold) {
    //printf("increase button clicked\n");
    //printf("param_index: %d\n",param_index);
    //printf("current_prop: %d\n",current_property);
    angles[preset][param_index][current_property] += step;
    update_params();
    current_button = BTN_INCREASE;
  }
  else if (ui_element_clicked(touch_x,touch_y,decrease_button) && current_property != -1 && hold) {
    //printf("decrease button clicked\n");
    angles[preset][param_index][current_property] -= step;
    update_params();
    current_button = BTN_DECREASE;
  }
  else if (ui_element_clicked(touch_x,touch_y,step_increase) && hold) {
    //printf("step increase button clicked\n");
    step += delta_step;
    current_button = BTN_STEP_INCREASE;
  }
  else if (ui_element_clicked(touch_x,touch_y,step_decrease) && hold) {
    //printf("step decrease button clicked\n");
    step -= delta_step;
    current_button = BTN_STEP_DECREASE;
  }
  else if (ui_element_clicked(touch_x,touch_y,left_arrow_button) && key_up) {
    next_param(-1);
    current_button = BTN_LEFT_ARROW;
  }
  else if (ui_element_clicked(touch_x,touch_y,right_arrow_button) && key_up) {
    next_param(1);
    current_button = BTN_RIGHT_ARROW;
  }
  else if (ui_element_clicked(touch_x,touch_y,step_text) && key_up) {
    toggle_step();
    current_button = BTN_STEP;
  }
  else if (ui_element_clicked(touch_x,touch_y,preset_button) && key_up) {
    toggle_presets();
  }
  else {
    for (int i=0; i < param_value_count[preset][param_index]; i++) {
      if (ui_element_clicked(touch_x,touch_y,property_buttons[0][i]) && key_up) {
        if (current_property == i) {
          // Toggle off
          current_property = -1;
        }
        else {
          // Toggle on
          current_property = i;
        }
        //printf("param element %i clicked\n",i);
      }
    }
  }

}

