#import "ui.h"

void bb_ui_draw_car( UIState*);
void bb_draw_lane_fill ( UIState*);
long bb_currentTimeInMilis();
int bb_get_status( UIState *);
int bb_ui_draw_measure( UIState *,  const char*, const char*, const char*, 
		int, int, int, NVGcolor, NVGcolor, NVGcolor, int, int, int );
bool bb_handle_ui_touch( UIState *, int, int);
int bb_get_button_status( UIState *, char *);
void bb_draw_button( UIState *, int);
void bb_draw_buttons( UIState *);
void bb_ui_draw_custom_alert( UIState *);
void bb_ui_draw_measures_left( UIState *, int , int , int  );
void bb_ui_draw_measures_right( UIState *, int , int , int  );
void ui_draw_vision_grid( UIState *);
void bb_ui_draw_logo( UIState *);
void bb_ui_draw_UI( UIState *);
void bb_ui_init( UIState *);
void bb_ui_play_sound( UIState *, int );
void bb_ui_set_car( UIState *, char *, char *);
void  bb_ui_poll_update(  UIState *);
void bb_ui_draw_vision_alert( UIState *, int , int ,
                                  const char* , const char* );


