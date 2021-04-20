#include "route.hpp"

Route::Route(QString route_name_) : route_name(route_name_.replace("|", "/")){
  api = new CommaApi;
}

void Route::_get_segments_remote(){
}
