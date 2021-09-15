#include <QApplication>

#include "selfdrive/ui/replay/replay.h"

int main(int argc, char *argv[]){
  QApplication a(argc, argv);

  QString route(argv[1]);
  if (route == "") {
    printf("Usage: ./replay \"route\"\n");
    printf("  For a public demo route, use '3533c53bb29502d1|2019-12-10--01-13-27'\n");
    return 1;
  }

  Replay *replay = new Replay(route);
  replay->start();

  return a.exec();
}
