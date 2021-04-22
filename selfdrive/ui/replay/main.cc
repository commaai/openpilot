#include <QApplication>

#include "replay.hpp"

int main(int argc, char *argv[]){
  QApplication a(argc, argv);

  QString route(argv[1]);
  route = route.replace("|", "/");
  if (route == "") {
    printf("Usage: ./replay \"route\"\n");
    return 1;
  }

  int seek = QString(argv[2]).toInt();

  Replay *replay = new Replay(route, seek);
  replay->stream(seek);

  return a.exec();
}
