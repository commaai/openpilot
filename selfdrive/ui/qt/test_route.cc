#include "route.hpp"

#include <QApplication>

int main(int argc, char* argv[]){
  QApplication a(argc, argv);

  Route *test = new  Route("0982d79ebb0de295|2021-01-17--17-13-08");

  QObject::connect(test, &Route::doneParsing, [=](){
    auto logs = test->log_paths();
    for(auto &log : logs){
      qDebug() << log;
    }
  });

  return a.exec();
}
