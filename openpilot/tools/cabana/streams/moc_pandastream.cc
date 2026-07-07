/****************************************************************************
** Meta object code from reading C++ file 'pandastream.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.13)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "pandastream.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'pandastream.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.13. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_PandaStream_t {
    QByteArrayData data[1];
    char stringdata0[12];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_PandaStream_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_PandaStream_t qt_meta_stringdata_PandaStream = {
    {
QT_MOC_LITERAL(0, 0, 11) // "PandaStream"

    },
    "PandaStream"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_PandaStream[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void PandaStream::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    (void)_o;
    (void)_id;
    (void)_c;
    (void)_a;
}

QT_INIT_METAOBJECT const QMetaObject PandaStream::staticMetaObject = { {
    QMetaObject::SuperData::link<LiveStream::staticMetaObject>(),
    qt_meta_stringdata_PandaStream.data,
    qt_meta_data_PandaStream,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *PandaStream::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *PandaStream::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_PandaStream.stringdata0))
        return static_cast<void*>(this);
    return LiveStream::qt_metacast(_clname);
}

int PandaStream::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = LiveStream::qt_metacall(_c, _id, _a);
    return _id;
}
struct qt_meta_stringdata_OpenPandaWidget_t {
    QByteArrayData data[1];
    char stringdata0[16];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_OpenPandaWidget_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_OpenPandaWidget_t qt_meta_stringdata_OpenPandaWidget = {
    {
QT_MOC_LITERAL(0, 0, 15) // "OpenPandaWidget"

    },
    "OpenPandaWidget"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_OpenPandaWidget[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void OpenPandaWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    (void)_o;
    (void)_id;
    (void)_c;
    (void)_a;
}

QT_INIT_METAOBJECT const QMetaObject OpenPandaWidget::staticMetaObject = { {
    QMetaObject::SuperData::link<AbstractOpenStreamWidget::staticMetaObject>(),
    qt_meta_stringdata_OpenPandaWidget.data,
    qt_meta_data_OpenPandaWidget,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *OpenPandaWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *OpenPandaWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_OpenPandaWidget.stringdata0))
        return static_cast<void*>(this);
    return AbstractOpenStreamWidget::qt_metacast(_clname);
}

int OpenPandaWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = AbstractOpenStreamWidget::qt_metacall(_c, _id, _a);
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
