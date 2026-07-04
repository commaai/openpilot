/****************************************************************************
** Meta object code from reading C++ file 'replaystream.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.13)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "replaystream.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'replaystream.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.13. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_ReplayStream_t {
    QByteArrayData data[5];
    char stringdata0[57];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ReplayStream_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ReplayStream_t qt_meta_stringdata_ReplayStream = {
    {
QT_MOC_LITERAL(0, 0, 12), // "ReplayStream"
QT_MOC_LITERAL(1, 13, 10), // "qLogLoaded"
QT_MOC_LITERAL(2, 24, 0), // ""
QT_MOC_LITERAL(3, 25, 26), // "std::shared_ptr<LogReader>"
QT_MOC_LITERAL(4, 52, 4) // "qlog"

    },
    "ReplayStream\0qLogLoaded\0\0"
    "std::shared_ptr<LogReader>\0qlog"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ReplayStream[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   19,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Void, 0x80000000 | 3,    4,

       0        // eod
};

void ReplayStream::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<ReplayStream *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->qLogLoaded((*reinterpret_cast< std::shared_ptr<LogReader>(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 0:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< std::shared_ptr<LogReader> >(); break;
            }
            break;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (ReplayStream::*)(std::shared_ptr<LogReader> );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ReplayStream::qLogLoaded)) {
                *result = 0;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject ReplayStream::staticMetaObject = { {
    QMetaObject::SuperData::link<AbstractStream::staticMetaObject>(),
    qt_meta_stringdata_ReplayStream.data,
    qt_meta_data_ReplayStream,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *ReplayStream::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ReplayStream::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_ReplayStream.stringdata0))
        return static_cast<void*>(this);
    return AbstractStream::qt_metacast(_clname);
}

int ReplayStream::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = AbstractStream::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    }
    return _id;
}

// SIGNAL 0
void ReplayStream::qLogLoaded(std::shared_ptr<LogReader> _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
struct qt_meta_stringdata_OpenReplayWidget_t {
    QByteArrayData data[1];
    char stringdata0[17];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_OpenReplayWidget_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_OpenReplayWidget_t qt_meta_stringdata_OpenReplayWidget = {
    {
QT_MOC_LITERAL(0, 0, 16) // "OpenReplayWidget"

    },
    "OpenReplayWidget"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_OpenReplayWidget[] = {

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

void OpenReplayWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    (void)_o;
    (void)_id;
    (void)_c;
    (void)_a;
}

QT_INIT_METAOBJECT const QMetaObject OpenReplayWidget::staticMetaObject = { {
    QMetaObject::SuperData::link<AbstractOpenStreamWidget::staticMetaObject>(),
    qt_meta_stringdata_OpenReplayWidget.data,
    qt_meta_data_OpenReplayWidget,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *OpenReplayWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *OpenReplayWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_OpenReplayWidget.stringdata0))
        return static_cast<void*>(this);
    return AbstractOpenStreamWidget::qt_metacast(_clname);
}

int OpenReplayWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = AbstractOpenStreamWidget::qt_metacall(_c, _id, _a);
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
