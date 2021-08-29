/****************************************************************************
** Meta object code from reading C++ file 'api.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.8)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "api.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'api.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.8. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_HttpRequest_t {
    QByteArrayData data[11];
    char stringdata0[133];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_HttpRequest_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_HttpRequest_t qt_meta_stringdata_HttpRequest = {
    {
QT_MOC_LITERAL(0, 0, 11), // "HttpRequest"
QT_MOC_LITERAL(1, 12, 11), // "requestDone"
QT_MOC_LITERAL(2, 24, 0), // ""
QT_MOC_LITERAL(3, 25, 7), // "success"
QT_MOC_LITERAL(4, 33, 16), // "receivedResponse"
QT_MOC_LITERAL(5, 50, 8), // "response"
QT_MOC_LITERAL(6, 59, 14), // "failedResponse"
QT_MOC_LITERAL(7, 74, 11), // "errorString"
QT_MOC_LITERAL(8, 86, 15), // "timeoutResponse"
QT_MOC_LITERAL(9, 102, 14), // "requestTimeout"
QT_MOC_LITERAL(10, 117, 15) // "requestFinished"

    },
    "HttpRequest\0requestDone\0\0success\0"
    "receivedResponse\0response\0failedResponse\0"
    "errorString\0timeoutResponse\0requestTimeout\0"
    "requestFinished"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_HttpRequest[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       6,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       4,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   44,    2, 0x06 /* Public */,
       4,    1,   47,    2, 0x06 /* Public */,
       6,    1,   50,    2, 0x06 /* Public */,
       8,    1,   53,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       9,    0,   56,    2, 0x08 /* Private */,
      10,    0,   57,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void, QMetaType::Bool,    3,
    QMetaType::Void, QMetaType::QString,    5,
    QMetaType::Void, QMetaType::QString,    7,
    QMetaType::Void, QMetaType::QString,    7,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void HttpRequest::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<HttpRequest *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->requestDone((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 1: _t->receivedResponse((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 2: _t->failedResponse((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 3: _t->timeoutResponse((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 4: _t->requestTimeout(); break;
        case 5: _t->requestFinished(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (HttpRequest::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&HttpRequest::requestDone)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (HttpRequest::*)(const QString & );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&HttpRequest::receivedResponse)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (HttpRequest::*)(const QString & );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&HttpRequest::failedResponse)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (HttpRequest::*)(const QString & );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&HttpRequest::timeoutResponse)) {
                *result = 3;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject HttpRequest::staticMetaObject = { {
    &QObject::staticMetaObject,
    qt_meta_stringdata_HttpRequest.data,
    qt_meta_data_HttpRequest,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *HttpRequest::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *HttpRequest::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_HttpRequest.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int HttpRequest::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 6)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 6;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 6)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 6;
    }
    return _id;
}

// SIGNAL 0
void HttpRequest::requestDone(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void HttpRequest::receivedResponse(const QString & _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void HttpRequest::failedResponse(const QString & _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void HttpRequest::timeoutResponse(const QString & _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
