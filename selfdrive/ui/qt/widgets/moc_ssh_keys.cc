/****************************************************************************
** Meta object code from reading C++ file 'ssh_keys.hpp'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.8)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "ssh_keys.hpp"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ssh_keys.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.8. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_SshToggle_t {
    QByteArrayData data[1];
    char stringdata0[10];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_SshToggle_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_SshToggle_t qt_meta_stringdata_SshToggle = {
    {
QT_MOC_LITERAL(0, 0, 9) // "SshToggle"

    },
    "SshToggle"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_SshToggle[] = {

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

void SshToggle::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject SshToggle::staticMetaObject = { {
    &ToggleControl::staticMetaObject,
    qt_meta_stringdata_SshToggle.data,
    qt_meta_data_SshToggle,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *SshToggle::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *SshToggle::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_SshToggle.stringdata0))
        return static_cast<void*>(this);
    return ToggleControl::qt_metacast(_clname);
}

int SshToggle::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = ToggleControl::qt_metacall(_c, _id, _a);
    return _id;
}
struct qt_meta_stringdata_SshControl_t {
    QByteArrayData data[6];
    char stringdata0[61];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_SshControl_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_SshControl_t qt_meta_stringdata_SshControl = {
    {
QT_MOC_LITERAL(0, 0, 10), // "SshControl"
QT_MOC_LITERAL(1, 11, 14), // "failedResponse"
QT_MOC_LITERAL(2, 26, 0), // ""
QT_MOC_LITERAL(3, 27, 11), // "errorString"
QT_MOC_LITERAL(4, 39, 7), // "timeout"
QT_MOC_LITERAL(5, 47, 13) // "parseResponse"

    },
    "SshControl\0failedResponse\0\0errorString\0"
    "timeout\0parseResponse"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_SshControl[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   29,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       4,    0,   32,    2, 0x08 /* Private */,
       5,    0,   33,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void, QMetaType::QString,    3,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void SshControl::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<SshControl *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->failedResponse((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 1: _t->timeout(); break;
        case 2: _t->parseResponse(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (SshControl::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&SshControl::failedResponse)) {
                *result = 0;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject SshControl::staticMetaObject = { {
    &AbstractControl::staticMetaObject,
    qt_meta_stringdata_SshControl.data,
    qt_meta_data_SshControl,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *SshControl::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *SshControl::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_SshControl.stringdata0))
        return static_cast<void*>(this);
    return AbstractControl::qt_metacast(_clname);
}

int SshControl::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = AbstractControl::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 3)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 3)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 3;
    }
    return _id;
}

// SIGNAL 0
void SshControl::failedResponse(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
